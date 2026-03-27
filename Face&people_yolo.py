"""
Thermal Video - Face & People Detection
========================================
Detects people (full body) and faces in thermal footage (.mp4, .avi, etc.)
Outputs an annotated video with bounding boxes and people/face counts.

Requirements:
    pip install ultralytics opencv-python cvzone numpy

Usage:
    python thermal_detection.py
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# ── Try importing YOLO (ultralytics) ─────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Run:  pip install ultralytics")
    print("[INFO] Falling back to OpenCV HOG people detector + Haar face detector.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  THERMAL PRE-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_thermal(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint16:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTOR CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class YOLODetector:
    PERSON_ID = 0

    def __init__(self, conf: float = 0.35):
        print("[INFO] Loading YOLOv8n model …")
        self.person_model = YOLO("yolov8n.pt")
        try:
            self.face_model = YOLO("yolov8n-face.pt")
            self.face_model_available = True
            print("[INFO] YOLOv8-face model loaded.")
        except Exception:
            self.face_model_available = False
            print("[WARN] yolov8n-face.pt not found; using Haar fallback.")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self.conf = conf

    def detect(self, frame: np.ndarray):
        people, faces = [], []

        results = self.person_model(frame, conf=self.conf, classes=[self.PERSON_ID], verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people.append((x1, y1, x2, y2, float(box.conf[0])))

        if self.face_model_available:
            face_results = self.face_model(frame, conf=self.conf, verbose=False)[0]
            for box in face_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2, float(box.conf[0])))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in detected:
                faces.append((x, y, x + w, y + h, 1.0))

        return people, faces


class FallbackDetector:
    def __init__(self, conf: float = 0.35):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.conf = conf
        print("[INFO] Using OpenCV HOG people detector + Haar face cascade.")

    def detect(self, frame: np.ndarray):
        people, faces = [], []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (x, y, w, h), wt in zip(rects, weights):
            if wt[0] > 0.3:
                people.append((x, y, x + w, y + h, float(wt[0])))

        detected = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for (x, y, w, h) in detected:
            faces.append((x, y, x + w, y + h, 1.0))

        return people, faces


# ═════════════════════════════════════════════════════════════════════════════
#  ANNOTATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

COLOR_PERSON = (0, 255, 120)
COLOR_FACE   = (0, 180, 255)
COLOR_HUD    = (20, 20, 20)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS  = 2

def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
    corner = 12
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, THICKNESS + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, THICKNESS + 1)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, THICKNESS + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, THICKNESS + 1)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, THICKNESS + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, THICKNESS + 1)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, THICKNESS + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, THICKNESS + 1)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
    badge_y1 = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

def draw_hud(frame, n_people, n_faces, fps, frame_no):
    h, w = frame.shape[:2]
    lines = [f"PEOPLE : {n_people}", f"FACES  : {n_faces}", f"FPS    : {fps:.1f}", f"FRAME  : {frame_no}"]
    pad, lh = 8, 22
    box_h = len(lines) * lh + pad * 2
    box_w = 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), COLOR_HUD, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, txt in enumerate(lines):
        color = COLOR_PERSON if "PEOPLE" in txt else COLOR_FACE if "FACES" in txt else (200, 200, 200)
        cv2.putText(frame, txt, (18, 10 + pad + lh * (i + 1) - 4), FONT, 0.55, color, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run(input_path: str, output_path: str, conf: float, show: bool):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_IN = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[INFO] Input  : {input_path}")
    print(f"[INFO] Size   : {W}×{H}  |  FPS: {FPS_IN:.2f}  |  Frames: {TOTAL}")
    print(f"[INFO] Output : {output_path}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS_IN, (W, H))

    detector = YOLODetector(conf) if YOLO_AVAILABLE else FallbackDetector(conf)

    frame_no  = 0
    t_prev    = time.time()
    fps_display = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_no += 1
        frame = preprocess_thermal(raw_frame)
        people, faces = detector.detect(frame)

        for i, (x1, y1, x2, y2, conf_v) in enumerate(people):
            draw_box(frame, x1, y1, x2, y2, f"Person {i+1}  {conf_v:.0%}", COLOR_PERSON)
        for i, (x1, y1, x2, y2, conf_v) in enumerate(faces):
            draw_box(frame, x1, y1, x2, y2, f"Face {i+1}  {conf_v:.0%}", COLOR_FACE)

        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        draw_hud(frame, len(people), len(faces), fps_display, frame_no)
        writer.write(frame)

        if show:
            cv2.imshow("Thermal Detection  [Q to quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User interrupted.")
                break

        if frame_no % 50 == 0:
            pct = frame_no / max(TOTAL, 1) * 100
            print(f"  … {frame_no}/{TOTAL} frames ({pct:.1f}%)  people={len(people)}  faces={len(faces)}", flush=True)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"\n[DONE] Annotated video saved → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  HARD-CODED VIDEO ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Hardcoded video path
    input_video = r"C:/Users/DELL/Desktop/FYP/YOLO/Inputs/Outdoor01.MP4"
    output_video = r"C:/Users/DELL/Desktop/FYP/YOLO/Output/Outdoor01_annotated.mp4"
    conf_threshold = 0.35
    show_preview = True  # Change to False if you don't want live preview

    run(input_video, output_video, conf_threshold, show_preview)