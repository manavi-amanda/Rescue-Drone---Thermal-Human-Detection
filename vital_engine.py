import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# Import your custom modules
from Respirotory_Rate_1 import YOLODetector, preprocess_thermal, analyse_respiratory_rate

# In-memory store for results (For production, use Redis/Database)
task_results = {}

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def run_headless(task_id, input_path, output_dir):
    """Processes video, extracts intensity, saves PNG, returns summary."""
    cap = cv2.VideoCapture(input_path)
    FPS_IN = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = YOLODetector(conf=0.35)

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH))
    landmarker = FaceLandmarker.create_from_options(options)

    intensity_log = []
    nose_trail = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        processed_frame = preprocess_thermal(frame)
        people, faces = detector.detect(processed_frame)

        # ROI Extraction Logic
        for (fx1, fy1, fx2, fy2, _) in faces:
            crop = processed_frame[fy1:fy2, fx1:fx2]
            if crop.size == 0: continue

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)

            if result.face_landmarks:
                fh, fw = crop.shape[:2]
                nose_lm = result.face_landmarks[0][4]
                nx, ny = int(nose_lm.x * fw) + fx1, int(nose_lm.y * fh) + fy1

                # compute_roi and extract_roi_intensity should be imported or defined here
                # Assuming they are available from your Respirotory_Rate_1 or defined helpers
                from Respirotory_Rate_1 import compute_roi, extract_roi_intensity

                nose_trail.append((nx, ny))
                if len(nose_trail) > 60: nose_trail.pop(0)

                cx, cy, rx, ry = compute_roi((nx, ny), nose_trail, (fx1, fy1, fx2, fy2), processed_frame.shape)
                intensity_log.append(extract_roi_intensity(processed_frame, cx, cy, rx, ry))

    cap.release()
    landmarker.close()

    # Generate Analysis
    if intensity_log:
        plot_path = os.path.join(output_dir, f"analysis_{task_id}.png")
        stats = analyse_respiratory_rate(intensity_log, FPS_IN, plot_path)

        # Save summary to global dict
        task_results[task_id] = {
            "status": "Completed",
            "final_bpm": stats.get('final_bpm'),
            "clinical_note": stats.get('note'),
            "duration_seconds": stats.get('duration_s'),
            "frame_count": stats.get('n_samples'),
            "plot_url": f"/download/analysis_{task_id}.png"
        }
    else:
        task_results[task_id] = {"status": "Failed", "error": "No vitals detected in video"}

    # Optional: Delete input video to save space
    if os.path.exists(input_path):
        os.remove(input_path)