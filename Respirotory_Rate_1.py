"""
Thermal Video - Face, Nose Trail, Breathing ROI & Respiratory Rate
===================================================================
- Detects people (YOLO full body)
- Detects faces (YOLO face / Haar fallback)
- Tracks nose tip (MediaPipe FaceLandmarker)
- Draws fading nose trail
- Marks exhale/inhale ROI ellipse below nose tip
- Extracts mean pixel intensity inside ROI
- Analyses intensity signal in 3 ways:
    Step 1 : Detrend + smooth
    Step 2 : FFT  -> dominant breathing frequency
    Step 3 : Peak detection -> count breath cycles
    Step 4 : Bandpass filter -> cleanest signal
    Step 5 : Combine all three -> final BPM
- Saves CSV + analysis plots + prints final result

Requirements:
    pip install ultralytics opencv-python mediapipe==0.10.32 numpy scipy matplotlib
"""

import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#  MEDIAPIPE IMPORTS
# ══════════════════════════════════════════════════════
import mediapipe as mp
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

try:
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
except ImportError:
    from mediapipe.tasks.python.vision import RunningMode

# ══════════════════════════════════════════════════════
#  ANALYSIS IMPORTS
# ══════════════════════════════════════════════════════
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════
#  AUTO-DOWNLOAD MODEL
# ══════════════════════════════════════════════════════
MODEL_PATH = r"C:/Users/DELL/Desktop/FYP/YOLO/face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Model saved -> {MODEL_PATH}")

# ══════════════════════════════════════════════════════
#  BUILD FACE LANDMARKER
# ══════════════════════════════════════════════════════
face_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
)
print("[INFO] MediaPipe FaceLandmarker loaded successfully!")

# ══════════════════════════════════════════════════════
#  THERMAL PRE-PROCESSING
# ══════════════════════════════════════════════════════
def preprocess_thermal(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint16:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════════════
#  YOLO DETECTOR
# ══════════════════════════════════════════════════════
class YOLODetector:
    PERSON_ID = 0

    def __init__(self, conf: float = 0.35):
        print("[INFO] Loading YOLOv8n model ...")
        self.person_model = YOLO("yolov8n.pt")
        try:
            self.face_model = YOLO("yolov8n-face.pt")
            self.face_model_available = True
            print("[INFO] YOLOv8-face model loaded.")
        except Exception:
            self.face_model_available = False
            print("[WARN] yolov8n-face.pt not found — using Haar cascade.")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self.conf = conf

    def detect(self, frame):
        people, faces = [], []
        for box in self.person_model(frame, conf=self.conf,
                                     classes=[self.PERSON_ID], verbose=False)[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people.append((x1, y1, x2, y2, float(box.conf[0])))
        if self.face_model_available:
            for box in self.face_model(frame, conf=self.conf, verbose=False)[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2, float(box.conf[0])))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
                faces.append((x, y, x + w, y + h, 1.0))
        return people, faces

# ══════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════
COLOR_PERSON   = (0, 255, 120)
COLOR_FACE     = (0, 180, 255)
COLOR_NOSE     = (0, 0, 255)
COLOR_ROI      = (0, 215, 255)
COLOR_ROI_FILL = (0, 215, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    c = 12
    for px, py, dx, dy in [
        (x1,y1,c,0),(x1,y1,0,c),(x2,y1,-c,0),(x2,y1,0,c),
        (x1,y2,c,0),(x1,y2,0,-c),(x2,y2,-c,0),(x2,y2,0,-c),
    ]:
        cv2.line(frame, (px, py), (px+dx, py+dy), color, 3)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
    cv2.rectangle(frame, (x1, max(y1-th-6, 0)), (x1+tw+6, y1), color, -1)
    cv2.putText(frame, label, (x1+3, y1-3), FONT, 0.50, (0,0,0), 1, cv2.LINE_AA)


def compute_roi(nose_pt, trail, face_box, frame_shape):
    nx, ny = nose_pt
    face_w = face_box[2] - face_box[0]
    face_h = face_box[3] - face_box[1]
    offset_y = max(int(face_h * 0.08), 8)
    cx = nx
    cy = ny + offset_y
    min_rx = max(int(face_w * 0.18), 14)
    min_ry = max(int(face_h * 0.08), 8)
    if len(trail) >= 4:
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        rx = max(min_rx, int((max(xs)-min(xs)) * 0.6) + min_rx)
        ry = max(min_ry, int((max(ys)-min(ys)) * 0.5) + min_ry)
    else:
        rx, ry = min_rx, min_ry
    rx = min(rx, int(face_w * 0.45))
    ry = min(ry, int(face_h * 0.30))
    H, W = frame_shape[:2]
    cx = int(np.clip(cx, rx, W - rx))
    cy = int(np.clip(cy, ry, H - ry))
    return (cx, cy, rx, ry)


def draw_roi(frame, cx, cy, rx, ry, intensity):
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, COLOR_ROI_FILL, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    segments = 36
    pts = [(int(cx + rx * np.cos(np.radians(i * 360 / segments))),
            int(cy + ry * np.sin(np.radians(i * 360 / segments))))
           for i in range(segments + 1)]
    for i in range(0, segments, 2):
        cv2.line(frame, pts[i], pts[i+1], COLOR_ROI, 2, cv2.LINE_AA)
    label = f"ROI  {intensity:.0f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    lx = cx - tw // 2
    ly = cy + ry + 16
    cv2.rectangle(frame, (lx-4, ly-th-3), (lx+tw+4, ly+3), (20,20,20), -1)
    cv2.putText(frame, label, (lx, ly), FONT, 0.45, COLOR_ROI, 1, cv2.LINE_AA)


def extract_roi_intensity(frame, cx, cy, rx, ry):
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray, mask=mask)[0]

# ══════════════════════════════════════════════════════════════════
#  RESPIRATORY RATE ANALYSIS
#  Runs after video is processed, directly on the intensity_log list.
#  Saves one combined plot image.  Returns (fft_bpm, peak_bpm, band_bpm, final_bpm)
# ══════════════════════════════════════════════════════════════════
def analyse_respiratory_rate(intensity_log: list, fps: float, plot_path: str):
    """
    5-step analysis pipeline on the raw intensity_log captured during video processing.

    Step 1 : Raw signal visualisation
    Step 2 : Detrend + smooth
    Step 3 : FFT  → dominant frequency in 0.1-0.8 Hz band
    Step 4 : Peak detection on smoothed signal
    Step 5 : Butterworth bandpass filter + peak detection
    Final  : Median of the three estimates → robust BPM

    Saves a 5-panel figure to plot_path.
    Returns dict with all BPM values.
    """

    signal_raw = np.array(intensity_log, dtype=float)
    n          = len(signal_raw)
    time       = np.arange(n) / fps

    if n < int(fps * 3):
        print("[WARN] Signal too short for reliable analysis (< 3 s). Skipping.")
        return None

    print("\n" + "="*60)
    print("  RESPIRATORY RATE ANALYSIS")
    print("="*60)
    print(f"  Signal length : {n} samples  ({n/fps:.1f} s)")
    print(f"  Sample rate   : {fps:.1f} fps")
    print(f"  Breathing band: 0.1 – 0.8 Hz  (6 – 48 BPM)")
    print("-"*60)

    # ── STEP 1 : Raw signal ────────────────────────────────────────
    print("\n[Step 1] Raw ROI intensity signal captured.")
    print(f"         min={signal_raw.min():.1f}  max={signal_raw.max():.1f}"
          f"  mean={signal_raw.mean():.1f}  std={signal_raw.std():.2f}")

    # ── STEP 2 : Detrend + smooth ──────────────────────────────────
    signal_detrended = detrend(signal_raw)
    smooth_win       = max(3, int(fps * 0.5))          # 0.5 s window
    signal_smooth    = uniform_filter1d(signal_detrended, size=smooth_win)
    # Normalise to 0-1 for visualisation
    sig_norm = signal_smooth - signal_smooth.min()
    denom    = sig_norm.max()
    sig_norm = sig_norm / denom if denom > 0 else sig_norm

    print(f"\n[Step 2] Detrended + smoothed  (window={smooth_win} frames = {smooth_win/fps:.2f}s).")

    # ── STEP 3 : FFT ───────────────────────────────────────────────
    fft_vals  = np.abs(np.fft.rfft(signal_smooth))
    freqs     = np.fft.rfftfreq(n, d=1.0 / fps)
    band_mask = (freqs >= 0.1) & (freqs <= 0.8)

    if band_mask.sum() == 0:
        fft_bpm = float('nan')
        print("[Step 3] FFT: not enough frequency resolution — signal too short.")
    else:
        peak_freq = freqs[band_mask][np.argmax(fft_vals[band_mask])]
        fft_bpm   = round(peak_freq * 60, 1)
        print(f"\n[Step 3] FFT dominant frequency : {peak_freq:.4f} Hz  ->  {fft_bpm} BPM")

    # ── STEP 4 : Peak detection ────────────────────────────────────
    min_dist  = max(1, int(fps * 1.0))                 # min 1 s between breaths
    peaks, props = find_peaks(sig_norm, distance=min_dist, prominence=0.05)
    duration_s   = n / fps
    peak_bpm     = round((len(peaks) / duration_s) * 60, 1) if len(peaks) > 0 else float('nan')

    print(f"\n[Step 4] Peak detection : {len(peaks)} peaks in {duration_s:.1f}s  ->  {peak_bpm} BPM")
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fps
        print(f"         Inter-peak intervals: min={intervals.min():.2f}s  "
              f"max={intervals.max():.2f}s  mean={intervals.mean():.2f}s")

    # ── STEP 5 : Bandpass filter ───────────────────────────────────
    nyq    = fps / 2.0
    low_n  = 0.1 / nyq
    high_n = min(0.8 / nyq, 0.99)                      # must be < 1
    b, a   = butter(N=3, Wn=[low_n, high_n], btype="band")
    signal_band = filtfilt(b, a, signal_detrended)

    # Normalise bandpass for peak detection
    sb_norm = signal_band - signal_band.min()
    db      = sb_norm.max()
    sb_norm = sb_norm / db if db > 0 else sb_norm

    band_peaks, _ = find_peaks(sb_norm, distance=min_dist, prominence=0.05)
    band_bpm      = round((len(band_peaks) / duration_s) * 60, 1) if len(band_peaks) > 0 else float('nan')

    print(f"\n[Step 5] Bandpass filter (0.1-0.8 Hz): {len(band_peaks)} peaks  ->  {band_bpm} BPM")

    # ── FINAL : Combine ────────────────────────────────────────────
    estimates = [v for v in [fft_bpm, peak_bpm, band_bpm] if not np.isnan(v)]
    if estimates:
        final_bpm = round(float(np.median(estimates)), 1)
    else:
        final_bpm = float('nan')

    print("\n" + "-"*60)
    print(f"  FFT estimate    : {fft_bpm}  BPM")
    print(f"  Peak detection  : {peak_bpm}  BPM")
    print(f"  Bandpass filter : {band_bpm}  BPM")
    print(f"  FINAL (median)  : {final_bpm}  BPM")
    normal = "Normal (12-20 BPM)" if 12 <= final_bpm <= 20 else \
             "Low " if final_bpm < 12 else \
             "Elevated"
    print(f"  Clinical note   : {normal}")
    print("="*60 + "\n")

    # ── PLOT ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16))
    fig.suptitle("Respiratory Rate Analysis — ROI Intensity Signal", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.55)

    # Panel 1 — raw signal
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, signal_raw, color="#4a90d9", linewidth=0.8, alpha=0.9)
    ax1.set_title("Step 1 — Raw ROI intensity", fontsize=11)
    ax1.set_ylabel("Intensity (0-255)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — detrended + smoothed
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, signal_detrended, color="#aaaaaa", linewidth=0.6, alpha=0.7, label="Detrended")
    ax2.plot(time, signal_smooth,    color="#e07b39", linewidth=1.2, label=f"Smoothed (w={smooth_win}fr)")
    ax2.set_title("Step 2 — Detrended + smoothed signal", fontsize=11)
    ax2.set_ylabel("Amplitude"); ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # Panel 3 — FFT
    ax3 = fig.add_subplot(gs[2])
    bpm_axis = freqs[band_mask] * 60
    ax3.plot(bpm_axis, fft_vals[band_mask], color="#9b59b6", linewidth=1.2)
    ax3.axvline(fft_bpm, color="red", linestyle="--", linewidth=1.5,
                label=f"Peak: {fft_bpm} BPM")
    ax3.fill_between(bpm_axis, fft_vals[band_mask], alpha=0.15, color="#9b59b6")
    ax3.set_title("Step 3 — FFT power spectrum (breathing band 6–48 BPM)", fontsize=11)
    ax3.set_xlabel("Breaths per minute"); ax3.set_ylabel("Power")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # Panel 4 — peak detection
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(time, sig_norm, color="#27ae60", linewidth=1.0, label="Normalised signal")
    ax4.plot(time[peaks], sig_norm[peaks], "rv", markersize=7,
             label=f"{len(peaks)} peaks  →  {peak_bpm} BPM")
    ax4.set_title("Step 4 — Peak detection on smoothed signal", fontsize=11)
    ax4.set_ylabel("Normalised amplitude"); ax4.set_xlabel("Time (s)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # Panel 5 — bandpass + peaks
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(time, sb_norm, color="#e74c3c", linewidth=1.0, label="Bandpass filtered")
    ax5.plot(time[band_peaks], sb_norm[band_peaks], "b^", markersize=7,
             label=f"{len(band_peaks)} peaks  →  {band_bpm} BPM")
    ax5.set_title("Step 5 — Butterworth bandpass filter (0.1–0.8 Hz) + peaks", fontsize=11)
    ax5.set_ylabel("Normalised amplitude"); ax5.set_xlabel("Time (s)")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    # Final BPM annotation box
    fig.text(0.5, 0.01,
             f"FINAL RESPIRATORY RATE:  {final_bpm} BPM   |   "
             f"FFT: {fft_bpm}  |  Peaks: {peak_bpm}  |  Bandpass: {band_bpm}  |  {normal}",
             ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#e0a800"))

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Analysis plot saved -> {plot_path}")

    return {
        "fft_bpm":   fft_bpm,
        "peak_bpm":  peak_bpm,
        "band_bpm":  band_bpm,
        "final_bpm": final_bpm,
        "note":      normal,
        "n_samples": n,
        "duration_s": round(duration_s, 2),
    }

# ══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════
def run(input_path: str, output_path: str, conf: float = 0.35, show: bool = True):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_IN = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} | {FPS_IN:.1f} fps | {TOTAL} frames")

    writer     = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS_IN, (W, H))
    detector   = YOLODetector(conf)
    nose_trail = []
    TRAIL_LEN  = 60
    frame_no   = 0
    intensity_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame         = preprocess_thermal(frame)
        people, faces = detector.detect(frame)

        for i, (x1, y1, x2, y2, cv_val) in enumerate(people):
            draw_box(frame, x1, y1, x2, y2, f"Person {i+1} {cv_val:.0%}", COLOR_PERSON)

        for i, (fx1, fy1, fx2, fy2, cv_val) in enumerate(faces):
            draw_box(frame, fx1, fy1, fx2, fy2, f"Face {i+1} {cv_val:.0%}", COLOR_FACE)

            crop = frame[fy1:fy2, fx1:fx2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )
            result = face_landmarker.detect(mp_img)

            if result.face_landmarks:
                fh, fw  = crop.shape[:2]
                nose_lm = result.face_landmarks[0][4]
                nx = int(nose_lm.x * fw) + fx1
                ny = int(nose_lm.y * fh) + fy1
                nose_pt = (nx, ny)

                nose_trail.append(nose_pt)
                if len(nose_trail) > TRAIL_LEN:
                    nose_trail.pop(0)

                cv2.circle(frame, nose_pt, 5, COLOR_NOSE, -1)
                cv2.circle(frame, nose_pt, 7, COLOR_NOSE, 1)

                for j in range(1, len(nose_trail)):
                    alpha = j / max(len(nose_trail), 1)
                    cv2.line(frame, nose_trail[j-1], nose_trail[j],
                             (0, 0, int(55 + 200 * alpha)),
                             max(1, int(alpha * 3)))

                cx, cy, rx, ry = compute_roi(
                    nose_pt, nose_trail, (fx1, fy1, fx2, fy2), frame.shape
                )
                intensity = extract_roi_intensity(frame, cx, cy, rx, ry)
                intensity_log.append(intensity)
                draw_roi(frame, cx, cy, rx, ry, intensity)

        hud = f"Frame:{frame_no}  People:{len(people)}  Faces:{len(faces)}"
        cv2.rectangle(frame, (8, 8), (8 + len(hud)*9, 28), (20, 20, 20), -1)
        cv2.putText(frame, hud, (12, 23), FONT, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if show:
            cv2.imshow("Thermal  |  Face + Nose Trail + Breathing ROI  [Q=quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_no += 1
        if frame_no % 50 == 0:
            print(f"  ... {frame_no}/{TOTAL}  people={len(people)}  faces={len(faces)}")

    cap.release()
    writer.release()
    face_landmarker.close()
    if show:
        cv2.destroyAllWindows()

    # ── Save CSV (unchanged from original) ────────────────────────
    if intensity_log:
        csv_path = output_path.replace(".mp4", "_roi_intensity.csv")
        with open(csv_path, "w") as f:
            f.write("frame,intensity\n")
            for idx, val in enumerate(intensity_log):
                f.write(f"{idx},{val:.4f}\n")
        print(f"[INFO] ROI intensity CSV saved -> {csv_path}")

    print(f"\n[DONE] Video saved -> {output_path}")

    # ── Run respiratory rate analysis directly on intensity_log ───
    if intensity_log:
        plot_path = output_path.replace(".mp4", "_rr_analysis.png")
        results   = analyse_respiratory_rate(intensity_log, FPS_IN, plot_path)

        if results:
            # Optionally write a brief summary text file next to outputs
            summary_path = output_path.replace(".mp4", "_rr_summary.txt")
            with open(summary_path, "w") as f:
                f.write("RESPIRATORY RATE ANALYSIS SUMMARY\n")
                f.write("="*40 + "\n")
                f.write(f"Video          : {input_path}\n")
                f.write(f"Duration       : {results['duration_s']} s\n")
                f.write(f"Samples        : {results['n_samples']}\n")
                f.write(f"FPS            : {FPS_IN:.1f}\n\n")
                f.write(f"FFT estimate   : {results['fft_bpm']} BPM\n")
                f.write(f"Peak detection : {results['peak_bpm']} BPM\n")
                f.write(f"Bandpass filter: {results['band_bpm']} BPM\n")
                f.write(f"FINAL (median) : {results['final_bpm']} BPM\n")
                f.write(f"Clinical note  : {results['note']}\n")
            print(f"[INFO] Summary text saved -> {summary_path}")

    return intensity_log


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    input_video  = r"C:/Users/DELL/Desktop/FYP/YOLO/Inputs/indoor1.MP4"
    output_video = r"C:/Users/DELL/Desktop/FYP/YOLO/Output/indoor01_nosetrail_new.mp4"
    run(input_video, output_video, conf=0.35, show=True)