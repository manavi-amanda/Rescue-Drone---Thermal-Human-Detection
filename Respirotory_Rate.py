import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# =========================================================
# 1. BANDPASS FILTER FUNCTION
# Keeps only breathing frequency range (0.08 - 0.7 Hz)
# 0.08 Hz ≈ 5 BPM
# 0.7 Hz ≈ 42 BPM
# =========================================================

def bandpass_filter(signal, fs, low=0.08, high=0.7):
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)


# =========================================================
# 2. LOAD VIDEO
# =========================================================

video_path = r"Inputs/indoor1.MP4"
cap = cv2.VideoCapture(video_path)
#get the length of the video in seconds
video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

# Read FPS from video
fps = cap.get(cv2.CAP_PROP_FPS)

# If metadata incorrect, manually set FPS
fps = 15

print(f"Video FPS: {fps}")

# =========================================================
# 3. FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Store breathing signal over time
temperature_signal = []

# =========================================================
# 4. PROCESS VIDEO FRAME BY FRAME
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (thermal intensity)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Define Nose ROI (lower-middle part of face)
        nose_y1 = y + int(h * 0.55)
        nose_y2 = y + int(h * 0.75)
        nose_x1 = x + int(w * 0.3)
        nose_x2 = x + int(w * 0.7)

        # Draw Nose ROI
        cv2.rectangle(frame,
                      (nose_x1, nose_y1),
                      (nose_x2, nose_y2),
                      (255, 0, 0),
                      2)

        cv2.putText(frame,
                    "Nose ROI",
                    (nose_x1, nose_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

        # Extract Nose ROI
        nose_roi = gray[nose_y1:nose_y2, nose_x1:nose_x2]

        if nose_roi.size > 0:
            # More stable than single min pixel:
            sorted_pixels = np.sort(nose_roi.flatten())
            coldest_mean = np.mean(sorted_pixels[:5])  # average of 5 coldest pixels
            temperature_signal.append(coldest_mean)

    cv2.imshow("Respiratory ROI Detection", frame)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# =========================================================
# 5. SIGNAL VALIDATION (MODIFIED)
# =========================================================

temperature_signal = np.array(temperature_signal)

if len(temperature_signal) == 0:
    print("No signal extracted. Check face detection.")
    exit()

# Calculate real duration after signal extraction
duration = len(temperature_signal) / fps
print(f"Actual Signal Duration: {duration:.2f} seconds")

# Warning for short recordings
if duration < 30:
    print("Warning: Recording is shorter than 30 seconds. Respiratory rate estimate may be less accurate.")


# =========================================================
# 6. FILTER BREATHING SIGNAL
# =========================================================

filtered_signal = bandpass_filter(temperature_signal, fps)


# =========================================================
# 7. DETECT BREATHING PEAKS
# =========================================================

# Adaptive minimum distance between breaths
min_distance = int(fps * 1)

peaks, properties = find_peaks(
    filtered_signal,
    distance=min_distance,
    prominence=0.2 * np.std(filtered_signal)  # ignore small noise peaks
)

breaths = len(peaks)

# Respiratory Rate calculation
respiratory_rate = (breaths / video_length) * 60

print("\nDetected Breaths:", breaths)
print("Respiratory Rate:", round(respiratory_rate, 2), "breaths/min")


# =========================================================
# 8. PLOT BREATHING SIGNAL
# =========================================================

time_axis = np.arange(len(filtered_signal)) / fps

plt.figure(figsize=(12,5))

plt.plot(time_axis, filtered_signal, label="Filtered Signal")
plt.plot(time_axis[peaks], filtered_signal[peaks], "rx", label="Detected Breaths")

plt.title(f"Respiratory Rate = {respiratory_rate:.2f} BPM")
plt.xlabel("Time (seconds)")
plt.ylabel("Thermal Intensity")
plt.legend()
plt.grid()

plt.show()