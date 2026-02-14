from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# 1. Load YOLOv8 thermal human model
# -----------------------------
model_path = r"C:\Users\DELL\.cache\huggingface\hub\models--arnabdhar--YOLOv8-human-detection-thermal\snapshots\a5d30f1e8e185a54a83cb6288d38c806e303bdd2\model.pt" # Update with model path
model = YOLO(model_path)

# -----------------------------
# 2. Open thermal video
# -----------------------------
video_path = "thermal005.mp4"  # input video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# -----------------------------
# 3. Video properties
# -----------------------------
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Output video
output_path = "thermaloutput005.mp4"
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -----------------------------
# 4. Process frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Ensure frame has 3 channels (BGR)
    # -----------------------------
    if len(frame.shape) == 2:  # grayscale
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame

    # -----------------------------
    # Run YOLO detection
    # -----------------------------
    results = model(frame_bgr, conf=0.4)

    # -----------------------------
    # Loop over all detected humans
    # -----------------------------
    for box in results[0].boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)

        # Crop bounding box
        person_roi = frame_bgr[y1:y2, x1:x2]

        # Convert ROI to grayscale to get thermal intensity
        if len(person_roi.shape) == 3:
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = person_roi

        # -----------------------------
        # Map mean pixel intensity to Celsius (linear 34°C–40°C)
        # --------------------------
        temp_c = (np.mean(gray_roi) / 255) * (40 - 34) + 34 #not accurate mapping


        # Print to console
        print(f"Human detected: Temperature = {temp_c:.1f}°C")

        # Annotate on frame
        label = f"Temp: {temp_c:.1f} C"
        cv2.putText(frame_bgr, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -----------------------------
    # Display and save frame
    # -----------------------------
    cv2.imshow("Thermal Human Detection", frame_bgr)
    out.write(frame_bgr)

    # Quit early with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5. Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("Detection finished! Output saved as:", output_path)
