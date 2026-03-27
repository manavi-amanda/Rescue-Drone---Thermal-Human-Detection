import urllib.request
import os

# Destination
save_path = "yolov8n-face.pt"

# Hugging Face raw model URL (community shared)
url = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt"

if not os.path.exists(save_path):
    print("Downloading yolov8n-face.pt ...")
    urllib.request.urlretrieve(url, save_path)
    print("Downloaded and saved to", save_path)
else:
    print("yolov8n-face.pt already exists.")