from huggingface_hub import hf_hub_download

# Download the pretrained YOLOv8 thermal human detector
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-human-detection-thermal",
    filename="model.pt"
)

print("Model downloaded at:", model_path)
