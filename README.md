# Thermal Human Detection using YOLOv8 


## ⚙ Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python numpy huggingface_hub
```

Recommended environment:

* Python 3.10 or 3.11
* GPU optional (CPU works but slower)

---

## ⬇ Download Model Script

Run once to download the pretrained model:

```bash
python download_model.py
```

This downloads the model from Hugging Face and prints its path.

---

## ▶ Run Detection

Place your thermal video in the project folder and update:

```python
video_path = "thermal005.mp4"
```

Then run:

```bash
python Thermal_human_detector.py
```

---

## 📊 Temperature Estimation Method

Temperature is estimated using a **linear mapping** from grayscale intensity:

```
temp = (mean_pixel / 255) × (40 − 34) + 34
```

⚠ **Important:**
This is **not real body temperature measurement**.
It is only a visualization approximation.

---

## 🎥 Output

The program generates:

* annotated video file
* bounding boxes around detected humans
* temperature label near each detection

Output file example:

```
thermaloutput005.mp4
```

---

## ⚠ Known Limitations

* Temperature values are simulated, not calibrated
* Accuracy depends on thermal camera quality
* Cannot detect humans if contrast is too low
* Works best with clear thermal silhouettes

---

## 🧠 Model Source

The pretrained model is designed for **thermal human detection** and is downloaded automatically from Hugging Face Hub.

---

## 🚀 Future Improvements

* Real radiometric temperature extraction
* Tracking IDs for individuals
* FPS optimization
* Live camera input
* Safety distance detection

---

## API Service Setup and Testing

### 🚀 Run the API Service

To start the FastAPI server, run the following command from your project root:

```bash
python app_gateway.py
```

The server will start at:

* API Base URL: `http://127.0.0.1:8000`
* Interactive Documentation: `http://127.0.0.1:8000/docs`


### 🧪 Testing the API

You can test the detection and vital analysis endpoints using `curl` from your terminal.

#### 1. Upload Video for Analysis

Replace `thermal005.mp4` with the path to your own video file. This request starts the background processing task.

```bash
curl -X 'POST' 'http://127.0.0.1:8000/analyze/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@thermal005.mp4'
```

---