import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import modal

app = modal.App("yolo-cnn-inference")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]",
        "onnxruntime",
        "ultralytics",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy"
    )
  .add_local_file("yolov11.pt", remote_path="/root/yolov11.pt")
  .add_local_file("Pabin_Model.onnx", remote_path="/root/Pabin_Model.onnx")
  
)

# Tells Modal to import these modules only inside the cloud container
with image.imports():
  import cv2
  import numpy as np
  import base64
  import onnxruntime as ort
  import torch
  
  torch.set_default_device("cpu")
  from ultralytics import YOLO

@app.cls(
    image=image,
    cpu=1.5,
    memory=2048,
    container_idle_timeout=30,  # Container stops 15 seconds after processing a photo
)

class PlasticInference:
  @modal.enter()
  def load_models(self):
    print("Loading YOLO and ONNX CNN models into CPU RAM...")
    self.yolo_model = YOLO("/root/yolov11.pt")
    
    # Load ONNX session (No TensorFlow required)
    self.ort_session = ort.InferenceSession(
        "/root/Pabin_Model.onnx", 
        providers=['CPUExecutionProvider']
    )
    self.input_name = self.ort_session.get_inputs()[0].name
    self.class_labels = {0: 'HDPE', 1: 'PET', 2: 'PP', 3: 'PS'}
  
  @modal.fastapi_endpoint(method="POST")
  def predict(self, data: dict):
    try:
      # 1. Decode base64 frame from request
      image_data = data.get("image", "").split(",")[-1]
      image_bytes = base64.b64decode(image_data)
      np_arr = np.frombuffer(image_bytes, np.uint8)
      frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

      if frame is None:
          return {"error": "Invalid frame payload"}, 400

      # 2. Run YOLO Detection
      results = self.yolo_model(frame, conf=0.5, verbose=False)[0]
      detections = []

      # 3. Crop detected bounding boxes and pass through CNN
      for detection in results.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cropped_image = frame[y1:y2, x1:x2]
        if cropped_image.size == 0:
            continue

        # The Keras model expects RGB values, while OpenCV decodes BGR.
        img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = (img / 255.0).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # ONNX Inference
        outputs = self.ort_session.run(None, {self.input_name: img_array})
        prediction = outputs[0]
        predicted_class_index = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_class_index])

        predicted_label = (
            self.class_labels.get(predicted_class_index, "Unknown")
            if confidence >= 0.40
            else "Unknown"
        )

        detections.append({
            "class": predicted_label,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2]
        })
      return {"status": "success", "detections": detections}

    except Exception as e:
        return {"error": str(e)}, 500


