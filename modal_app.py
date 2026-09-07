import modal
import cv2
import numpy as np
import base64

app = modal.App("yolo-cnn-inference")

image = (
  modal.Image.debian_slim()
  .pip_install("ultralytics", "torch", "torchvision", "opencv-python-headless", "numpy")
)

@app.cls(image = image, cpu = 1.5, memory = 2048)
class ModelInference:
  @modal.enter()
  def load_models(self):
    # Executes once when container starts (Warming up the model)
    from ultralytics import YOLO
    print("Loading YOLO and CNN models into RAM...")
    self.yolo_model = YOLO("yolov8n.pt")
    
  @modal.web_endpoint(method = "POST")
  def predict(self, data:dict):
    try:
      # 1. Decode base64 frame from Flask
      image_data = data.get("image", "").split(",")[-1]
      image_bytes = base64.b64decode(image_data)
      np_arr = np.frombuffer(image_bytes, np.uint8)
      frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
      
      if frame is None:
        return {"error": "Invalid frame"}, 400
      
      results = self.yolo_model(frame, verbose =  false)
      
      detections = []
      for r in results:
        for box in r.boxes:
          detections.append({
              "class": self.yolo_model.names[int(box.cls[0])],
              "confidence": float(box.conf[0]),
              "bbox": box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
          })
          
      return {"status": "success", "detections": detections}
    
    except Exception as e:
            return {"error": str(e)}, 500
    
    
  

