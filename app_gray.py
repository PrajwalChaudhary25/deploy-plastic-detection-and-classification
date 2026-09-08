import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODAL_ENDPOINT = os.environ.get(
    "MODAL_ENDPOINT_URL", 
    "https://prajwalchy25--yolo-cnn-inference-plasticinference-predict-dev.modal.run"
)

@app.route("/")
def index():
    return render_template("index.html")
  
@app.route("/upload", methods=["POST"])
def upload_file():
  if "file" not in request.files:
      return jsonify({"error": "No file uploaded"}), 400

  file = request.files["file"]
  if file.filename == "":
      return jsonify({"error": "No selected file"}), 400

  # Read image bytes and convert to base64 to send to Modal
  import base64
  image_bytes = file.read()
  base64_image = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

  # Forward payload to Modal
  try:
      response = requests.post(MODAL_ENDPOINT, json={"image": base64_image}, timeout=20)
      return jsonify(response.json()), response.status_code
  except Exception as e:
      return jsonify({"error": str(e)}), 500

@app.route("/photo_inference")
def photo_inference():
        return render_template("inference.html")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))