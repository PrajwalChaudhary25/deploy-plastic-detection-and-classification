import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["3000 per day", "1000 per hour"],
    storage_uri="memory://"
)

MODAL_ENDPOINT = os.environ.get(
    "MODAL_ENDPOINT_URL", 
    "https://<your-username>--yolo-cnn-inference-modelinference-predict.modal.run"
)

@app.route("/")
def index():
    return render_template("index.html")
  
@app.route("/video_inference")
def video_inference():
    return render_template("video_inference.html")
  
@app.route("/Video_inference_web")
def video_inference_web():
    return render_template("Video_inference_web.html")
  
# Rate limit camera frame stream endpoint: Max 5 requests per second per IP address
@app.route("/process_frame", methods=["POST"])
@limiter.limit("5 per second")
def process_frame():
  try:
    payload = request.get_json()
    if not payload or "image" not in payload:
        return jsonify({"error": "No image payload provided"}), 400
      
    # Relay base64 frame payload directly to Modal web endpoint
    modal_response = requests.post(MODAL_ENDPOINT, json=payload, timeout=5)
    return jsonify(modal_response.json()), modal_response.status_code
  
  except requests.exceptions.Timeout:
        return jsonify({"error": "Model inference request timed out"}), 504
  except Exception as e:
      return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))