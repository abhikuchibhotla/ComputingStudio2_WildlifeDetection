from flask import Flask, render_template, request, jsonify
from detector import Detector

app = Flask(__name__)
detector = Detector()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Check if the request is multipart/form-data (file upload)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            # Read file bytes and convert to base64 string for the detector
            img_bytes = file.read()
            import base64
            data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        
        # Fallback to JSON body (for backward compatibility or testing)
        elif request.is_json:
            body = request.json
            if not body or "image" not in body:
                return jsonify({"error": "No image field in request"}), 400
            data_url = body["image"]
        else:
            return jsonify({"error": "Invalid request format. Use file upload or JSON."}), 400

        result = detector.process_frame(data_url)
        return jsonify({
            "annotated_image": result["annotated_image"],
            "depth_map": result["depth_map"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
