import requests
import base64
import cv2
import numpy as np
import os

def generate_dummy_image():
    """Generates a simple black image and returns it as a base64 string."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"

def test_detection_endpoint():
    url = "http://127.0.0.1:5000/detect"
    image_data = generate_dummy_image()
    
    print(f"[*] Sending request to {url}...")
    try:
        # Test JSON fallback (backward compatibility)
        response = requests.post(url, json={"image": image_data})
        
        if response.status_code == 200:
            data = response.json()
            if "image" in data and data["image"].startswith("data:image/jpeg;base64,"):
                print("[SUCCESS] JSON fallback Integration Test Passed: Received annotated image.")
                return True
            else:
                print(f"[FAILURE] Unexpected response format: {data}")
        else:
            print(f"[FAILURE] Server returned error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[ERROR] Could not connect to server: {e}")
    
    return False

if __name__ == "__main__":
    test_detection_endpoint()
