import requests
import cv2
import numpy as np
import base64
import os

def generate_valid_image_file(filename):
    """Generates a valid JPEG image file."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle so it's not just a black void
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.imwrite(filename, img)
    return filename

def test_file_upload():
    url = "http://127.0.0.1:5000/detect"
    temp_filename = "test_image.jpg"
    generate_valid_image_file(temp_filename)
    
    print(f"[*] Sending file upload request to {url}...")
    try:
        with open(temp_filename, 'rb') as f:
            files = {'image': (temp_filename, f, 'image/jpeg')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "image" in data and data["image"].startswith("data:image/jpeg;base64,"):
                print("[SUCCESS] File Upload Integration Test Passed: Received annotated image.")
                return True
            else:
                print(f"[FAILURE] Unexpected response format: {data}")
        else:
            print(f"[FAILURE] Server returned error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[ERROR] Could not connect to server or upload failed: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return False

if __name__ == "__main__":
    test_file_upload()
