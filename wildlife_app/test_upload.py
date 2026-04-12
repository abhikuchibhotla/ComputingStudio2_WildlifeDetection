import requests

def test_file_upload():
    url = "http://127.0.0.1:5000/detect"
    # Using the test script itself as a dummy file to upload
    file_path = r'D:\CS2\ComputingStudio2_WildlifeDetection\wildlife_app\test_integration.py'
    
    print(f"[*] Sending file upload request to {url}...")
    try:
        with open(file_path, 'rb') as f:
            files = {'image': (file_path, f, 'image/jpeg')}
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
    
    return False

if __name__ == "__main__":
    test_file_upload()
