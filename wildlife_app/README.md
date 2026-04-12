# Wildlife Detection Application

A lightweight web application for performing object detection on images using YOLO models. This application provides a simple interface to upload images and receive annotated results.

## Features

- **Image Upload**: Upload local images via a web interface.
- **YOLO Inference**: Uses the `ultralytics` library to perform object detection.
- **Automatic Model Fallback**: Automatically loads a default `yolo11n.pt` model if a custom model is not found in the `models/` directory.
- **Base64 Pipeline**: Efficiently handles image data transfer between the client and server using Base64 encoding.

## Workflow

1.  **User Input**: The user selects an image file through the web interface.
2.    **Client-side Processing**: The browser captures the file and sends it via a `POST` request as `multipart/form-data` to the `/detect` endpoint.
3.  **Backend Handling**:
    - The Flask server receives the image.
    - The image bytes are decoded into an OpenCV-compatible format.
4.  **Inference**:
    - The `Detector` class passes the image to the loaded YOLO model.
    - The model performs detection and generates bounding boxes.
    - The annotated image is re-encoded into a Base64 JPEG string.
5.  **Result Delivery**: The server returns the annotated image string as a JSON response.
6.  **UI Update**: The browser receives the response and updates the image display on the page.

## Dependencies

The application requires Python 3.x and the following packages:

- `flask`: Web framework for the backend server.
- `ultralytics`: Core library for YOLO model inference.
- `opencv-python`: For image decoding and annotation.
- `numpy`: For numerical operations on image arrays.

## Installation and Setup

### 1. Environment Setup
It is recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (Linux/macOS)
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Model Configuration
Place your custom YOLO model weights (`.pt` file) in the `models/` directory. By default, the application looks for `models/best.pt`. If no model is found, it will automatically download and use `yolo11n.pt`.

### 4. Running the Application
Start the Flask server:

```bash
python app.py
```

Once started, open your browser and navigate to `http://127.0.0.1:5000`.
