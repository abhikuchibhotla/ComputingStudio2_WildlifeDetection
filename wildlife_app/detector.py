import cv2
import numpy as np
import base64
import os
from typing import Optional, List
from ultralytics import YOLO # type: ignore

class Detector:
    """
    A class to handle YOLO object detection inference.
    """

    def __init__(self, model_path: str = "models/best.pt") -> None:
        """
        Initializes the Detector with a YOLO model.

        Args:
            model_path (str): Path to the custom YOLO model weights. 
                             Defaults to 'models/best.pt'.
        """
        if not os.path.exists(model_path):
            # Fallback to default YOLO model if custom model is not found
            self.model = YOLO("yolo11n.pt")
        else:
            self.model = YOLO(model_path)

    def process_frame(
        self, 
        data_url: str, 
        classes: Optional[List[int]] = None, 
        conf: float = 0.05
    ) -> str:
        """
        Processes a base64 encoded image string and returns an annotated base64 string.

        Args:
            data_url (str): Base64 encoded image string (can include data URI prefix).
            classes (Optional[List[int]]): List of class indices to detect.
            conf (float): Confidence threshold.

        Returns:
            str: Base64 encoded annotated image string with 'data:image/jpeg;base64,' prefix.

        Raises:
            ValueError: If the image decoding fails.
        """
        # Handle data URI prefix if present
        if "," in data_url:
            img_bytes = base64.b64decode(data_url.split(",")[1])
        else:
            img_bytes = base64.b64decode(data_url)

        # Decode bytes to OpenCV image
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image — empty or invalid frame")

        # Run inference
        results = self.model(frame, classes=classes, conf=conf)
        
        # Annotate the frame
        annotated = results[0].plot()

        # Encode annotated image back to base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded_string = base64.b64encode(buffer).decode("utf-8")
        
        return f"data:image/jpeg;base64,{encoded_string}"
