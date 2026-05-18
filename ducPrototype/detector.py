import cv2
import numpy as np
import base64
import os
from typing import Optional, List, Dict, Any
from ultralytics import YOLO # type: ignore
import torch
import timm
from einops import rearrange

class Detector:
    def __init__(self, model_path: str = "models/best.pt", depth_model_path: str = "models/depth_anything_v2/depth_anything_v2_vitl.pth") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize YOLO
        if not os.path.exists(model_path):
            self.model = YOLO("yolo11n.pt")
        else:
            self.model = YOLO(model_path)

        # Initialize Depth Estimator
        self.depth_model = self._load_depth_model(depth_model_path)

    def process_frame(self, data_url: str, classes: Optional[List[int]] = None, conf: float = 0.05) -> Dict[str, str]:
        if "," in data_url:
            img_bytes = base64.b64decode(data_url.split(",")[1])
        else:
            img_bytes = base64.b64decode(data_url)

        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image — empty or invalid frame")

        # 1. Run YOLO inference
        results = self.model(frame, classes=classes, conf=conf)
        annotated = results[0].plot()

        _, buffer_ann = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded_ann = f"data:image/jpeg;base64,{base64.b64encode(buffer_ann).decode('utf-8')}"

        # 2. Run Depth Estimation
        depth_map_url = ""
        if self.depth_model is not None:
            depth_map_url = self._estimate_depth(frame)
        else:
            _, buffer_dep = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            depth_map_url = f"data:image/jpeg;base64,{base64.b64encode(buffer_dep).decode('utf-8')}"

        return {
            "annotated_image": encoded_ann,
            "depth_map": depth_map_url
        }

    def _load_depth_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Depth model weights not found at {model_path}.")
            return None
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device).eval()
            print("Depth model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading depth model: {e}")
            return None

    def _estimate_depth(self, frame: np.ndarray) -> str:
        try:
            depth = self.depth_model.infer_image(frame)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth_map = depth_norm.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

            _, buffer = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"