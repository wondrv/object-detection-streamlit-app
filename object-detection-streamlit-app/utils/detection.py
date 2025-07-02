import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import os

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize object detector
        
        Args:
            model_name: Name of the YOLO model
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for overlapping boxes
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.class_names = []
        
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            if self.model_name == "Custom Model":
                # Load custom model from models directory
                model_path = "models/custom_model.pt"
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    raise FileNotFoundError(f"Custom model not found at {model_path}")
            else:
                # Load pre-trained model
                model_map = {
                    "YOLOv8n": "yolov8n.pt",
                    "YOLOv8s": "yolov8s.pt",
                    "YOLOv8m": "yolov8m.pt"
                }
                self.model = YOLO(model_map.get(self.model_name, "yolov8n.pt"))
            
            # Get class names
            self.class_names = self.model.names
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Perform object detection on image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detection results
        """
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Perform inference
        results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    }
                    detections.append(detection)
        
        return detections
    
    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """
        Get detection statistics
        
        Args:
            detections: List of detection results
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {"total_detections": 0, "classes_detected": {}}
        
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "total_detections": len(detections),
            "classes_detected": class_counts,
            "average_confidence": np.mean([d['confidence'] for d in detections])
        }