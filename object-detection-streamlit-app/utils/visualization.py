import cv2
import numpy as np
from typing import List, Dict, Tuple
import random

def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """Generate random colors for different classes"""
    colors = []
    for _ in range(num_classes):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        colors.append(color)
    return colors

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw detection results on image
    
    Args:
        image: Input image
        detections: List of detection results
        
    Returns:
        Annotated image
    """
    if not detections:
        return image.copy()
    
    # Create a copy of the image
    annotated_image = image.copy()
    
    # Generate colors for classes
    unique_classes = list(set([d['class_id'] for d in detections]))
    colors = generate_colors(len(unique_classes))
    class_colors = {class_id: colors[i] for i, class_id in enumerate(unique_classes)}
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        class_id = detection['class_id']
        
        # Get color for this class
        color = class_colors[class_id]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (bbox[0], bbox[1] - label_height - baseline),
            (bbox[0] + label_width, bbox[1]),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_image,
            label,
            (bbox[0], bbox[1] - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return annotated_image

def create_detection_summary(detections: List[Dict]) -> np.ndarray:
    """
    Create a summary visualization of detections
    
    Args:
        detections: List of detection results
        
    Returns:
        Summary image
    """
    if not detections:
        return np.zeros((100, 400, 3), dtype=np.uint8)
    
    # Create summary image
    summary_height = 50 + len(set([d['class_name'] for d in detections])) * 30
    summary_image = np.ones((summary_height, 400, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(summary_image, "Detection Summary", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Class counts
    class_counts = {}
    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    y_pos = 60
    for class_name, count in class_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(summary_image, text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 30
    
    return summary_image