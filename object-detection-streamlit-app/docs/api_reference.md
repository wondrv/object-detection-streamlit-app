# API Reference

## Core Classes

### ObjectDetector

```python
class ObjectDetector:
    """Main object detection class using YOLOv8"""
    
    def __init__(self, model_name: str, confidence_threshold: float, nms_threshold: float):
        """
        Initialize object detector
        
        Args:
            model_name: Name of YOLO model ('YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'Custom Model')
            confidence_threshold: Confidence threshold for detections (0.0-1.0)
            nms_threshold: Non-maximum suppression threshold (0.0-1.0)
        """
    
    def load_model(self) -> None:
        """Load the specified YOLO model"""
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Perform object detection on image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] 
            - confidence: Detection confidence
            - class_id: Class identifier
            - class_name: Human-readable class name
        """
    
    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """
        Generate detection statistics
        
        Args:
            detections: List of detection results
            
        Returns:
            Statistics dictionary with counts and confidence metrics
        """
```

### Visualization Module

```python
def draw_detections(image: np.ndarray, detections: List[Dict], 
                   box_thickness: int = 2, font_scale: float = 0.5) -> np.ndarray:
    """
    Draw detection results on image
    
    Args:
        image: Input image (BGR format)
        detections: Detection results
        box_thickness: Bounding box line thickness
        font_scale: Text font scale
        
    Returns:
        Annotated image with bounding boxes and labels
    """
```

### File Handling Module

```python
def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded Streamlit file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved temporary file
    """

def download_image_from_url(url: str) -> Tuple[Image.Image, str, str]:
    """
    Download image from URL with validation
    
    Args:
        url: Image URL to download
        
    Returns:
        Tuple of (PIL Image, file extension, error message)
    """

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration with encoding handling
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """

def is_valid_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL format
    """

def is_supported_image_url(url: str) -> bool:
    """
    Check if URL points to supported image format
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be a supported image
    """
```

## Configuration Schema

### Main Configuration (config.yaml)

```yaml
model:
  default_model: str              # Default model name
  confidence_threshold: float     # Default confidence (0.0-1.0)
  nms_threshold: float           # Default NMS threshold (0.0-1.0)
  custom_model_path: str         # Path to custom model file

ui:
  theme: str                     # UI theme ('light' or 'dark')
  sidebar_expanded: bool         # Default sidebar state
  page_title: str               # Browser tab title
  page_icon: str                # Browser tab icon

processing:
  max_image_size: int           # Maximum image dimension
  max_video_size_mb: int        # Maximum video file size (MB)
  supported_image_formats: list # Allowed image extensions
  supported_video_formats: list # Allowed video extensions

detection:
  draw_confidence: bool         # Show confidence scores
  draw_class_names: bool        # Show class labels
  box_thickness: int           # Bounding box line width
  font_scale: float            # Text font size
  font_thickness: int          # Text line thickness
```

## Usage Examples

### Basic Detection Example

```python
from utils.detection import ObjectDetector
from utils.visualization import draw_detections
import cv2

# Initialize detector
detector = ObjectDetector(
    model_name="YOLOv8n",
    confidence_threshold=0.5,
    nms_threshold=0.4
)

# Load and process image
image = cv2.imread("path/to/image.jpg")
detections = detector.detect(image)

# Draw results
annotated_image = draw_detections(image, detections)

# Save result
cv2.imwrite("result.jpg", annotated_image)

# Print detection info
for detection in detections:
    print(f"Found {detection['class_name']} with confidence {detection['confidence']:.2f}")
```

### Custom Model Integration

```python
# Use custom trained model
detector = ObjectDetector(
    model_name="Custom Model",
    confidence_threshold=0.6,
    nms_threshold=0.4
)

# The detector will automatically load from config.yaml custom_model_path
```

### URL Detection Example

```python
from utils.file_handler import download_image_from_url, is_supported_image_url
import cv2
import numpy as np

# Validate and download image
url = "https://example.com/image.jpg"
if is_supported_image_url(url):
    pil_image, file_ext, error = download_image_from_url(url)
    
    if pil_image:
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Perform detection
        detections = detector.detect(opencv_image)
```

### Configuration Loading

```python
from utils.file_handler import load_config

# Load application configuration
config = load_config("config/config.yaml")

# Access configuration values
default_model = config['model']['default_model']
confidence = config['model']['confidence_threshold']
max_file_size = config['processing']['max_video_size_mb']
```

## Data Structures

### Detection Result Format

```python
detection = {
    'bbox': [x1, y1, x2, y2],      # Bounding box coordinates
    'confidence': 0.85,             # Detection confidence (0.0-1.0)
    'class_id': 0,                  # Numeric class identifier
    'class_name': 'person'          # Human-readable class name
}
```

### Detection Statistics Format

```python
stats = {
    'total_detections': 5,                    # Total number of objects found
    'class_counts': {                         # Count per class
        'person': 2,
        'car': 1,
        'bicycle': 2
    },
    'average_confidence': 0.78,               # Mean confidence score
    'confidence_range': [0.65, 0.92],        # Min and max confidence
    'processing_time': 1.23                   # Processing time in seconds
}
```

## Error Handling

### Common Exceptions

```python
# Model loading errors
try:
    detector = ObjectDetector("YOLOv8n", 0.5, 0.4)
except Exception as e:
    print(f"Model loading failed: {e}")

# File handling errors
try:
    image = cv2.imread("image.jpg")
    detections = detector.detect(image)
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Detection failed: {e}")

# URL download errors
try:
    image, ext, error = download_image_from_url(url)
    if error:
        print(f"Download failed: {error}")
except Exception as e:
    print(f"URL processing failed: {e}")
```

## Performance Considerations

### Memory Management

```python
# Clear model cache when switching models
detector = None  # Release current model
detector = ObjectDetector("YOLOv8m", 0.5, 0.4)  # Load new model

# Process large videos in chunks
def process_video_chunks(video_path, chunk_size=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
            
        # Process chunk
        for frame in frames:
            detections = detector.detect(frame)
            # Process detections...
        
        frame_count += len(frames)
```

### GPU Acceleration

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("GPU not available, using CPU")

# Monitor GPU memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    memory_cached = torch.cuda.memory_reserved() / 1024**2      # MB
    print(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Cached: {memory_cached:.1f}MB")
```