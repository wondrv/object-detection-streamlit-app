# Usage Instructions

## Getting Started

### Launch Application
```bash
# Navigate to project directory
cd object-detection-streamlit-app

# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch application
streamlit run app.py

# Open browser and navigate to http://localhost:8501
```

## Feature Guide

### 1. Image Detection

**Step-by-step Process:**
1. **Navigate to Image Detection Tab**
   - Select the "📷 Image Detection" tab from the main interface
   
2. **Upload Image**
   - Supported formats: JPG, JPEG, PNG, BMP, WebP
   - Maximum file size: 10MB
   - Drag and drop or use file uploader

3. **Configure Detection Parameters**
   - **Confidence Threshold**: 0.1 - 1.0 (default: 0.5)
   - **NMS Threshold**: 0.1 - 1.0 (default: 0.4)
   - **Model Selection**: Choose from available YOLO models

4. **Run Detection**
   - Click "🔍 Detect Objects" button
   - Wait for processing (typically 1-3 seconds)
   - View annotated results and statistics

### 2. URL Detection

**Process:**
1. **Navigate to URL Detection Tab**
   - Select "🔗 URL Detection" from the tab menu

2. **Enter Image URL**
   ```bash
   # Example URLs you can try:
   https://ultralytics.com/images/bus.jpg
   https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
   ```
   - Paste direct image URL in the text input
   - Use provided example URLs for quick testing

3. **Execute Detection**
   - Click "🔍 Detect from URL" button
   - System will download and process the image
   - Results displayed with download option

### 3. Video Processing

**Workflow:**
1. **Upload Video File**
   - Supported formats: MP4, AVI, MOV, MKV
   - Maximum size: 100MB (configurable)

2. **Processing Options**
   - **Frame Sampling**: Process every Nth frame
   - **Output Quality**: Choose output resolution
   - **Detection Overlay**: Enable/disable bounding boxes

3. **Download Results**
   - Processed video with detection overlays
   - Frame-by-frame detection statistics
   - Summary report in JSON format

### 4. Model Configuration

**Available Models:**
- **YOLOv8n**: Fast inference, standard accuracy (6MB)
- **YOLOv8s**: Balanced speed and accuracy (22MB)
- **YOLOv8m**: High accuracy, slower inference (52MB)
- **Custom Model**: Use your own trained models

**Parameter Tuning:**
- **Confidence Threshold**: Minimum confidence for detections
- **NMS Threshold**: Non-maximum suppression for overlapping boxes
- **Custom Classes**: Filter specific object classes

## Advanced Usage

### Custom Model Integration

1. **Prepare Model**
   ```bash
   # Place your model in models directory
   cp your_model.pt models/custom_model.pt
   ```

2. **Update Configuration**
   ```yaml
   # config/config.yaml
   model:
     custom_model_path: "models/custom_model.pt"
   ```

3. **Select in Application**
   - Choose "Custom Model" from dropdown
   - Application loads automatically

### Batch Processing

For processing multiple files programmatically:

```python
from utils.detection import ObjectDetector
import cv2
import os

# Initialize detector
detector = ObjectDetector("YOLOv8n", 0.5, 0.4)

# Process directory of images
input_dir = "path/to/images"
output_dir = "path/to/results"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        # Detect objects
        detections = detector.detect(image)
        
        # Save results
        # (Add your processing logic here)
```

## Tips and Best Practices

### Performance Optimization
- Use GPU acceleration when available
- Choose appropriate model size for your use case
- Optimize image resolution for better speed
- Clear cache regularly for better memory usage

### Accuracy Improvement
- Adjust confidence threshold based on your needs
- Use higher resolution models for better accuracy
- Consider custom training for specific domains
- Fine-tune NMS threshold for overlapping objects

### Troubleshooting
- Check console output for error messages
- Verify image format compatibility
- Ensure sufficient memory for large files
- Update dependencies if encountering issues