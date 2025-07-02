import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import yaml
import sys
import io
import io

# Add utils to path
sys.path.append('utils')

# Import custom utilities with error handling
try:
    from utils.detection import ObjectDetector
    from utils.visualization import draw_detections
    from utils.file_handler import save_uploaded_file, load_config, download_image_from_url, is_supported_image_url
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.info("Please make sure all required files are in the utils/ directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'config' not in st.session_state:
    st.session_state.config = None

def load_app_config():
    """Load application configuration with error handling"""
    try:
        config = load_config("config/config.yaml")
        st.session_state.config = config
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        # Use minimal default config
        default_config = {
            'model': {'default_model': 'YOLOv8n', 'confidence_threshold': 0.5, 'nms_threshold': 0.4},
            'ui': {'theme': 'light', 'sidebar_expanded': True},
            'processing': {'max_image_size': 1920, 'max_video_size_mb': 100}
        }
        st.session_state.config = default_config
        return default_config

def main():
    st.title("🔍 Object Detection dengan Streamlit")
    st.markdown("**Detect objects from uploaded images, URLs, videos, or webcam!**")
    st.markdown("---")
    
    # Load configuration
    if st.session_state.config is None:
        config = load_app_config()
    else:
        config = st.session_state.config
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Pilih Model:",
        options=["YOLOv8n", "YOLOv8s", "YOLOv8m", "Custom Model"],
        index=0
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=1.0,
        value=config.get('model', {}).get('confidence_threshold', 0.5),
        step=0.1
    )
    
    # NMS threshold
    nms_threshold = st.sidebar.slider(
        "NMS Threshold:",
        min_value=0.1,
        max_value=1.0,
        value=config.get('model', {}).get('nms_threshold', 0.4),
        step=0.1
    )
    
    # Initialize detector
    if st.sidebar.button("Load Model") or st.session_state.detector is None:
        with st.spinner("Loading model..."):
            try:
                st.session_state.detector = ObjectDetector(
                    model_name=model_choice,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please check your internet connection and try again.")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Detection", "🔗 URL Detection", "🎥 Video Detection", "📹 Webcam Detection"])
    
    with tab1:
        image_detection_tab()
    
    with tab2:
        url_detection_tab()
    
    with tab3:
        video_detection_tab()
        
    with tab4:
        webcam_detection_tab()

def image_detection_tab():
    st.header("📷 Image Object Detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload gambar:",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Objects"):
                if st.session_state.detector is not None:
                    with st.spinner("Detecting objects..."):
                        try:
                            # Convert PIL to OpenCV format
                            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Perform detection
                            results = st.session_state.detector.detect(opencv_image)
                            
                            # Draw detections
                            annotated_image = draw_detections(opencv_image, results)
                            
                            # Convert back to RGB for display
                            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            
                            # Display results
                            st.subheader("Detection Results")
                            st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)
                            
                            # Display detection statistics
                            st.subheader("Detection Statistics")
                            detection_stats = st.session_state.detector.get_detection_stats(results)
                            st.json(detection_stats)
                            
                        except Exception as e:
                            st.error(f"Error during detection: {str(e)}")
                else:
                    st.error("Please load a model first!")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def url_detection_tab():
    st.header("🔗 URL Image Object Detection")
    st.info("Enter an image URL to detect objects directly from the web")
    
    # URL input
    url_input = st.text_input(
        "Enter image URL:",
        placeholder="https://example.com/image.jpg",
        help="Paste a direct link to an image (JPG, PNG, BMP, GIF, WebP)"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        detect_button = st.button("🔍 Detect from URL", type="primary")
    
    with col2:
        if url_input:
            if is_supported_image_url(url_input):
                st.success("✅ Valid image URL format")
            else:
                st.error("❌ Invalid URL format")
    
    if detect_button and url_input:
        if not is_supported_image_url(url_input):
            st.error("Please enter a valid image URL")
            return
        
        with st.spinner("Downloading image from URL..."):
            # Download image from URL
            image, file_ext, error = download_image_from_url(url_input)
            
            if error:
                st.error(f"Error downloading image: {error}")
                st.info("Tips:")
                st.info("• Make sure the URL is accessible and points directly to an image")
                st.info("• Try right-clicking on an image and selecting 'Copy image address'")
                st.info("• Some websites may block automated downloads")
                return
            
            if image is None:
                st.error("Could not load image from URL")
                return
        
        st.success("Image downloaded successfully!")
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption=f"Image from: {url_input}", use_container_width=True)
        
        # Show image info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Format", file_ext.upper())
        
        if st.session_state.detector is not None:
            with st.spinner("Detecting objects..."):
                try:
                    # Convert PIL to OpenCV format
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Perform detection
                    results = st.session_state.detector.detect(opencv_image)
                    
                    # Draw detections
                    annotated_image = draw_detections(opencv_image, results)
                    
                    # Convert back to RGB for display
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    # Display results
                    st.subheader("Detection Results")
                    st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)
                    
                    # Display detection statistics
                    st.subheader("Detection Statistics")
                    detection_stats = st.session_state.detector.get_detection_stats(results)
                    
                    if detection_stats["total_detections"] > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Objects", detection_stats["total_detections"])
                        with col2:
                            st.metric("Avg Confidence", f"{detection_stats['average_confidence']:.2f}")
                        
                        # Show detected classes
                        st.write("**Detected Classes:**")
                        for class_name, count in detection_stats["classes_detected"].items():
                            st.write(f"• {class_name}: {count}")
                    else:
                        st.info("No objects detected in this image")
                    
                    # Provide download option for annotated image
                    st.subheader("Download Results")
                    
                    # Convert annotated image to bytes for download
                    annotated_pil = Image.fromarray(annotated_image_rgb)
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"detected_objects_{url_input.split('/')[-1].split('.')[0]}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
        else:
            st.error("Please load a model first!")

def video_detection_tab():
    st.header("🎥 Video Object Detection")
    st.info("Video detection feature is available but may require additional processing time.")
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Upload video:",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        try:
            # Save uploaded video
            temp_file = save_uploaded_file(uploaded_video)
            
            # Display video info
            st.subheader("Video Information")
            cap = cv2.VideoCapture(temp_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FPS", f"{fps:.2f}")
            with col2:
                st.metric("Total Frames", total_frames)
            with col3:
                st.metric("Duration", f"{duration:.2f}s")
            
            cap.release()
            
            if st.button("🔍 Process Video"):
                if st.session_state.detector is not None:
                    st.warning("Video processing may take several minutes depending on video length.")
                    process_video(temp_file)
                else:
                    st.error("Please load a model first!")
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

def webcam_detection_tab():
    st.header("📹 Real-time Webcam Detection")
    st.info("Note: Webcam detection requires additional setup for deployment.")
    
    # Placeholder for webcam functionality
    if st.button("Start Webcam Detection"):
        st.warning("Webcam detection is not implemented in this demo. Please refer to the documentation for implementation details.")

def process_video(video_path):
    """Process video file and detect objects"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            st.error("Could not read video file. Please try a different format.")
            return
        
        # Create output video path
        output_path = "temp_output.mp4"
        
        # Video writer setup
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = None
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        successful_frames = 0
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Perform detection
                results = st.session_state.detector.detect(frame)
                
                # Draw detections
                annotated_frame = draw_detections(frame, results)
                
                # Initialize video writer
                if out is None:
                    height, width = annotated_frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Write frame
                out.write(annotated_frame)
                successful_frames += 1
                
            except Exception as e:
                st.warning(f"Error processing frame {frame_count}: {str(e)}")
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} (Success: {successful_frames})")
        
        cap.release()
        if out is not None:
            out.release()
        
        # Display results
        if successful_frames > 0:
            st.success(f"Video processing completed! Processed {successful_frames}/{total_frames} frames.")
            
            # Provide download link
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
        else:
            st.error("No frames were processed successfully.")
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()