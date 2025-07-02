import tempfile
import os
import yaml
from typing import Dict, Any
import streamlit as st
import requests
import re
from PIL import Image
import io

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with proper encoding handling
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Try UTF-8 encoding first
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except UnicodeDecodeError:
        try:
            # Fallback to utf-8-sig for BOM issues
            with open(config_path, 'r', encoding='utf-8-sig') as file:
                config = yaml.safe_load(file)
            return config
        except UnicodeDecodeError:
            try:
                # Last fallback to latin-1
                with open(config_path, 'r', encoding='latin-1') as file:
                    config = yaml.safe_load(file)
                return config
            except Exception as e:
                st.error(f"Error loading config with encoding fallbacks: {str(e)}")
                return get_default_config()
    except FileNotFoundError:
        st.warning(f"Config file not found: {config_path}. Using default settings.")
        # Create default config file
        create_default_config_file(config_path)
        return get_default_config()
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'default_model': 'YOLOv8n',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        },
        'ui': {
            'theme': 'light',
            'sidebar_expanded': True
        },
        'processing': {
            'max_image_size': 1920,
            'max_video_size_mb': 100
        }
    }

def create_default_config_file(config_path: str) -> None:
    """
    Create default config file if it doesn't exist
    
    Args:
        config_path: Path where to create config file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config
        default_config = get_default_config()
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(default_config, file, default_flow_style=False, allow_unicode=True)
        
        st.info(f"Created default config file at: {config_path}")
    except Exception as e:
        st.error(f"Error creating default config file: {str(e)}")

def create_download_link(file_path: str, download_name: str) -> None:
    """
    Create download link for processed file
    
    Args:
        file_path: Path to file
        download_name: Name for download
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download {download_name}",
                data=f.read(),
                file_name=download_name,
                mime="application/octet-stream"
            )

def download_image_from_url(url: str) -> tuple:
    """
    Download image from URL and return PIL Image object and file extension
    
    Args:
        url: URL of the image
        
    Returns:
        Tuple of (PIL Image object, file extension, error message)
    """
    try:
        # Validate URL format
        if not is_valid_url(url):
            return None, None, "Invalid URL format"
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the image with timeout
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check if content type is image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None, None, "URL does not point to an image"
        
        # Get file extension from content type or URL
        file_ext = get_file_extension_from_url(url, content_type)
        
        # Load image from response content
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        return image, file_ext, None
        
    except requests.exceptions.Timeout:
        return None, None, "Request timeout - the server took too long to respond"
    except requests.exceptions.ConnectionError:
        return None, None, "Connection error - please check your internet connection"
    except requests.exceptions.HTTPError as e:
        return None, None, f"HTTP error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, None, f"Request error: {str(e)}"
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def is_valid_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def get_file_extension_from_url(url: str, content_type: str) -> str:
    """
    Get file extension from URL or content type
    
    Args:
        url: Image URL
        content_type: HTTP content type
        
    Returns:
        File extension
    """
    # Try to get extension from URL first
    url_ext = url.split('.')[-1].lower().split('?')[0]  # Remove query parameters
    if url_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']:
        return url_ext
    
    # Get extension from content type
    content_type_map = {
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'image/png': 'png',
        'image/bmp': 'bmp',
        'image/gif': 'gif',
        'image/webp': 'webp'
    }
    
    return content_type_map.get(content_type.lower(), 'jpg')

def is_supported_image_url(url: str) -> bool:
    """
    Check if URL potentially points to a supported image format
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be a supported image
    """
    if not is_valid_url(url):
        return False
    
    # Check file extension in URL
    url_lower = url.lower()
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    for ext in supported_extensions:
        if ext in url_lower:
            return True
    
    # If no extension found, still allow as it might be a dynamic URL
    return True