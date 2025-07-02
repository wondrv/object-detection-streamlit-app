import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from utils.file_handler import download_image_from_url, is_valid_url, is_supported_image_url

def test_url_functions():
    """Test URL handling functions"""
    
    print("Testing URL validation...")
    
    # Test valid URLs
    valid_urls = [
        "https://example.com/image.jpg",
        "http://test.com/photo.png",
        "https://domain.org/pic.gif?q=1"
    ]
    
    for url in valid_urls:
        assert is_valid_url(url), f"Valid URL failed: {url}"
        print(f"✅ {url}")
    
    # Test invalid URLs
    invalid_urls = [
        "not_a_url",
        "ftp://example.com/image.jpg",
        "https://",
        ""
    ]
    
    for url in invalid_urls:
        assert not is_valid_url(url), f"Invalid URL passed: {url}"
        print(f"❌ {url}")
    
    print("\nTesting image URL support...")
    
    # Test supported image URLs
    image_urls = [
        "https://example.com/image.jpg",
        "https://example.com/image.png",
        "https://example.com/image.gif"
    ]
    
    for url in image_urls:
        assert is_supported_image_url(url), f"Image URL failed: {url}"
        print(f"✅ {url}")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_url_functions()
