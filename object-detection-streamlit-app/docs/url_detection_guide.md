# URL Detection Feature Guide

## 🔗 Using URL Detection

The URL Detection feature allows you to perform object detection directly on images from the internet without needing to download them manually first.

### How to Use

1. **Navigate to URL Detection Tab**
   - Open the app and click on the "🔗 URL Detection" tab

2. **Enter Image URL**
   - Paste a direct link to an image in the text input field
   - Supported formats: JPG, JPEG, PNG, BMP, GIF, WebP
   - The app will validate the URL format automatically

3. **Detect Objects**
   - Click the "🔍 Detect from URL" button
   - The app will download the image and perform object detection
   - Results will be displayed with bounding boxes and labels

### Supported URL Types

✅ **Supported:**
- Direct image links (e.g., `https://example.com/image.jpg`)
- Images with query parameters (e.g., `https://site.com/pic.png?size=large`)
- Most common image hosting services
- Social media direct image links

❌ **Not Supported:**
- Web page URLs (must be direct image links)
- Images requiring authentication
- Some CDN-protected images
- Non-image URLs

### Example URLs to Try

The app includes several example URLs you can test:

1. **Ultralytics Bus Example**
   - `https://ultralytics.com/images/bus.jpg`
   - Great for testing vehicle detection

2. **YOLOv5 Zidane Example**
   - `https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg`
   - Good for testing person detection

3. **Unsplash Sample**
   - `https://images.unsplash.com/photo-1544568100-847a948585b9?w=800&h=600&fit=crop`
   - High-quality sample image

### Tips for Best Results

1. **Right-click and "Copy Image Address"**
   - This ensures you get the direct image URL

2. **Check URL Format**
   - URLs should end with image extensions (.jpg, .png, etc.)
   - Or be from known image hosting services

3. **Test Connection**
   - The app will show validation status for entered URLs

4. **Download Results**
   - You can download the annotated image with detected objects

### Troubleshooting

**"Error downloading image: Connection error"**
- Check your internet connection
- Try a different image URL

**"URL does not point to an image"**
- Make sure you're using a direct image link
- Try right-clicking the image and copying the image address

**"Request timeout"**
- The server is taking too long to respond
- Try a different image or check your connection

**"HTTP error: 403/404"**
- The image may be protected or not accessible
- Try a different image URL

### Features

- **Real-time URL validation**
- **Automatic image format detection**
- **Download annotated results**
- **Image information display (width, height, format)**
- **Detection statistics**
- **Example URLs for quick testing**

### Security Notes

- The app only downloads images, not web pages
- URLs are validated for proper format
- Downloads timeout after 10 seconds for security
- No personal information is stored or transmitted
