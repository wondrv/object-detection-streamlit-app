import os
import yaml

def fix_config_encoding():
    """Fix config file encoding issues"""
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # Default configuration
    default_config = {
        'model': {
            'default_model': 'YOLOv8n',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'custom_model_path': 'models/custom_model.pt'
        },
        'ui': {
            'theme': 'light',
            'sidebar_expanded': True,
            'page_title': 'Object Detection App',
            'page_icon': '🔍'
        },
        'processing': {
            'max_image_size': 1920,
            'max_video_size_mb': 100,
            'supported_image_formats': ['jpg', 'jpeg', 'png', 'bmp'],
            'supported_video_formats': ['mp4', 'avi', 'mov', 'mkv']
        },
        'detection': {
            'draw_confidence': True,
            'draw_class_names': True,
            'box_thickness': 2,
            'font_scale': 0.5,
            'font_thickness': 1
        }
    }
    
    # Write config with proper encoding
    config_path = 'config/config.yaml'
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(default_config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✅ Config file created successfully at: {config_path}")
        
        # Verify by reading it back
        with open(config_path, 'r', encoding='utf-8') as file:
            loaded_config = yaml.safe_load(file)
        print("✅ Config file verified - can be loaded successfully")
        
    except Exception as e:
        print(f"❌ Error creating config file: {str(e)}")

def create_directory_structure():
    """Create required directory structure"""
    directories = [
        'config',
        'models',
        'utils',
        'assets/sample_images',
        'assets/demo_videos',
        'docs',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

if __name__ == "__main__":
    print("🔧 Fixing encoding issues and creating directory structure...")
    create_directory_structure()
    fix_config_encoding()
    print("🎉 Setup completed!")