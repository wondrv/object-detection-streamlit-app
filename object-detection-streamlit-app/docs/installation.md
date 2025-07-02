# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for dependencies and models
- **GPU**: Optional but recommended (CUDA-compatible for faster inference)

### Software Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **Package Manager**: pip (latest version)

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/wondrv/object-detection-streamlit-app.git
cd object-detection-streamlit-app
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Alternative: Using conda
conda create -n object-detection python=3.8
conda activate object-detection
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation
```bash
# Test basic functionality
python tests/test_url_detection.py

# Check Streamlit installation
streamlit --version
```

## Troubleshooting Installation

### Common Issues

**Python Version Issues:**
```bash
# Check Python version
python --version

# If using wrong version, specify Python 3.8+
python3.8 -m venv venv
```

**Package Conflicts:**
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

**GPU Support Issues:**
```bash
# Check CUDA availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Optional Dependencies

### Development Tools
```bash
pip install pytest black flake8 mypy
```

### Additional ML Libraries
```bash
pip install jupyter matplotlib seaborn
```