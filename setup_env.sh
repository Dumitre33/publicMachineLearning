#!/bin/bash
# ============================================
# E-Commerce Fashion Image Generation
# Environment Setup Script for Git Bash/WSL
# ============================================

echo ""
echo "========================================"
echo " Fashion Image Generation - Setup"
echo " Optimized for RTX 4060 (8GB VRAM)"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.10+ and add it to PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo "Found $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi
python -m venv venv

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo ""
echo "[4/4] Installing dependencies (this may take 10-15 minutes)..."
echo "       Installing PyTorch with CUDA 12.1 support..."
pip install -r requirements.txt

# Verify CUDA is available
echo ""
echo "========================================"
echo " Verifying GPU Setup"
echo "========================================"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/Scripts/activate  (Git Bash)"
echo "  source venv/bin/activate      (Linux/WSL)"
echo ""
echo "To start training Projected GAN:"
echo "  python projected_gan/train.py"
echo ""
echo "To start training LoRA:"
echo "  python stable_diffusion_lora/train_lora.py"
echo ""
