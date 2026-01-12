@echo off
REM ============================================
REM E-Commerce Fashion Image Generation
REM Environment Setup Script for Windows
REM ============================================

echo.
echo ========================================
echo  Fashion Image Generation - Setup
echo  Optimized for RTX 4060 (8GB VRAM)
echo ========================================
echo.

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
echo.
echo [1/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv

REM Activate virtual environment
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo [4/4] Installing dependencies (this may take 10-15 minutes)...
echo        Installing PyTorch with CUDA 12.1 support...
pip install -r requirements.txt

REM Verify CUDA is available
echo.
echo ========================================
echo  Verifying GPU Setup
echo ========================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate
echo.
echo To start training Projected GAN:
echo   python projected_gan\train.py
echo.
echo To start training LoRA:
echo   python stable_diffusion_lora\train_lora.py
echo.
pause
