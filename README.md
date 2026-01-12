# 🛍️ E-Commerce Fashion Image Generation

Generate high-quality fashion product images using deep learning. This project implements two approaches optimized for **RTX 4060 (8GB VRAM)**:

1. **Projected GAN** - Fast unconditional image generation
2. **Stable Diffusion + LoRA** - Text-conditioned image generation

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Generation](#generation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## ✨ Features

| Feature | Projected GAN | Stable Diffusion LoRA |
|---------|--------------|----------------------|
| Training Time | ~2-4 hours | ~4-8 hours |
| VRAM Usage | ~4-6 GB | ~6-8 GB |
| Output Resolution | 256x256 | 512x512 |
| Text Conditioning | ❌ | ✅ |
| Few-shot Learning | ✅ | ✅ |

## 💻 Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or better
- **OS**: Windows 10/11
- **Python**: 3.10+
- **CUDA**: 12.1+

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project
cd C:\Users\Alex\Desktop\root\20.beneav2

# Run setup script (creates venv and installs dependencies)
setup_env.bat
```

Or manually:

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Activate venv first
venv\Scripts\activate

# Download sample data (or use your own images)
python data/download_deepfashion.py --source sample

# Prepare dataset for training
python data/prepare_dataset.py
```

### 3. Train Models

**Option A: Projected GAN (Fast, Unconditional)**
```bash
python projected_gan/train.py
```

**Option B: Stable Diffusion LoRA (Text-Conditioned)**
```bash
python stable_diffusion_lora/train_lora.py
```

### 4. Generate Images

**Projected GAN:**
```bash
python projected_gan/generate.py --checkpoint outputs/projected_gan/checkpoint_final.pt
```

**Stable Diffusion LoRA:**
```bash
python stable_diffusion_lora/generate.py --lora-path outputs/lora/checkpoint-final --prompt "a red dress"
```

## 📊 Dataset Preparation

### Using DeepFashion from Kaggle

1. Create a [Kaggle account](https://www.kaggle.com/)
2. Get your API token from Account → API → Create New Token
3. Place `kaggle.json` in `~/.kaggle/`
4. Run:

```bash
pip install kaggle
python data/download_deepfashion.py --source kaggle
python data/prepare_dataset.py
```

### Using Your Own Images

```bash
# Option 1: Specify local folder
python data/download_deepfashion.py --source local --local-path /path/to/your/images

# Option 2: Copy images to data/raw folder
# Then run prepare script
python data/prepare_dataset.py --source-dir data/raw/your_folder
```

### Dataset Structure After Preparation

```
data/
├── raw/                          # Original images
├── processed/
│   ├── projected_gan/           # 256x256 images for GAN
│   │   ├── img_000000.jpg
│   │   └── ...
│   └── lora/                    # 512x512 images for LoRA
│       ├── images/
│       │   ├── img_000000.jpg
│       │   └── ...
│       └── metadata.jsonl       # Image-caption pairs
```

## 🏋️ Training

### Projected GAN

```bash
# Default training
python projected_gan/train.py

# With custom config
python projected_gan/train.py --config config/projected_gan_config.yaml

# Resume from checkpoint
python projected_gan/train.py --resume outputs/projected_gan/checkpoint_00005000.pt
```

**Training Tips:**
- Default: 1000k images (~2-4 hours on RTX 4060)
- Check samples in `outputs/projected_gan/samples_*.png`
- Monitor with TensorBoard: `tensorboard --logdir outputs/projected_gan/logs`

### Stable Diffusion LoRA

```bash
# Default training
python stable_diffusion_lora/train_lora.py

# With custom config
python stable_diffusion_lora/train_lora.py --config config/lora_config.yaml
```

**Training Tips:**
- Default: 100 epochs with batch size 1
- Effective batch size = 4 (with gradient accumulation)
- Check validation images in `outputs/lora/val_*.png`

## 🎨 Generation

### Projected GAN

```bash
# Generate 16 random images
python projected_gan/generate.py \
    --checkpoint outputs/projected_gan/checkpoint_final.pt \
    --num-samples 16

# Generate with truncation (higher quality, less diversity)
python projected_gan/generate.py \
    --checkpoint outputs/projected_gan/checkpoint_final.pt \
    --truncation 0.5

# Create interpolation video
python projected_gan/generate.py \
    --checkpoint outputs/projected_gan/checkpoint_final.pt \
    --interpolate
```

### Stable Diffusion LoRA

```bash
# Single prompt
python stable_diffusion_lora/generate.py \
    --lora-path outputs/lora/checkpoint-final \
    --prompt "a professional photo of a summer dress"

# Multiple prompts from file
python stable_diffusion_lora/generate.py \
    --lora-path outputs/lora/checkpoint-final \
    --prompts-file my_prompts.txt

# Use default fashion prompts
python stable_diffusion_lora/generate.py \
    --lora-path outputs/lora/checkpoint-final \
    --use-defaults

# Generate grid
python stable_diffusion_lora/generate.py \
    --lora-path outputs/lora/checkpoint-final \
    --use-defaults \
    --grid
```

## 📁 Project Structure

```
20.beneav2/
├── config/
│   ├── projected_gan_config.yaml    # GAN settings
│   └── lora_config.yaml             # LoRA settings
├── data/
│   ├── download_deepfashion.py      # Dataset download
│   ├── prepare_dataset.py           # Data preprocessing
│   ├── raw/                         # Original images
│   └── processed/                   # Prepared datasets
├── projected_gan/
│   ├── __init__.py
│   ├── model.py                     # Generator & Discriminator
│   ├── train.py                     # Training script
│   └── generate.py                  # Generation script
├── stable_diffusion_lora/
│   ├── __init__.py
│   ├── train_lora.py                # LoRA training
│   └── generate.py                  # Generation script
├── utils/
│   ├── __init__.py
│   └── common.py                    # Utility functions
├── outputs/
│   ├── projected_gan/               # GAN checkpoints & samples
│   └── lora/                        # LoRA checkpoints & samples
├── requirements.txt
├── setup_env.bat                    # Windows setup script
├── setup_env.sh                     # Linux/Git Bash setup
└── README.md
```

## ⚙️ Configuration

### Projected GAN (`config/projected_gan_config.yaml`)

Key settings:
```yaml
model:
  img_size: 256            # Output resolution
  latent_dim: 256          # Latent space size

training:
  batch_size: 8            # Batch size (8 fits in 8GB)
  total_kimg: 1000         # Total images (thousands)
  mixed_precision: true    # Enable FP16 for memory savings
```

### LoRA (`config/lora_config.yaml`)

Key settings:
```yaml
lora:
  rank: 128                # LoRA rank (higher = more capacity)
  alpha: 128               # Usually same as rank

training:
  batch_size: 1            # Must be 1 for 8GB VRAM
  gradient_accumulation_steps: 4  # Effective batch = 4
  learning_rate: 1.0e-4
  mixed_precision: "fp16"
  gradient_checkpointing: true
  enable_xformers: true    # Critical for memory!
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Projected GAN:**
- Reduce `batch_size` in config (try 4)
- Reduce `img_size` to 128

**LoRA:**
- Ensure `batch_size: 1`
- Enable `gradient_checkpointing: true`
- Enable `enable_xformers: true`
- Reduce `lora.rank` to 64

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### xformers Installation Failed

```bash
# Try pre-built wheel
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

### Slow Training

- Enable `mixed_precision: true`
- Use `cudnn_benchmark: true`
- Reduce logging frequency
- Use SSD for data storage

## 📚 References

- [Projected GANs Converge Faster](https://arxiv.org/abs/2111.01007) - Sauer et al., 2021
- [Analyzing and Improving StyleGAN](https://arxiv.org/abs/1912.04958) - Karras et al., 2019
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## 📄 License

This project is for educational purposes. Please check individual model and dataset licenses before commercial use.

---

**Author:** DUMITRE Ioan-Alexandru  
**Hardware:** Optimized for RTX 4060 (8GB VRAM)
