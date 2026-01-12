"""
Projected GAN Training Script (Bulletproof Version)
====================================================

Train a Projected GAN on DeepFashion for fashion image generation.
Optimized for RTX 4060 (8GB VRAM) with stability fixes.

Features:
- Spectral normalization
- EMA (Exponential Moving Average) for generator
- Gradient clipping
- NaN detection and early stopping
- Learning rate warmup
- FP32 for generator (more stable)
- Checkpointing with recovery

Usage:
    python projected_gan/train.py
    python projected_gan/train.py --config config/projected_gan_config.yaml
"""

import os
import sys
import yaml
import argparse
import random
import copy
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from projected_gan.model import (
    Generator, 
    ProjectedDiscriminator,
    hinge_loss_dis,
    hinge_loss_gen,
    r1_penalty,
)

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available")


# ============================================
# Dataset with Strong Augmentation
# ============================================

class FashionDataset(Dataset):
    """Image folder dataset with strong augmentation for small datasets."""
    
    def __init__(
        self,
        root: str,
        img_size: int = 256,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.img_size = img_size
        
        # Find all images
        self.images = []
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for path in self.root.rglob('*'):
            if path.suffix.lower() in extensions:
                self.images.append(path)
        
        if not self.images:
            raise ValueError(f"No images found in {root}")
        
        print(f"Found {len(self.images)} images")
        
        # Strong augmentation for small datasets
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.1)),  # Slight upscale for crop variety
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            # Return random valid image on error
            return self[random.randint(0, len(self) - 1)]


# ============================================
# EMA (Exponential Moving Average)
# ============================================

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow weights to model (for inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================
# Trainer (Bulletproof)
# ============================================

class Trainer:
    """Bulletproof Projected GAN trainer with stability fixes."""
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda',
    ):
        self.config = config
        self.device = device
        
        # Create models with spectral normalization
        self.G = Generator(
            z_dim=config['model']['latent_dim'],
            img_size=config['model']['img_size'],
            use_spectral_norm=True,
        ).to(device)
        
        self.D = ProjectedDiscriminator(use_spectral_norm=True).to(device)
        
        # EMA for generator (produces smoother outputs)
        self.ema = EMA(self.G, decay=config['training'].get('ema_decay', 0.9999))
        
        # Optimizers with conservative learning rates
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=config['training']['g_lr'],
            betas=tuple(config['training']['betas']),
        )
        
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=config['training']['d_lr'],
            betas=tuple(config['training']['betas']),
        )
        
        # Mixed precision - use FP32 for Generator for stability
        self.use_amp_d = config['training'].get('mixed_precision', True)
        self.use_amp_g = False  # FP32 for generator is more stable
        self.scaler_D = GradScaler('cuda', enabled=self.use_amp_d)
        
        # Training state
        self.step = 0
        self.kimg = 0
        self.nan_count = 0
        self.max_nan_count = config['training'].get('max_nan_count', 20)
        
        # Gradient clipping
        self.grad_clip = config['training'].get('grad_clip', 0.5)
        
        # Learning rate warmup
        self.warmup_steps = config['training'].get('warmup_steps', 500)
        self.base_g_lr = config['training']['g_lr']
        self.base_d_lr = config['training']['d_lr']
        
        # Output directory
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed latents for consistent samples
        self.fixed_z = torch.randn(
            config['output'].get('num_samples', 16),
            config['model']['latent_dim'],
            device=device,
        )
        
        # Tensorboard
        self.writer = None
        if config['logging'].get('use_tensorboard', True) and HAS_TENSORBOARD:
            log_dir = self.output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir)
    
    def get_lr_scale(self) -> float:
        """Get learning rate scale for warmup."""
        if self.step >= self.warmup_steps:
            return 1.0
        return (self.step + 1) / self.warmup_steps
    
    def update_lr(self):
        """Update learning rate with warmup."""
        scale = self.get_lr_scale()
        for param_group in self.opt_G.param_groups:
            param_group['lr'] = self.base_g_lr * scale
        for param_group in self.opt_D.param_groups:
            param_group['lr'] = self.base_d_lr * scale
    
    def train_step(self, real_images: torch.Tensor) -> dict:
        """Single training step with stability checks."""
        batch_size = real_images.shape[0]
        
        # Update learning rate (warmup)
        self.update_lr()
        
        # ============================================
        # Train Discriminator
        # ============================================
        self.opt_D.zero_grad()
        
        with autocast('cuda', enabled=self.use_amp_d):
            # Generate fake images
            z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
            
            # Generator runs in FP32 for stability
            with autocast('cuda', enabled=False):
                fake_images = self.G(z.float())
            
            # Discriminator outputs
            real_outputs = self.D(real_images)
            fake_outputs = self.D(fake_images.detach())
            
            # Hinge loss
            d_loss = hinge_loss_dis(real_outputs, fake_outputs)
        
        # Check for NaN
        if torch.isnan(d_loss) or torch.isinf(d_loss):
            self.nan_count += 1
            print(f"\n[!] NaN detected in D loss at step {self.step} (count: {self.nan_count}/{self.max_nan_count})")
            if self.nan_count >= self.max_nan_count:
                raise RuntimeError("Training collapsed! Too many NaN losses. Try lowering learning rate or check data.")
            return {'d_loss': float('nan'), 'g_loss': float('nan'), 'r1_loss': 0.0}
        
        # Backward and step
        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.unscale_(self.opt_D)
        clip_grad_norm_(self.D.parameters(), self.grad_clip)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        
        # ============================================
        # R1 regularization (every N steps)
        # ============================================
        r1_loss = torch.tensor(0.0, device=self.device)
        r1_interval = self.config['training'].get('d_reg_interval', 8)
        
        if self.step > 0 and self.step % r1_interval == 0:
            try:
                self.opt_D.zero_grad()
                
                # Create float32 copy for gradient computation
                real_images_r1 = real_images.detach().float().requires_grad_(True)
                
                # Run discriminator in FP32 for stable gradients
                with autocast('cuda', enabled=False):
                    real_outputs = self.D(real_images_r1)
                
                # Compute R1 penalty
                r1_grads = torch.autograd.grad(
                    outputs=sum([o.sum() for o in real_outputs]),
                    inputs=real_images_r1,
                    create_graph=True,
                )[0]
                
                r1_loss = r1_grads.pow(2).reshape(r1_grads.size(0), -1).sum(1).mean()
                r1_scaled = self.config['training']['r1_gamma'] * r1_loss
                r1_scaled.backward()
                clip_grad_norm_(self.D.parameters(), self.grad_clip)
                self.opt_D.step()
            except RuntimeError as e:
                if 'grad' in str(e).lower():
                    pass  # Skip R1 if gradient fails
                else:
                    raise e
        
        # ============================================
        # Train Generator (in FP32 for stability)
        # ============================================
        self.opt_G.zero_grad()
        
        # Generator runs in FP32
        z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device).float()
        fake_images = self.G(z)
        
        # Discriminator can use AMP
        with autocast('cuda', enabled=self.use_amp_d):
            fake_outputs = self.D(fake_images)
            g_loss = hinge_loss_gen(fake_outputs)
        
        # Check for NaN
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            self.nan_count += 1
            print(f"\n[!] NaN detected in G loss at step {self.step} (count: {self.nan_count}/{self.max_nan_count})")
            if self.nan_count >= self.max_nan_count:
                raise RuntimeError("Training collapsed! Too many NaN losses.")
            return {'d_loss': d_loss.item(), 'g_loss': float('nan'), 'r1_loss': 0.0}
        
        # Backward (no scaler for FP32)
        g_loss.backward()
        clip_grad_norm_(self.G.parameters(), self.grad_clip)
        self.opt_G.step()
        
        # Update EMA
        self.ema.update(self.G)
        
        # Reset NaN counter on successful step
        self.nan_count = 0
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'r1_loss': r1_loss.item() if isinstance(r1_loss, torch.Tensor) else r1_loss,
        }
    
    @torch.no_grad()
    def generate_samples(self, filename: str = 'samples.png', use_ema: bool = True):
        """Generate and save sample images using EMA weights."""
        self.G.eval()
        
        # Use EMA weights for better quality
        if use_ema:
            self.ema.apply_shadow(self.G)
        
        samples = self.G(self.fixed_z)
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0, 1)  # Ensure valid range
        
        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, self.output_dir / filename)
        
        # Restore original weights
        if use_ema:
            self.ema.restore(self.G)
        
        self.G.train()
        return samples
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save training checkpoint including EMA."""
        checkpoint = {
            'step': self.step,
            'kimg': self.kimg,
            'nan_count': self.nan_count,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'ema_shadow': self.ema.shadow,
            'config': self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.step = checkpoint['step']
        self.kimg = checkpoint['kimg']
        self.nan_count = checkpoint.get('nan_count', 0)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        
        # Load EMA if available
        if 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def train(self, dataloader: DataLoader):
        """Full training loop with bulletproof stability."""
        config = self.config['training']
        total_kimg = config['total_kimg']
        batch_size = config['batch_size']
        
        print("\n" + "="*60)
        print(" BULLETPROOF Projected GAN Training")
        print("="*60)
        print(f"Total images: {total_kimg}k")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"Generator LR: {self.base_g_lr}")
        print(f"Discriminator LR: {self.base_d_lr}")
        print(f"Gradient clip: {self.grad_clip}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"EMA decay: {self.ema.decay}")
        print(f"Mixed precision (D): {self.use_amp_d}")
        print(f"Mixed precision (G): {self.use_amp_g} (FP32 for stability)")
        print("="*60 + "\n")
        
        # Progress bar
        total_steps = int(total_kimg * 1000 / batch_size)
        pbar = tqdm(total=total_steps, initial=self.step, desc="Training")
        
        # Training loop
        try:
            while self.kimg < total_kimg:
                for real_images in dataloader:
                    real_images = real_images.to(self.device)
                    
                    # Training step
                    losses = self.train_step(real_images)
                    
                    # Update counters
                    self.step += 1
                    self.kimg = self.step * batch_size / 1000
                    pbar.update(1)
                    
                    # Logging
                    if self.step % config.get('log_interval', 100) == 0:
                        d_str = f"{losses['d_loss']:.3f}" if not np.isnan(losses['d_loss']) else "NaN"
                        g_str = f"{losses['g_loss']:.3f}" if not np.isnan(losses['g_loss']) else "NaN"
                        lr_scale = self.get_lr_scale()
                        
                        pbar.set_postfix({
                            'D': d_str,
                            'G': g_str,
                            'LR': f"{lr_scale:.2f}",
                        })
                        
                        if self.writer and not np.isnan(losses['d_loss']):
                            self.writer.add_scalar('Loss/D', losses['d_loss'], self.step)
                            self.writer.add_scalar('Loss/G', losses['g_loss'], self.step)
                            self.writer.add_scalar('Loss/R1', losses['r1_loss'], self.step)
                            self.writer.add_scalar('LR/scale', lr_scale, self.step)
                    
                    # Generate samples
                    sample_interval = config.get('sample_interval', 1000)
                    if self.step % sample_interval == 0:
                        samples = self.generate_samples(f'samples_{self.step:08d}.png')
                        
                        if self.writer:
                            self.writer.add_images('Samples', samples, self.step)
                    
                    # Save checkpoint
                    checkpoint_interval = config.get('checkpoint_interval', 5000)
                    if self.step % checkpoint_interval == 0:
                        self.save_checkpoint(f'checkpoint_{self.step:08d}.pt')
                        print(f"\n[*] Checkpoint saved at step {self.step}")
                    
                    # Check if done
                    if self.kimg >= total_kimg:
                        break
        
        except RuntimeError as e:
            print(f"\n[!] Training stopped: {e}")
            print("[*] Saving emergency checkpoint...")
            self.save_checkpoint('checkpoint_emergency.pt')
            self.generate_samples('samples_emergency.png')
            raise
        
        pbar.close()
        
        # Save final checkpoint
        self.save_checkpoint('checkpoint_final.pt')
        self.generate_samples('samples_final.png')
        
        print("\n" + "="*60)
        print(" Training Complete!")
        print("="*60)
        print(f"Final checkpoint: {self.output_dir / 'checkpoint_final.pt'}")
        print(f"Samples: {self.output_dir / 'samples_final.png'}")


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'projected_gan_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Projected GAN (Bulletproof)")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data', type=str, help='Path to training data')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override data path if provided
    if args.data:
        config['data']['train_dir'] = args.data
    
    # Set seed
    set_seed(config['hardware'].get('seed', 42))
    
    # Device
    device = config['hardware'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Enable cudnn benchmark
    if device == 'cuda':
        torch.backends.cudnn.benchmark = config['hardware'].get('cudnn_benchmark', True)
    
    # Print GPU info
    if device == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataset
    data_dir = PROJECT_ROOT / config['data']['train_dir']
    
    if not data_dir.exists():
        print(f"\nERROR: Training data not found at {data_dir}")
        print("Please run the data preparation scripts first:")
        print("  python data/download_deepfashion.py")
        print("  python data/prepare_dataset.py")
        sys.exit(1)
    
    dataset = FashionDataset(
        root=data_dir,
        img_size=config['model']['img_size'],
        augment=config['data']['augmentation']['enabled'],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        drop_last=True,
    )
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
