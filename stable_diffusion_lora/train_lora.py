"""
Stable Diffusion LoRA Training Script
=====================================

Fine-tune Stable Diffusion v1.5 with LoRA on DeepFashion dataset.
Optimized for RTX 4060 (8GB VRAM).

Usage:
    python stable_diffusion_lora/train_lora.py
    python stable_diffusion_lora/train_lora.py --config config/lora_config.yaml

Memory Optimizations:
- LoRA (only trains adapter layers)
- FP16 mixed precision
- Gradient checkpointing
- xformers attention
- Batch size 1 with gradient accumulation
"""

import os
import sys
import json
import yaml
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Hugging Face imports
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

# Try to import optional dependencies
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    print("Warning: accelerate not available")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================
# Dataset
# ============================================

class FashionLoRADataset(Dataset):
    """
    Dataset for LoRA training with image-caption pairs.
    Expects metadata.jsonl with {"file_name": "...", "text": "..."}
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Load metadata
        self.samples = []
        metadata_path = self.data_dir / 'metadata.jsonl'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    img_path = self.data_dir / 'images' / item['file_name']
                    if img_path.exists():
                        self.samples.append({
                            'image_path': img_path,
                            'caption': item['text'],
                        })
        else:
            # Fallback: use all images with default caption
            image_dir = self.data_dir / 'images'
            if not image_dir.exists():
                image_dir = self.data_dir
            
            default_caption = "a high quality fashion photograph of clothing"
            
            for img_path in image_dir.rglob('*.jpg'):
                self.samples.append({
                    'image_path': img_path,
                    'caption': default_caption,
                })
        
        if not self.samples:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Resize
        if self.center_crop:
            # Resize maintaining aspect ratio, then center crop
            ratio = self.resolution / min(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
            # Center crop
            left = (image.size[0] - self.resolution) // 2
            top = (image.size[1] - self.resolution) // 2
            image = image.crop((left, top, left + self.resolution, top + self.resolution))
        else:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Random horizontal flip
        if self.random_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Tokenize caption
        tokens = self.tokenizer(
            sample['caption'],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            'pixel_values': image,
            'input_ids': tokens.input_ids.squeeze(0),
        }


# ============================================
# Trainer
# ============================================

class LoRATrainer:
    """Stable Diffusion LoRA trainer optimized for 8GB VRAM."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        print("\n" + "="*50)
        print(" Loading Stable Diffusion v1.5")
        print("="*50)
        
        # Load models
        model_id = config['model']['name']
        
        print("Loading tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        
        print("Loading text encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        ).to(device)
        
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae"
        ).to(device, dtype=torch.float16)
        
        print("Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        ).to(device, dtype=torch.float16)
        
        print("Loading scheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Apply LoRA to UNet
        print("\nApplying LoRA configuration...")
        lora_config = LoraConfig(
            r=config['lora']['rank'],
            lora_alpha=config['lora']['alpha'],
            init_lora_weights="gaussian",
            target_modules=config['lora']['target_modules'],
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Enable memory optimizations
        if config['training'].get('gradient_checkpointing', True):
            self.unet.enable_gradient_checkpointing()
        
        if config['training'].get('enable_xformers', True):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                print("[OK] xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
            weight_decay=config['training']['adam_weight_decay'],
            eps=config['training']['adam_epsilon'],
        )
        
        # Output directory
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = None
        if config['logging'].get('use_tensorboard', True) and HAS_TENSORBOARD:
            log_dir = self.output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.global_step = 0
        
        print("\n[OK] Model loaded and ready for training")
    
    def train_step(self, batch: dict) -> float:
        """Single training step."""
        # Move batch to device
        pixel_values = batch['pixel_values'].to(self.device, dtype=torch.float16)
        input_ids = batch['input_ids'].to(self.device)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Encode text
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample timestep
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample
        
        # Loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    @torch.no_grad()
    def generate_validation_images(self, prompts: list, num_images: int = 4):
        """Generate validation images."""
        # Create pipeline with LoRA weights
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config['model']['name'],
            unet=self.unet,
            text_encoder=self.text_encoder,
            vae=self.vae,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        pipeline.safety_checker = None  # Disable for fashion images
        
        images = []
        for prompt in prompts[:num_images]:
            image = pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            images.append(image)
        
        # Clear pipeline from memory
        del pipeline
        torch.cuda.empty_cache()
        
        return images
    
    def save_checkpoint(self, filename: str = None):
        """Save LoRA weights."""
        if filename is None:
            filename = f"checkpoint-{self.global_step}"
        
        save_path = self.output_dir / filename
        
        # Save only LoRA weights
        self.unet.save_pretrained(save_path)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path / 'training_state.pt')
        
        print(f"Saved checkpoint to {save_path}")
    
    def train(self, dataloader: DataLoader):
        """Full training loop."""
        config = self.config['training']
        
        num_epochs = config['num_train_epochs']
        max_train_steps = config.get('max_train_steps')
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        
        # Calculate total steps
        if max_train_steps is None:
            max_train_steps = num_epochs * len(dataloader) // gradient_accumulation_steps
        
        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            config['lr_scheduler'],
            optimizer=self.optimizer,
            num_warmup_steps=config['lr_warmup_steps'],
            num_training_steps=max_train_steps,
        )
        
        print("\n" + "="*50)
        print(" Starting LoRA Training")
        print("="*50)
        print(f"Total epochs: {num_epochs}")
        print(f"Total steps: {max_train_steps}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {config['batch_size'] * gradient_accumulation_steps}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"LoRA rank: {self.config['lora']['rank']}")
        print("="*50 + "\n")
        
        # Training loop
        progress_bar = tqdm(
            range(max_train_steps),
            desc="Training",
        )
        
        self.unet.train()
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                # Accumulate gradients
                loss = self.train_step(batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(),
                        config['max_grad_norm'],
                    )
                    
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % config.get('log_steps', 10) == 0:
                        progress_bar.set_postfix({
                            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        if self.writer:
                            self.writer.add_scalar(
                                'Loss/train',
                                loss.item() * gradient_accumulation_steps,
                                self.global_step
                            )
                            self.writer.add_scalar(
                                'LR',
                                lr_scheduler.get_last_lr()[0],
                                self.global_step
                            )
                    
                    # Save checkpoint
                    if self.global_step % config.get('save_steps', 500) == 0:
                        self.save_checkpoint()
                    
                    # Validation
                    validation_epochs = self.config['training'].get('validation_epochs', 10)
                    if epoch % validation_epochs == 0 and step == 0 and epoch > 0:
                        val_prompt = self.config['training']['validation_prompt']
                        images = self.generate_validation_images([val_prompt])
                        
                        # Save validation images
                        for i, img in enumerate(images):
                            img.save(self.output_dir / f'val_{self.global_step}_{i}.png')
                    
                    # Check if done
                    if self.global_step >= max_train_steps:
                        break
            
            if self.global_step >= max_train_steps:
                break
        
        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint('checkpoint-final')
        
        print("\n" + "="*50)
        print(" Training Complete!")
        print("="*50)
        print(f"Final checkpoint: {self.output_dir / 'checkpoint-final'}")
        print("\nTo generate images:")
        print(f"  python stable_diffusion_lora/generate.py --lora-path {self.output_dir / 'checkpoint-final'}")


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'lora_config.yaml'
    
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
    parser = argparse.ArgumentParser(description="Train Stable Diffusion LoRA")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
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
        print("CUDA not available, using CPU (this will be very slow!)")
        device = 'cpu'
    
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
    
    # Create trainer (loads models)
    trainer = LoRATrainer(config, device)
    
    # Create dataset
    dataset = FashionLoRADataset(
        data_dir=data_dir,
        tokenizer=trainer.tokenizer,
        resolution=config['data']['resolution'],
        center_crop=config['data'].get('center_crop', True),
        random_flip=config['data'].get('random_flip', True),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 workers for Windows compatibility
        pin_memory=True,
    )
    
    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
