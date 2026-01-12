"""
Stable Diffusion LoRA Fine-tuning
=================================

Fine-tune Stable Diffusion v1.5 with LoRA for fashion image generation.
Uses parameter-efficient training suitable for RTX 4060 (8GB VRAM).

Features:
- LoRA (Low-Rank Adaptation) training
- Memory-efficient with xformers and gradient checkpointing
- Text-conditioned fashion image generation
"""

__all__ = ['train_lora', 'generate']
