"""
Projected GAN for Fashion Image Generation
==========================================

A lightweight GAN implementation using feature projection
with pretrained networks for fast, stable training.

Optimized for RTX 4060 (8GB VRAM)
"""

from .model import Generator, ProjectedDiscriminator
from .train import Trainer

__all__ = ['Generator', 'ProjectedDiscriminator', 'Trainer']
