"""
Common utility functions
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        print(f"Total VRAM: {total_memory:.1f} GB")
        
        if torch.cuda.memory_allocated() > 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffix."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
