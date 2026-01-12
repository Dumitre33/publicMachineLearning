"""
Projected GAN Model Architecture (Bulletproof Version)
=======================================================

Generator: Lightweight synthesis network with spectral normalization
Discriminator: Uses frozen pretrained EfficientNet features

Based on: "Projected GANs Converge Faster" (Sauer et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import timm
from typing import List, Tuple
import numpy as np


# ============================================
# Generator Architecture
# ============================================

class MappingNetwork(nn.Module):
    """Maps latent z to intermediate w space."""
    
    def __init__(
        self,
        z_dim: int = 256,
        w_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
        self.mapping = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization."""
    
    def __init__(self, w_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-8)
        self.style = nn.Linear(w_dim, num_features * 2)
        
        # Initialize to identity transform
        nn.init.ones_(self.style.weight[:num_features])
        nn.init.zeros_(self.style.weight[num_features:])
        self.style.bias.data[:num_features] = 1.0
        self.style.bias.data[num_features:] = 0.0
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        style = self.style(w)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        x = self.norm(x)
        return gamma * x + beta


class SynthesisBlock(nn.Module):
    """Synthesis block with style modulation and spectral normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        upsample: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.upsample = upsample
        
        # Apply spectral normalization for stability
        if use_spectral_norm:
            self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.adain1 = AdaIN(w_dim, out_channels)
        self.adain2 = AdaIN(w_dim, out_channels)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Learnable noise scale - initialize to small value
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        w: torch.Tensor,
        noise: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv1(x)
        
        # Add noise with learnable scale
        if noise is not None:
            x = x + self.noise_scale1 * noise[0]
        else:
            x = x + self.noise_scale1 * torch.randn_like(x) * 0.1
        
        x = self.adain1(x, w)
        x = self.activation(x)
        
        x = self.conv2(x)
        
        # Add noise
        if noise is not None:
            x = x + self.noise_scale2 * noise[1]
        else:
            x = x + self.noise_scale2 * torch.randn_like(x) * 0.1
        
        x = self.adain2(x, w)
        x = self.activation(x)
        
        return x


class Generator(nn.Module):
    """
    Lightweight generator for fashion image synthesis.
    
    Architecture: 
    - Mapping network: z -> w
    - Synthesis network: constant -> image (with style modulation)
    - Spectral normalization for training stability
    """
    
    def __init__(
        self,
        z_dim: int = 256,
        w_dim: int = 256,
        img_size: int = 256,
        img_channels: int = 3,
        base_channels: int = 32,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_size = img_size
        
        # Calculate number of layers needed
        self.num_layers = int(np.log2(img_size)) - 2  # Start from 4x4
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Learnable constant input (4x4) - initialize with small values
        self.const = nn.Parameter(torch.randn(1, base_channels * 16, 4, 4) * 0.01)
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        
        in_ch = base_channels * 16
        for i in range(self.num_layers):
            out_ch = max(base_channels, in_ch // 2)
            self.blocks.append(
                SynthesisBlock(in_ch, out_ch, w_dim, upsample=True, use_spectral_norm=use_spectral_norm)
            )
            in_ch = out_ch
        
        # Final convolution to RGB with spectral norm
        if use_spectral_norm:
            self.to_rgb = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, img_channels, 1)),
                nn.Tanh(),
            )
        else:
            self.to_rgb = nn.Sequential(
                nn.Conv2d(in_ch, img_channels, 1),
                nn.Tanh(),
            )
    
    def forward(
        self,
        z: torch.Tensor,
        return_latents: bool = False,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Map z to w
        w = self.mapping(z)
        
        # Start from constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Synthesis blocks
        for block in self.blocks:
            x = block(x, w)
        
        # To RGB
        img = self.to_rgb(x)
        
        # Clamp output to valid range for extra safety
        img = torch.clamp(img, -1.0, 1.0)
        
        if return_latents:
            return img, w
        return img
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        device: str = 'cuda',
        truncation: float = 1.0,
    ) -> torch.Tensor:
        """Generate random samples."""
        z = torch.randn(num_samples, self.z_dim, device=device)
        
        if truncation < 1.0:
            # Truncation trick for better quality
            z = z * truncation
        
        return self(z)


# ============================================
# Projected Discriminator
# ============================================

class FeatureProjector(nn.Module):
    """Projects features from pretrained network with spectral norm."""
    
    def __init__(self, in_channels: int, out_channels: int = 128, use_spectral_norm: bool = True):
        super().__init__()
        
        if use_spectral_norm:
            self.proj = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DiscriminatorHead(nn.Module):
    """Discriminator head for each feature scale with spectral norm."""
    
    def __init__(self, in_channels: int, use_spectral_norm: bool = True):
        super().__init__()
        
        if use_spectral_norm:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.head = spectral_norm(nn.Conv2d(in_channels, 1, 1))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.head = nn.Conv2d(in_channels, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)


class ProjectedDiscriminator(nn.Module):
    """
    Discriminator using frozen pretrained features.
    
    Uses EfficientNet features at multiple scales for
    multi-scale discrimination. Includes spectral normalization
    for training stability.
    """
    
    def __init__(
        self,
        backbone: str = 'tf_efficientnet_lite0',
        proj_channels: int = 128,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        # Load pretrained backbone (frozen)
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],  # Multi-scale features
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            features = self.backbone(dummy)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Feature projectors and discriminator heads with spectral norm
        self.projectors = nn.ModuleList()
        self.heads = nn.ModuleList()
        
        for dim in self.feature_dims:
            self.projectors.append(FeatureProjector(dim, proj_channels, use_spectral_norm))
            self.heads.append(DiscriminatorHead(proj_channels, use_spectral_norm))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of discriminator outputs at different scales.
        
        Gradients flow through the backbone (though backbone weights are frozen).
        This is needed for both generator training and R1 regularization.
        """
        # Normalize input to ImageNet stats
        x = self._normalize(x)
        
        # Extract features - backbone is frozen but gradients flow through
        features = self.backbone(x)
        
        # Project features and get discriminator outputs
        outputs = []
        for feat, proj, head in zip(features, self.projectors, self.heads):
            projected = proj(feat)
            out = head(projected)
            outputs.append(out)
        
        return outputs
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet stats."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Then normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        return (x - mean) / std


# ============================================
# Losses
# ============================================

def hinge_loss_dis(real_outputs: List[torch.Tensor], fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean(F.relu(1 - real))
        loss += torch.mean(F.relu(1 + fake))
    return loss / len(real_outputs)


def hinge_loss_gen(fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for generator."""
    loss = 0
    for fake in fake_outputs:
        loss -= torch.mean(fake)
    return loss / len(fake_outputs)


def r1_penalty(
    real_images: torch.Tensor,
    real_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """R1 gradient penalty for regularization."""
    gradients = torch.autograd.grad(
        outputs=[o.sum() for o in real_outputs],
        inputs=real_images,
        create_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    
    # Handle case where gradients is None (can happen with AMP)
    if gradients is None:
        return torch.tensor(0.0, device=real_images.device)
    
    penalty = gradients.pow(2).reshape(gradients.size(0), -1).sum(1).mean()
    return penalty


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")
    
    # Create models with spectral norm
    G = Generator(z_dim=256, img_size=256, use_spectral_norm=True).to(device)
    D = ProjectedDiscriminator(use_spectral_norm=True).to(device)
    
    # Test forward pass
    z = torch.randn(4, 256, device=device)
    fake_images = G(z)
    print(f"Generated images shape: {fake_images.shape}")
    print(f"Image value range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    d_outputs = D(fake_images)
    print(f"Discriminator outputs: {[o.shape for o in d_outputs]}")
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator trainable parameters: {d_params:,}")
    print("Model test passed!")
