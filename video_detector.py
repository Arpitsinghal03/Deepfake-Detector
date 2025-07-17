import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional

class AttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights

class VideoDeepfakeDetector(nn.Module):
    def __init__(
        self,
        num_frames: int = 16,
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify the first layer to accept temporal dimension
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=num_frames * 3,  # RGB frames
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Add attention modules
        self.attention_modules = nn.ModuleList([
            AttentionModule(32),   # After first block
            AttentionModule(56),   # After second block
            AttentionModule(80),   # After third block
            AttentionModule(192),  # After fourth block
        ])
        
        # Modify classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            Tuple of (prediction, attention_maps)
        """
        batch_size = x.size(0)
        
        # Reshape input for temporal processing
        x = x.view(batch_size, -1, x.size(3), x.size(4))
        
        # Process through backbone with attention
        attention_maps = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in [2, 4, 6, 8]:  # After each major block
                attn_idx = len(attention_maps)
                if attn_idx < len(self.attention_modules):
                    x = self.attention_modules[attn_idx](x)
                    attention_maps.append(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.backbone.classifier(x)
        
        return x, attention_maps if self.training else None

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            _, attention_maps = self.forward(x)
        self.train()
        return attention_maps

def create_video_detector(
    num_frames: int = 16,
    pretrained: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> VideoDeepfakeDetector:
    """
    Factory function to create and initialize the video detector model
    
    Args:
        num_frames: Number of frames to process at once
        pretrained: Whether to use pretrained weights
        device: Device to place the model on
    
    Returns:
        Initialized VideoDeepfakeDetector model
    """
    model = VideoDeepfakeDetector(
        num_frames=num_frames,
        pretrained=pretrained
    )
    return model.to(device) 