import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AudioDeepfakeDetector(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,  # Mono audio
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # CNN layers for spectrogram processing
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               where height and width represent the spectrogram dimensions
        Returns:
            Tuple of (prediction, attention_maps)
        """
        # Process through CNN layers
        features = self.conv_layers(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        output = self.fc_layers(attended_features)
        
        return output, attention_weights if self.training else None
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            _, attention_maps = self.forward(x)
        self.train()
        return attention_maps

class AudioFeatureExtractor(nn.Module):
    """Helper class to extract features from raw audio for the detector"""
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio to mel spectrogram
        
        Args:
            audio: Raw audio tensor of shape (batch_size, samples)
        Returns:
            Mel spectrogram tensor of shape (batch_size, 1, n_mels, time)
        """
        # Compute spectrogram
        spectrogram = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=True
        )
        
        # Convert to power spectrogram
        power_spec = torch.abs(spectrogram) ** 2
        
        # Convert to mel scale
        mel_basis = torch.functional.mel_filters(
            self.sample_rate,
            self.n_fft,
            self.n_mels
        ).to(audio.device)
        
        mel_spec = torch.matmul(mel_basis, power_spec)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(1)
        
        return mel_spec

def create_audio_detector(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[AudioDeepfakeDetector, AudioFeatureExtractor]:
    """
    Factory function to create and initialize the audio detector model and feature extractor
    
    Args:
        device: Device to place the models on
    
    Returns:
        Tuple of (AudioDeepfakeDetector, AudioFeatureExtractor)
    """
    detector = AudioDeepfakeDetector().to(device)
    feature_extractor = AudioFeatureExtractor().to(device)
    return detector, feature_extractor 