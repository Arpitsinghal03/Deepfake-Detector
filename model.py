import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        """
        Initialize the deepfake detection model.
        
        Args:
        
            pretrained: Whether to use pretrained weights
        """
        super(DeepfakeDetector, self).__init__()
        
        # Use ResNet50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify the final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Raw logits for binary classification
        """
        return self.backbone(x)

def create_model(pretrained=True, device=None):
    """
    Create and initialize a deepfake detection model.
    
    Args:
        pretrained: Whether to use pretrained weights
        device: Device to place the model on
        
    Returns:
        DeepfakeDetector: Initialized model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepfakeDetector(pretrained=pretrained)
    model = model.to(device)
    return model

def save_model(model, path):
    """
    Save a model to disk.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(path, device=None):
    """
    Load a model from disk.
    
    Args:
        path: Path to the saved model
        device: Device to place the model on
        
    Returns:
        DeepfakeDetector: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(pretrained=False, device=device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model 