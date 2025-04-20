import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SkinToneModel(nn.Module):
    def __init__(self, num_classes=10):  # 10 skin tone classes
        super(SkinToneModel, self).__init__()
        # Load EfficientNet-B0
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze base layers (as done in training)
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Get input features for classifier
        in_features = self.base.classifier[1].in_features
        
        # Define custom classifier head
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base(x) 
