import torch
import torch.nn as nn
import torchvision.models as models
from .adapter import ChannelAdapter

class BinaryDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, input_channels=5):
        super(BinaryDeepfakeDetector, self).__init__()
        
        # Adaptador de canales
        self.adapter = ChannelAdapter(in_channels=input_channels, out_channels=3)
        
        # Backbone pre-entrenado
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Reemplazar capa fully connected
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone(x)
        return x
    
    def get_parameter_count(self):
        """Retorna el número total de parámetros"""
        return sum(p.numel() for p in self.parameters())