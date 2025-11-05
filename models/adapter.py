import torch.nn as nn

class ChannelAdapter(nn.Module):
    """Adaptador de canales para convertir 5 canales a 3"""
    def __init__(self, in_channels=5, out_channels=3):
        super(ChannelAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x