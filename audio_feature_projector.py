import torch
import torch.nn as nn

class AudioFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(AudioFeatureProjector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)