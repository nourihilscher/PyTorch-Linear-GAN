import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

##### Discriminator #####
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        input_features = 1 * image_size * image_size
    
        # Define hidden linear layers
        self.hcl1 = nn.Linear(input_features, 1024)
        self.hcl2 = nn.Linear(1024, 512)
        self.hcl3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
  
    def forward(self, x):
        x = self.leaky_relu(self.hcl1(x))
        x = self.leaky_relu(self.hcl2(x))
        x = self.leaky_relu(self.hcl3(x))
        x = torch.sigmoid(self.out(x))
        return x

##### Generator #####
def linear_block(in_features, out_features, batch_norm=True):
    layers = []
    if batch_norm:
        linear_layer = nn.Linear(in_features, out_features, bias=False)
        batch_norm = nn.BatchNorm1d(out_features)
        layers = [linear_layer, batch_norm]
    else:
        layers.append(nn.Linear(in_features, out_features))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, input_features, image_size):
        super().__init__()
        output_features = 1 * image_size * image_size
    
        # Define hidden linear layers
        self.hcl1 = linear_block(input_features, 256)
        self.hcl2 = linear_block(256, 512)
        self.hcl3 = linear_block(512, 1024)
        self.output = linear_block(1024, output_features)
    
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.hcl1(x))
        x = self.leaky_relu(self.hcl2(x))
        x = self.leaky_relu(self.hcl3(x))
        x = torch.tanh(self.output(x))
        return x