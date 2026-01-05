import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F

device = torch.device("cpu")

import torchvision
import torchinfo
import torchsummary

from data import Dataset

# encoder architecture
class Encoder(nn.Module):
    def __init__(self, latent_dim, normalize: bool = False):
        r'''
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        '''
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1) # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 32*2, 4, stride=2, padding=1) # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(32*2, 32*4, 4, stride=2, padding=1) # 7x7 --> 3x3
        self.conv4 = nn.Conv2d(32*4, 32*8, 4, stride=2, padding=1) # 3x3 --> 1x1
        self.conv5 = nn.Conv2d(32*8, latent_dim, 1) # 1x1 --> 1x1
        self.fc = nn.Linear(latent_dim, latent_dim)

        self.nonlinearity = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x).flatten(1))
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x)
        return x

    def extra_repr(self):
        return f'normalize={self.normalize}'


# decoder architecture
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        r'''
        latent_dim (int): Dimension of latent space
        '''
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent_dim, 32*8, 4, stride=2, padding=1, output_padding=1) # 1x1 --> 3x3
        self.conv2 = nn.ConvTranspose2d(32*8, 32*4, 4, stride=2, padding=1, output_padding=1) # 3x3 --> 7x7
        self.conv3 = nn.ConvTranspose2d(32*4, 32*2, 4, stride=2, padding=1) # 7x7 --> 14x14
        self.conv4 = nn.ConvTranspose2d(32*2, 32, 4, stride=2, padding=1) # 14x14 --> 28x28
        self.conv5 = nn.ConvTranspose2d(32, 1, 1) # 28x28 --> 28x28 (no change, just channel reduction)
        self.nonlinearity = nn.ReLU()

    def forward(self, z):
        z = z[..., None, None]  # make it convolution-friendly

        x = self.nonlinearity(self.conv1(z))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        return torch.sigmoid(self.conv5(x))  # sigmoid to ensure output in [0, 1] for grayscale

