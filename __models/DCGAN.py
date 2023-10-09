import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, dnu, 4, 2, 1),
            nn.BatchNorm2d(dnu),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu, dnu*2, 4, 2, 1),
            nn.BatchNorm2d(dnu*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*2, dnu*4, 4, 2, 1),
            nn.BatchNorm2d(dnu*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*4, dnu*8, 4, 2, 1),
            nn.BatchNorm2d(dnu*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, dnu*8, 4, 2, 1),
            nn.BatchNorm2d(dnu*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, 1, 4, 1, 0),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.model(x)
        return x

class Generator(nn.Module):
    def __init__(self, gnu=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, gnu*8, 4, 1, 0),
            nn.BatchNorm2d(gnu*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu*8, gnu*8, 4, 2, 1),
            nn.BatchNorm2d(gnu*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu*8, gnu*8, 4, 2, 1),
            nn.BatchNorm2d(gnu*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu*8, gnu*4, 4, 2, 1),
            nn.BatchNorm2d(gnu*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu*4, gnu*4, 4, 2, 1),
            nn.BatchNorm2d(gnu*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu*4, gnu, 4, 2, 1),
            # nn.BatchNorm2d(gnu),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu, 1, 3, 1, 1),
            nn.Tanh())
        
    def forward(self, x):
        return self.model(x)
