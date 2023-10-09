import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, dnu=64, loss='vanilla'):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, dnu, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu, dnu*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*2, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu, 4, 2, 1))
            )
        self.flatten = nn.Flatten()
        self.fc = spectral_norm(nn.Linear(dnu*4, 1))
        self.sigmoid = nn.Sigmoid()
        self.loss = loss
        
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.loss == 'vanilla':
            x = self.sigmoid(x)
        return x
    
class Discriminator_Hinge(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator_Hinge, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, dnu, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu, dnu*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*2, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu, 4, 2, 1))
            )
        self.flatten = nn.Flatten()
        self.fc = spectral_norm(nn.Linear(dnu*4, 1))
        
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    

class Generator(nn.Module):
    def __init__(self, gnu=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(gnu, gnu*8, 4, 1, 0),
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
            nn.BatchNorm2d(gnu),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu, 1, 3, 1, 1),
            nn.Tanh())
        
    def forward(self, x):
        return self.model(x)

class Enlarged_Generator(nn.Module):
    def __init__(self, gnu=64):
        super(Enlarged_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(gnu*4, gnu*8, 4, 1, 0),
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
    
    
def D_hinge_loss(real_output, fake_output):
    real_loss = torch.mean(torch.relu(1.0 - real_output))
    fake_loss = torch.mean(torch.relu(1.0 + fake_output))
    d_loss = real_loss + fake_loss
    return d_loss


def G_hinge_loss(fake_output):
    g_loss = -torch.mean(fake_output)
    return g_loss