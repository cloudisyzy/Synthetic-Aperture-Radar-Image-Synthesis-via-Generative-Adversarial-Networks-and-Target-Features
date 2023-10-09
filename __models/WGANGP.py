import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, dnu, 4, 2, 1),
            nn.InstanceNorm2d(dnu),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu, dnu*2, 4, 2, 1),
            nn.InstanceNorm2d(dnu*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*2, dnu*4, 4, 2, 1),
            nn.InstanceNorm2d(dnu*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*4, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d(dnu*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d(dnu*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, 1, 4, 1, 0)
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
            nn.BatchNorm2d(gnu),
            nn.ReLU(True),
            nn.ConvTranspose2d(gnu, 1, 3, 1, 1),
            nn.Tanh())
        
    def forward(self, x):
        return self.model(x)
    
def gradient_penalty(netD, real_img, fake_img, device):
    
    batch_size = real_img.shape[0]
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolated_img = real_img * epsilon + fake_img * (1-epsilon)
    mixed_scores = netD(interpolated_img)
    gradient = torch.autograd.grad(
                inputs = interpolated_img,
                outputs = mixed_scores,
                grad_outputs = torch.ones_like(mixed_scores),
                create_graph = True, retain_graph = True
                )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_panelty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_panelty