import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, dnu, 4, 2, 1),
            nn.InstanceNorm2d([64, 64]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu, dnu*2, 4, 2, 1),
            nn.InstanceNorm2d([32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*2, dnu*4, 4, 2, 1),
            nn.InstanceNorm2d([16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*4, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d([8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d([4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, 1, 4, 1, 0)
            )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class Discriminator_1(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator_1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, dnu, 4, 2, 1),
            nn.InstanceNorm2d(dnu, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu, dnu*2, 4, 2, 1),
            nn.InstanceNorm2d(dnu*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*2, dnu*4, 4, 2, 1),
            nn.InstanceNorm2d(dnu*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*4, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d(dnu*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dnu*8, dnu*8, 4, 2, 1),
            nn.InstanceNorm2d(dnu*8, affine=True),
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
    
def div_gradient_penalty(Discriminator, real_img, fake_img, batch_size, device, k=2, p=6):
    mu = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated_img = (mu*real_img + (1-mu)*fake_img).requires_grad_(True)
    interpolate_output = Discriminator(interpolated_img)
    initial_gradient = torch.ones_like(interpolate_output)
    
    interpolate_gradient = torch.autograd.grad(inputs=interpolated_img, outputs=interpolate_output, 
                                               grad_outputs=initial_gradient, create_graph=True, retain_graph=True, 
                                               only_inputs=True)[0]
    
    interpolate_gradient = interpolate_gradient.view(batch_size, -1)
    gradient_penalty = torch.pow(interpolate_gradient.norm(2, dim=1), p).mean() * k
    return gradient_penalty