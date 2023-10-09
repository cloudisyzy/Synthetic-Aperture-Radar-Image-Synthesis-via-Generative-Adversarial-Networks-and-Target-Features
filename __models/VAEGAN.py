import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),  
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mean = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dims, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 8, 8))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.deconv(x)
        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoEncoder,self).__init__()
        self.latent_dims = latent_dims
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)
        
    def reparameterize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn(*mean.size()).cuda()
        z = mean + std*epsilon
        return z
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_tilda = self.decoder(z)
        return mean, logvar, x_tilda
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x_af_conv = x.clone()
        x = self.linear(x)
        return x, x_af_conv