import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class Attention(nn.Module):
    def __init__(self, c_in):
        super(Attention, self).__init__()
        self.quary = nn.Conv2d(c_in, c_in//2, 1, 1, 0)
        self.key = nn.Conv2d(c_in, c_in//2, 1, 1, 0)
        self.value = nn.Conv2d(c_in, c_in, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, W, H = x.size()
        quary = self.quary(x).view(B, -1, W*H).permute(0, 2, 1)
        key = self.key(x).view(B, -1, W*H)
        value = self.value(x).view(B, -1, W*H)
        attention = self.softmax(torch.bmm(quary, key))
        output = torch.bmm(value, attention.permute(0, 2, 1))
        output = output.view(B, -1, W, H)
        output = output * self.gamma + x
        return output, attention
    

class G_residual_block(nn.Module):
    def __init__(self, c_in, c_out, upsample=True):
        super(G_residual_block, self).__init__()
        if upsample == True:
            shortcut = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(c_in, c_out, 1, 1, 0)
            ]
            residual = [
                nn.BatchNorm2d(c_in),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
            ]
        elif upsample == False:
            shortcut = [
                nn.Conv2d(c_in, c_out, 1, 1, 0)
            ]
            residual = [
                nn.BatchNorm2d(c_in),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(c_in, c_out, 3, 1, 1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
            ]
            
        self.shortcut = nn.Sequential(*shortcut)
        self.residual = nn.Sequential(*residual)
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
    
class ResNet_Generator(nn.Module):
    def __init__(self, hp_gnu=256, final_layer_BN=True):
        super(ResNet_Generator, self).__init__()
        self.feature_mapping = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.resnet_flow = nn.Sequential(
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 64, upsample=True)
        )
        if final_layer_BN == True:
            self.fc = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 1, 3, 1, 1),
                nn.Tanh()
            )
        elif final_layer_BN == False:
            self.fc = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 1, 3, 1, 1),
                nn.Tanh()
            )
        
    def forward(self, x):
        x = self.feature_mapping(x)
        x = self.resnet_flow(x)
        x = self.fc(x)
        return x

class ResNet_Generator_Attention(nn.Module):
    def __init__(self, hp_gnu=256):
        super(ResNet_Generator_Attention, self).__init__()
        self.feature_mapping = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.resnet_flow = nn.Sequential(
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 256, upsample=True),
            G_residual_block(256, 64, upsample=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention_1 = Attention(64)
        self.fc_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention_2 = Attention(64)
        self.fc_2 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.feature_mapping(x)
        x = self.resnet_flow(x)
        x, _ = self.attention_1(x)
        x = self.fc_1(x)
        x, _ = self.attention_2(x)
        x = self.fc_2(x)
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, dnu=64, loss='vanilla'):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, dnu, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu, 3, 1, 1))
            )
        self.flatten = nn.Flatten()
        self.fc = spectral_norm(nn.Linear(dnu*4, 1))
        self.loss = loss
        
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.loss == 'vanilla':
            x = torch.sigmoid(x)
        return x
    
    
class Discriminator_Attention(nn.Module):
    def __init__(self, dnu=64):
        super(Discriminator_Attention, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, dnu, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu*4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*4, dnu*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dnu*8, dnu*8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flow_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(dnu*8, dnu*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attention_1 = Attention(dnu*4)
        self.flow_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(dnu*4, dnu, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attention_2 = Attention(dnu)
        self.conv = nn.Conv2d(dnu, 1, 4, 1, 0)
        
    def forward(self, x):
        x = self.model(x)
        x, _ = self.attention_1(self.flow_1(x))
        x, _ = self.attention_2(self.flow_2(x))
        x = self.conv(x)
        return x
    
    
class D_residual_block_1(nn.Module):
    def __init__(self, c_in, c_out):
        super(D_residual_block_1, self).__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(c_in, c_out, 1, 1, 0))
        )
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv2d(c_in, c_out, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1))
        )
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
class D_residual_block_2(nn.Module):
    def __init__(self, c_in, c_out, downsample=False):
        super(D_residual_block_2, self).__init__()
        shortcut = []  
        if c_in != c_out and downsample == False:
            shortcut.append(
                spectral_norm(nn.Conv2d(c_in, c_out, 1, 1, 0))
            )
        if downsample == True:
            shortcut.append(
                spectral_norm(nn.Conv2d(c_in, c_out, 1, 1, 0))
            )
            shortcut.append(
                nn.AvgPool2d(2)
            )

        if downsample == True:
            residual = [
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(c_in, c_out, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1))
            ]
        else:
            residual = [
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(c_in, c_out, 3, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1))
            ]
            
        self.shortcut = nn.Sequential(*shortcut)
        self.residual = nn.Sequential(*residual)
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
class ResNet_Discriminator(nn.Module):
    def __init__(self, loss='vanilla'):
        super(ResNet_Discriminator, self).__init__()
        self.model = nn.Sequential(
            D_residual_block_1(1, 128),
            D_residual_block_2(128, 512, downsample=True),
            D_residual_block_2(512, 512, downsample=False),
            D_residual_block_2(512, 512, downsample=True),
            D_residual_block_2(512, 128, downsample=False),
            D_residual_block_2(128, 8, downsample=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        fc = [spectral_norm(nn.Linear(512, 1))]
        if loss == 'vanilla':
            fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc)        
      
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    
def D_hinge_loss(real_output, fake_output):
    real_loss = torch.mean(torch.relu(1.0 - real_output))
    fake_loss = torch.mean(torch.relu(1.0 + fake_output))
    d_loss = real_loss + fake_loss
    return d_loss


def G_hinge_loss(fake_output):
    g_loss = -torch.mean(fake_output)
    return g_loss