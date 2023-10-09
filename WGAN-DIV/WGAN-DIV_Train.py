#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("..")


# In[ ]:


# Import necessary public libraries as well as classes and functions written by the author
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
import PIL
from PIL import Image
import random
import os
from __utils.functions import calc_MSE, calc_SSIM, find_most_similar_img
from __utils.functions import Imagefolder
from __utils.functions import show_img, save_tensor_as_img, img_to_grid_save
from __utils.functions import weight_init, reset_grad
from __utils.functions import plot_line_graph
from __utils.functions import find_key
from __models.WGANDIV import Discriminator, Generator, div_gradient_penalty
sys.path.remove("..")


# In[ ]:


# Use GPU rather than CPU to accelerate the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Hyperparameters
hp_lr = 1e-4
hp_alpha = 0.9
hp_batch_size = 32
hp_epochs = 400
hp_gin = 256


# In[ ]:


# Move GAN to GPU
netD = Discriminator().to(device)
netG = Generator().to(device)


# In[ ]:


# You can see the detail of GAN by runing this block
print(summary(netD, (1,128,128)))
print(summary(netG, (256,1,1)))


# In[ ]:


# Load trained parameters (For fune-tuning only)
# netD.load_state_dict(torch.load(''))
# netG.load_state_dict(torch.load(''))


# In[ ]:


# Store the images (3 in total) after each iteration, "fixed_noise" is the latent vector of the three images
fixed_noise = torch.randn(36, hp_gin, 1, 1).to(device)
# Store the loss after each iteration
G_losses = []
D_losses = []
# Store the SSIM after each iteration
SSIM_history = []
# This dict stores the classes of SAR images and their corresponding values
class_dict = {'2S1': 0,
             'BMP2': 1,
             'BRDM2': 2,
             'BTR60': 3,
             'BTR70': 4,
             'D7': 5,
             'T62': 6,
             'T72': 7,
             'ZIL131': 8,
             'ZSU234': 9}


# In[ ]:


for classes in range(10):
    # Clear output after each loop
    clear_output(wait=True)
    
    # Define the optimizer of GAN
    netD = Discriminator().to(device)
    netG = Generator().to(device)
    solver_G = optim.RMSprop(netG.parameters(), lr=hp_lr, alpha=hp_alpha)
    solver_D = optim.RMSprop(netD.parameters(), lr=hp_lr, alpha=hp_alpha)

    # Initilize the weight of GAN
    netD = netD.apply(weight_init)
    netG = netG.apply(weight_init)
    
    category_name = find_key(class_dict, classes)
    
    # Load the training data
    dataset1, _ = Imagefolder(root='../_MSTAR/TRAIN/', normalize=True, category_idx=classes)
    dataloader1 = DataLoader(dataset1, batch_size=hp_batch_size, shuffle=True, drop_last=True)
    # validation data
    dataset2, _ = Imagefolder(root='../_MSTAR/TEST/', normalize=False, category_idx=classes) 
    dataloader2 = DataLoader(dataset2, batch_size=36, shuffle=True)

    print("——————————Now start training————————")
                            
#             Train Generator and Discriminator Equally

    for epoch in range(hp_epochs):
        for i, (img, _) in enumerate(dataloader1, 0):
    #############   ===================   #############
    #############   Train Discriminator   #############
    #############   ===================   #############
    #         Train with real data
            netD.train()
            reset_grad(solver_D, solver_G)
            real_img = img.to(device)
            dis_real = netD(real_img).view(-1)
    #         Train with fake data
            noise = torch.randn(real_img.shape[0], hp_gin, 1, 1, device=device)
            fake_img = netG(noise)
            dis_fake = netD(fake_img).view(-1)
            d_loss_real = torch.mean(dis_real)
            d_loss_fake = torch.mean(dis_fake)
            GP = div_gradient_penalty(netD, real_img, fake_img.detach(), batch_size=hp_batch_size, device=device, k=1, p=3)
            loss_D = -(d_loss_real - d_loss_fake) + GP
            loss_D.backward()
            solver_D.step()

    #############   ===============   #############
    #############   Train Generator   #############
    #############   ===============   #############
            netG.train()
            reset_grad(solver_D, solver_G)
            noise = torch.randn(hp_batch_size, hp_gin, 1, 1, device=device)
            fake_img = netG(noise)
            labels = torch.ones(hp_batch_size, 1).to(device)
            output = netD(fake_img).view(-1)
            loss_G = -torch.mean(output)
            loss_G.backward()
            solver_G.step()

    #         Print loss during training, easy to track the performance
            if i % 100 == 0:
                print('{%d/10}|"%s"|[%d/%d](%d/%d)\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (classes+1, category_name, epoch+1, hp_epochs, i, len(dataloader1),
                         loss_D.mean().item(), loss_G.mean().item()))

    #         Store the loss in two lists
            if i == 5:
                D_losses.append(loss_D.mean().item())
                G_losses.append(loss_G.mean().item())

    #     Show the generated images after 10 epoch, and calculate the SSIM
        if (epoch+1) % 10 == 0:
            netG.eval()
            with torch.no_grad():
                generated_samples = netG(fixed_noise).cpu()
                img_to_grid_save(generated_samples, root='history/%s/' %category_name, name='Epoch_%d' %(epoch+1), name_num=False, nrow=6)
                show_img(generated_samples, normalize=True, dpi=150)
                generated_samples = generated_samples * 0.5 + 0.5
                ssim = calc_SSIM(generated_samples, dataloader2, mode='bd', epoch=20)
                SSIM_history.append(ssim)
                print('-----------------')
                print('SSIM: %.4f' %ssim)
                print('-----------------')

    # Save the parameters of the GAN
    torch.save(netG.state_dict(), 'Generator_%s.pkl' %category_name)
    torch.save(netD.state_dict(), 'Discriminator_%s.pkl' %category_name)
    
    with open("WGAN-DIV_SSIM_list.txt", "a") as file:
        file.write('\n')
        file.write('%s\n' %category_name)
        for item in SSIM_history:
            file.write(str(item) + '\t')

#     # Save the generated samples：200*5=1000 images
    netG.eval()
    for img_loop in range(5):
        noise = torch.randn(200, hp_gin, 1, 1).to(device)
        gen_img = netG(noise)
        save_tensor_as_img(gen_img, root='images/%s' %category_name, normalize=True, name='batch_%d' %img_loop)


# In[ ]:


# Plot the Loss Progess of Discriminator and Generator
plot_line_graph(line1=D_losses, name1='Dis', line2=G_losses, name2='Gen', dpi=100, title='Loss progress history', xlabel='epoch', ylabel='')


# In[ ]:


# Plot the SSIM/MSE progress history
plot_line_graph(line1=SSIM_history, title='SSIM progress history', xlabel='step', ylabel='', dpi=100)

