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
from __models.VAEGAN import Discriminator, VariationalAutoEncoder
sys.path.remove("..")


# In[ ]:


# Use GPU rather than CPU to accelerate the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Hyperparameters
hp_lr = 3e-4
hp_batch_size = 32
hp_epochs = 20
hp_latent_dims = 32
hp_gamma_d = 15 
hp_gamma_e = 8 


# In[ ]:


# Move GAN to GPU
vae = VariationalAutoEncoder(latent_dims=hp_latent_dims).to(device)
netD = Discriminator().to(device)


# In[ ]:


# You can see the detail of GAN by runing this block
print(summary(netD, (1,128,128)))
print(summary(vae, (1,128,128)))


# In[ ]:


# Load trained parameters (For fune-tuning only)
# netD.load_state_dict(torch.load(''))
# vae.load_state_dict(torch.load(''))


# In[ ]:


# Store the loss after each iteration
G_losses = []
D_losses = []
E_losses = []
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


# Binary-CrossEntropy Loss is to calculate the loss function of a standard GAN
loss = nn.BCELoss()
# Specify the value of labels to help netD
real_label = 1.0
fake_label = 0.0


# In[ ]:


for classes in range(10):
    # Clear output after each loop
    clear_output(wait=True)
    
    # Define the optimizer of GAN
    vae = VariationalAutoEncoder(latent_dims=hp_latent_dims).to(device)
    netD = Discriminator().to(device)
    solver_E = optim.Adam(vae.encoder.parameters(), lr=hp_lr)
    solver_D = optim.Adam(vae.decoder.parameters(), lr=hp_lr)
    solver_Dis = optim.Adam(netD.parameters(), lr=hp_lr/3)

    # Initilize the weight of GAN
    vae = vae.apply(weight_init)
    netD = netD.apply(weight_init)
    
    category_name = find_key(class_dict, classes)
    
    # Load the training data
    dataset1, _ = Imagefolder(root='../_MSTAR/TRAIN/', normalize=True, category_idx=classes)
    dataloader_1 = DataLoader(dataset1, batch_size=hp_batch_size, shuffle=True, drop_last=True)
    # validation data
    dataset2, _ = Imagefolder(root='../_MSTAR/TEST/', normalize=True, category_idx=classes) 
    dataloader_2 = DataLoader(dataset2, batch_size=16, shuffle=True)

    print("——————————Now start training————————")

    for epoch in range(hp_epochs):
        for i, (img, _) in enumerate(dataloader_1):
            vae.train()
            netD.train()

            real_label = torch.ones(hp_batch_size, 1).cuda()
            fake_label = torch.zeros(hp_batch_size, 1).cuda()
            img = img.cuda()
            mean, logvar, recon_img = vae(img)
            noise = torch.randn(hp_batch_size, hp_latent_dims).cuda()
            sampled_img = vae.decoder(noise)

            #############   ===================   #############
            #############   Train Discriminator   #############
            #############   ===================   #############
            solver_Dis.zero_grad()

            dis_origin_result, dis_origin_latent = netD(img)
            dis_recon_result, dis_recon_latent = netD(recon_img)
            dis_sampled_result, dis_sampled_latent = netD(sampled_img)

            err_dis_origin = loss(dis_origin_result, real_label)
            err_dis_recon = loss(dis_recon_result, fake_label)
            err_dis_sampled = loss(dis_sampled_result, fake_label)
            dis_loss = err_dis_origin + err_dis_recon + err_dis_sampled

            dis_loss.backward(retain_graph=True)
            solver_Dis.step()

            #############   =============   #############
            #############   Train Decoder   #############
            #############   =============   #############
            # with torch.autograd.set_detect_anomaly(True):
            solver_D.zero_grad()

            dis_origin_result, dis_origin_latent = netD(img)
            dis_recon_result, dis_recon_latent = netD(recon_img)
            dis_sampled_result, dis_sampled_latent = netD(sampled_img)

            err_dis_origin = loss(dis_origin_result, real_label)
            err_dis_recon = loss(dis_recon_result, fake_label)
            err_dis_sampled = loss(dis_sampled_result, fake_label)

            d_loss_1 = err_dis_recon + 1*err_dis_sampled
            d_loss_2 = torch.mean((dis_origin_latent - dis_recon_latent)**2)
            d_loss = hp_gamma_d*d_loss_2 - d_loss_1

            d_loss.backward()
            solver_D.step()

            #############   =============   #############
            #############   Train Encoder   #############
            #############   =============   #############
            solver_E.zero_grad()

            mean, logvar, recon_img = vae(img)
            _, dis_origin_latent = netD(img)
            _, dis_recon_latent = netD(recon_img)
            e_loss_1 = torch.mean((dis_origin_latent - dis_recon_latent)**2)
            e_loss_2 = (-0.5 * torch.sum(-logvar.exp() - torch.pow(mean,2) + logvar + 1)) / torch.numel(mean.data)
            e_loss = hp_gamma_e*e_loss_1 + e_loss_2

            e_loss.backward()
            solver_E.step()

    #         Print loss during training, easy to track the performance
            if i % 100 == 0:
                print('{%d/10}|"%s"|[%d/%d](%d/%d)\tLoss_Dis: %.4f\tLoss_E: %.4f\tLoss_D: %.4f'
                      % (classes+1, category_name, epoch+1, hp_epochs, i, len(dataloader_1),
                         dis_loss.mean().item(), e_loss.mean().item(), d_loss.mean().item()))
                
    #         Store the loss in two lists
            if i == 5:
                D_losses.append(dis_loss.mean().item())
                G_losses.append(d_loss.mean().item())
                E_losses.append(e_loss.mean().item())

    #     Show the generated images after 10 epoch
        if (epoch+1) % 10 == 0:
            vae.eval()
            with torch.no_grad():
                noise = torch.randn(16, hp_latent_dims).cuda()
                generated_samples = vae.decoder(noise).cpu()
                show_img(generated_samples[0:9], normalize=True, dpi=100, title='Generated Samples')
                img_test, _ = next(iter(dataloader_2))
                img_test = img_test.cuda()
                _, _, recon_img_test = vae(img_test)
            show_img(img[0:16], normalize=True, dpi=100, title='Original Train Images')
            show_img(recon_img[0:16], normalize=True, dpi=100, title='Reconstructed Train Images')
            show_img(img_test, normalize=True, dpi=100, title='Original Test Images')
            show_img(recon_img_test, normalize=True, dpi=100, title='Reconstructed Test Images')
            grid_1 = make_grid(img[0:16], cmap='gray', normalize=True, nrow=4)
            grid_2 = make_grid(recon_img[0:16], cmap='gray', normalize=True, nrow=4)
            grid_3 = make_grid(img_test, cmap='gray', normalize=True, nrow=4)
            grid_4 = make_grid(recon_img_test, cmap='gray', normalize=True, nrow=4)
            big_grid = make_grid([grid_1, grid_2, grid_3, grid_4], cmap='gray', normalize=False, nrow=2, pad_value=32)
            root = ('history/%s/' %category_name)
            if not os.path.exists(root):
                os.makedirs(root)
            save_image(big_grid, root+'Epoch_compare_%d.png' %(epoch+1))
            img_to_grid_save(generated_samples[0:9], root='history/%s/' %category_name, name='Epoch_sampling_%d' %(epoch+1), name_num=False)
            
    # Save the parameters of the GAN
    torch.save(vae.state_dict(), 'VariationalAutoEncoder_%s.pkl' %category_name)
    torch.save(netD.state_dict(), 'Discriminator_%s.pkl' %category_name)

#     Save Reconstructed Images
    dataloader_1 = DataLoader(dataset1, batch_size=10, shuffle=False, drop_last=False)
    for i, (img,_) in enumerate(dataloader_1):
        vae.eval()
        for j in range(5):
            _, _, recon_img = vae(img.cuda())
            save_tensor_as_img(recon_img, root='images/VAEGAN_recon/%s' %category_name, fmt='gray', normalize=True, name='%d_%d' %(i,j))
#     Save Sampled Images     
    for j in range(5):
        noise = torch.randn(200, hp_latent_dims).cuda()
        generated_img = vae.decoder(noise)
        save_tensor_as_img(generated_img, root='images/VAEGAN_sample/%s' %category_name, fmt='gray', normalize=True, name='%s_%d' %(category_name,j))


# In[ ]:


# Plot the Loss Progess of Discriminator and Decoder(Generator)
plot_line_graph(line1=D_losses, name1='Dis', line2=G_losses, name2='Gen', dpi=100, title='Loss progress history', xlabel='epoch', ylabel='')


# In[ ]:


# Plot the Loss Progess of Encoder
plot_line_graph(line1=E_losses, title='Encoder loss history', xlabel='epoch', ylabel='', dpi=100)

