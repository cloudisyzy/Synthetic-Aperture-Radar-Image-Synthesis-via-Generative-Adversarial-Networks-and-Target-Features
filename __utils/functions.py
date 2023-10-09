import os
from pathlib import Path
import math
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import warnings


def calc_MSE(img1, img2, mode, epoch=20):
#    This function aims to calculate the Mean-Square Error between two batches
    mse_history = []
    MSE = MeanSquaredError(data_range=1.0)
    # Mode-1, two single batches comparison
    if mode == 'bb':
        mse = MSE(img1.detach().cpu(), img2.detach().cpu())
        return mse
    # Mode-2, a single batch vs a dataloader
    elif mode == 'bd':
        images1 = img1.detach().cpu()
        for i in range(epoch):
            images2,_ = next(iter(img2))
            if images1.shape[0] == images2.shape[0]:
                mse_history.append(MSE(images1, images2).item())
        return torch.mean(torch.tensor(mse_history))
    # Mode-3, dataloader vs dataloader
    elif mode == 'dd':
        for i in range(epoch):
            images1,_ = next(iter(img1))
            images2,_ = next(iter(img2))
            if images1.shape[0] == images2.shape[0]:
                mse_history.append(MSE(images1, images2).item())
        return torch.mean(torch.tensor(mse_history))
    # Warning, if the mode is not determined
    else:
        warnings.warn('Please specify a certain type of mode from: "bb", "bd", "dd"')
       
    
def calc_SSIM(img1, img2, mode, epoch=20):
#    This function aims to calculate the Structural Similarity between two batches
    ssim_history = []
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Mode-1, two single batches comparison
    if mode == 'bb':
        ssim = SSIM(img1.detach().cpu(), img2.detach().cpu())
        return ssim
    # Mode-2, a single batch vs a dataloader
    elif mode == 'bd':
        images1 = img1.detach().cpu()
        for i in range(epoch):
            images2,_ = next(iter(img2))
            if images1.shape[0] == images2.shape[0]:
                ssim_history.append(SSIM(images1, images2).item())
        return torch.mean(torch.tensor(ssim_history))
    # Mode-3, dataloader vs dataloader
    elif mode == 'dd':
        for i in range(epoch):
            images1,_ = next(iter(img1))
            images2,_ = next(iter(img2))
            if images1.shape[0] == images2.shape[0]:
                ssim_history.append(SSIM(images1, images2).item())
        return torch.mean(torch.tensor(ssim_history))
    # Warning, if the mode is not determined
    else:
        warnings.warn('Please specify a certain type of mode from: "bb", "bd", "dd"')


def find_most_similar_img(img, source_folder, display=True, dpi=200):
#    Aims to find the most similar image of the target image from a source folder according to minMSE and maxSSIM
    # Convert PIL.Image object to torch.Tensor if needed
    if isinstance(img, PIL.JpegImagePlugin.JpegImageFile):
        img = transforms.functional.to_tensor(img)
    # Automatic detect and denormalize tensors with the range of [-1,1]
    warnings.warn('\n''"find_most_similar_img" funtion contains automatic denormaliztion, make sure the source_folder do not apply any normalzation operation.')
    if torch.min(img) < 0:
        img = img * 0.5 + 0.5
    img = img.unsqueeze(0).detach().cpu()
    # Initialize the comparison parameter
    max_ssim = 0
    min_mse = 1
    # Search throughout the source folder
    for data in source_folder:
        source_img, label = data
        source_img.unsqueeze_(0)
        ssim = calc_SSIM(img, source_img, mode='bb').item()
        mse = calc_MSE(img, source_img, mode='bb').item()
        if ssim > max_ssim:
            img_max_ssim = source_img
            max_ssim = ssim
        if mse < min_mse:
            img_min_mse = source_img
            min_mse = mse
            
    img = img.squeeze(0)
    img_max_ssim = img_max_ssim.squeeze(0)
    img_min_mse = img_min_mse.squeeze(0)
    # Display and return the target images
    if display == True:
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.size'] = 4
        plt.figure(figsize=(3, 1))
        plt.subplot(1,3,1)
        plt.imshow(img.permute(1,2,0), cmap='gray')
        plt.axis('off')
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(img_max_ssim.permute(1,2,0), cmap='gray')
        plt.axis('off')
        plt.title('Image with Max SSIM \n (%.4f)' %max_ssim)
        plt.subplot(1,3,3)
        plt.imshow(img_min_mse.permute(1,2,0), cmap='gray')
        plt.axis('off')
        plt.title('Image with Min MSE \n (%.4f)' %min_mse)
        
    return (img_max_ssim, max_ssim, img_min_mse, min_mse)
    
    
def Imagefolder(root='C:/Users/Administrator/Machine Learning/Pytorch/Generative Models/SAR_Generation/SAR_category', normalize=True, tf='Default', category_idx='Full'):
#    This function is a extended version of original ImageFolder, is able to return the whole SAR dataset or a sub-class of SAR image    
    # Preprocessing of images
    if tf == 'Default' and normalize == True:
        tf = transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5),
                               transforms.Grayscale(num_output_channels=1),
                           ])
        warnings.warn('\n''Use default preprocessing techniques with normalization')
        
    elif tf == 'Default' and normalize == False:
        tf = transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Grayscale(num_output_channels=1),
                           ])
        warnings.warn('\n''Use default preprocessing techniques with no normalization')
        
    else:
        warnings.warn('\n''Use custom image transform')
    dataset = datasets.ImageFolder(root=root, transform=tf)
    # Return the whole SAR dataset
    if category_idx == "Full":
        warnings.warn('\n''Return the whole SAR dataset')
        return dataset
    # Return the one of the ten subset
    else:
        class_indices = [i for i, (x, y) in enumerate(dataset) if y == category_idx]
        subset = torch.utils.data.Subset(dataset, class_indices)
        warnings.warn('\n''Return the subset of class %d' %category_idx)
        return (subset, category_idx)
    
    
def save_tensor_as_img(img, root, normalize=True, fmt='gray', name='image'):
#    This function can fastly save torch.tensor as jpeg images
    # If root does not yet exist, then create one
    if not os.path.exists(root):
        os.makedirs(root)
    # Convert img from [-1,1] to [0,1] if needed
    if normalize == True:
        img = (img + 1) / 2
    img = img.detach().squeeze().cpu().numpy()
    # Store the image one by one
    for i, img_np in enumerate(img):
        # Convert img from [0,1] to [0,255]
        img = Image.fromarray(img_np*255)
        # Store Grayscale images (Default)
        if fmt == 'gray':
            img = img.convert('L')
        # Store RGB images
        elif fmt == 'rgb' or 'RGB':
            img = img.convert('RGB')
        else:
            warnings.warn('\n''Please enter a valid image format from: 1. gray; 2. RGB')
            break
            
        img.save(os.path.join(root, '%s_%d.jpeg' %(name, (i+1))), 'JPEG')
        
    if name == 'image':
        warnings.warn('\n''Use the default name "image" as the file names')
        
        
def img_to_grid_save(img, root, name_num, normalize=True, nrow=3, fmt='gray', name='grid'):
#    Sometimes user may want to save an image batch consists of multiple images, this function can do so
    if not os.path.exists(root):
        os.makedirs(root)
    grid = make_grid(img, cmap='gray', nrow=nrow, normalize=normalize).detach().squeeze().cpu()
    save_image(grid, root+'/%s_%d.png' %(name, name_num), normalize=False)
    if name == 'grid':
        warnings.warn('\n''Use the default name "image" as the file names')
            
            
def show_img(image, num_img=False, normalize=False, dpi=200, axis='off', cmap='gray', title=False): 
#    This function can fastly display torch.tensor as images
    # Adjust the resolution of the plot and add title
    plt.rcParams['figure.dpi'] = dpi
    plt.axis(axis)
    if title != False:
        plt.title(title)
    # Acquire the number of image need to be plotted
    if num_img == False:
        num_img = image.shape[0]
    # image size == 1
    if num_img == 1:
        if normalize == True:
            image = image * 0.5 + 0.5
        if image.dim() == 3:
            plt.imshow(image.detach().cpu().permute(1,2,0), cmap=cmap)
        elif image.dim() == 4:
            plt.imshow(image.squeeze(0).detach().cpu().permute(1,2,0), cmap=cmap)
    # image size == 3
    elif num_img == 3:
        grid = make_grid(image, cmap=cmap, normalize=normalize)
        plt.imshow(grid.detach().cpu().permute(1,2,0), cmap=cmap)
    # image size >= 3, plot square graph
    else:
        grid = make_grid(image, nrow=int(math.sqrt(num_img)), cmap=cmap, normalize=normalize)
        plt.imshow(grid.detach().cpu().permute(1,2,0), cmap=cmap)

    plt.show()
    

def plot_line_graph(line1, name1=False, name2=False, line2=False, xlabel='x', ylabel='y', title='title', xlim=False, ylim=False, dpi=500):
#    A fast version of plot line graph, usually used for visualizing loss progresses
    # Set dpi of figure
    plt.rcParams['figure.dpi'] = dpi
    # Uniform input type as lists
    if isinstance(line1, torch.Tensor):
        line1 = line1.tolist()
    if isinstance(line2, torch.Tensor):
        line2 = line2.tolist()
    # Plot a graph with one line
    if line2 == False:
        xlen = torch.arange(len(line1))
        plt.plot(xlen, line1)
        
    # Plot two lines in a graph
    else:
        if len(line1) != len(line2):
            warnings.warn('\n''The length of the two inputs are not the same')
        else:
            xlen = torch.arange(len(line1))
            plt.plot(xlen, line1)
            plt.plot(xlen, line2)
            plt.legend([name1, name2], loc='upper right')
    
    if (xlim != False):
        plt.xlim(*xlim)
    if (ylim != False):
        plt.ylim(*ylim)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# def weight_init(model):
# #     Initialization for NN models
#     last_layer_index = -1
#     for i, m in enumerate(model.modules()):
#         if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
#             last_layer_index = i
#         if i < last_layer_index:
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.xavier_normal_(m.weight)
#                 m.bias.data.fill_(0.01)
#             if isinstance(m, (nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
#                 nn.init.normal_(m.weight, mean=0, std=1)
#                 m.bias.data.fill_(0.01)


def weight_init(model):
#     Initialization for NN models
    last_layer_index = -1
    for i, m in enumerate(model.modules()):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            last_layer_index = i
        if i < last_layer_index:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, (nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        
def reset_grad(solver_D, solver_G):
#    Reset the gradient of a network
    solver_D.zero_grad()
    solver_G.zero_grad()
    

def find_key(dictionary, value):
#    Return the key of a dict according to its value
    for key, val in dictionary.items():
        if val == value:
            return key