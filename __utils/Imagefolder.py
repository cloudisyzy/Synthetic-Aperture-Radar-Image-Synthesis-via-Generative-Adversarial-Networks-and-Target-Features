import torch
from torchvision import datasets
import torchvision.transforms as transforms
import warnings

def Imagefolder(root='C:/Users/Administrator/Machine Learning/Pytorch/Generative Models/SAR_Generation/SAR_category', normalize=True, tf='Default', category_idx='Full'):
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