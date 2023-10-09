import os
from pathlib import Path
import torch
from torchvision.utils import save_image
import warnings
from PIL import Image

def save_tensor_as_img(img, root, normalize=True, fmt='gray', name='image'):
#     If root does not yet exist, then create one
    if not os.path.exists(root):
        os.makedirs(root)
#     Convert img from [-1,1] to [0,1] if needed
    if normalize == True:
        img = (img + 1) / 2
    img = img.detach().squeeze().cpu().numpy()
#     Store the image one by one
    for i, img_np in enumerate(img):
#         Convert img from [0,1] to [0,255]
        img = Image.fromarray(img_np*255)
#         Store Grayscale images (Default)
        if fmt == 'gray':
            img = img.convert('L')
#         Store RGB images
        elif fmt == 'rgb' or 'RGB':
            img = img.convert('RGB')
        else:
            warnings.warn('\n''Please enter a valid image format from: 1. gray; 2. RGB')
            break
            
        img.save(os.path.join(root, '%s_%d.jpeg' %(name, (i+1))), 'JPEG')
        
    if name == 'image':
        warnings.warn('\n''Use the default name "image" as the file names')
        
        
def img_to_grid_save(img, root, name_num, normalize=True, nrow=3, fmt='gray', name='grid'):
#     If root does not yet exist, then create one
    if not os.path.exists(root):
        os.makedirs(root)
    grid = make_grid(img, cmap='gray', nrow=nrow, normalize=normalize).detach().squeeze().cpu()
    save_image(grid, root+'/%s_%d.png' %(name, name_num), normalize=False)
    if name == 'grid':
        warnings.warn('\n''Use the default name "image" as the file names')