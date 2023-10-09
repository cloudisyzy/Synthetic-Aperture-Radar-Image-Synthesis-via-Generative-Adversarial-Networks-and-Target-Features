import PIL
from PIL import Image
import torch
from torchvision import transforms

import warnings
from functions import *
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure

def find_most_similar_img(img, source_folder, display=True, dpi=200):
#     Convert PIL.Image object to torch.Tensor if needed
    if isinstance(img, PIL.JpegImagePlugin.JpegImageFile):
        img = transforms.functional.to_tensor(img)
#     Automatic detect and denormalize tensors with the range of [-1,1]
    warnings.warn('\n''"find_most_similar_img" funtion contains automatic denormaliztion, make sure the source_folder do not apply any normalzation operation.')
    if torch.min(img) < 0:
        img = img * 0.5 + 0.5
    img = img.unsqueeze(0).detach().cpu()
#     Initialize the comparison parameter
    max_ssim = 0
    min_mse = 1
#     Search throughout the source folder
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
#     Display and return the target images
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