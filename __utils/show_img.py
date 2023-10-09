import torch
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_img(image, num_img=False, normalize=False, dpi=200, axis='off', cmap='gray'):   
# Adjust the resolution of the plot
    plt.rcParams['figure.dpi'] = dpi
    plt.axis(axis)
# Acquire the number of image need to be plotted
    if num_img == False:
        num_img = image.shape[0]
# image size == 1
    if num_img == 1:
        if normalize == True:
            image = image * 0.5 + 0.5
        plt.imshow(image.squeeze().detach().cpu().permute(1,2,0), cmap=cmap)
# image size == 3
    elif num_img == 3:
        grid = make_grid(image, cmap=cmap, normalize=normalize)
        plt.imshow(grid.detach().cpu().permute(1,2,0), cmap=cmap)
# image size >= 3, plot square graph
    else:
        grid = make_grid(image, nrow=int(math.sqrt(num_img)), cmap=cmap, normalize=normalize)
        plt.imshow(grid.detach().cpu().permute(1,2,0), cmap=cmap)
        
    plt.show()