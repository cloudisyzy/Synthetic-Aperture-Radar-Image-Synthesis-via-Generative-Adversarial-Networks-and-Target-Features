import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import warnings

def calc_SSIM(img1, img2, mode, epoch=20):
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