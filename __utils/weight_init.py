import torch
import torch.nn as nn

def weight_init(model):
    last_layer_index = -1
    for i, m in enumerate(model.modules()):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            last_layer_index = i
        if i < last_layer_index:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, (nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.bias.data.fill_(0.01)