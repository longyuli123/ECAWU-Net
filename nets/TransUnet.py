import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
# from torchsummary import summary
from .vit_seg_modeling_edge_enhanced import VisionTransformer as ViT_seg
from .vit_seg_modeling_edge_enhanced import CONFIGS as CONFIGS_ViT_seg

def get_transNet(n_classes):
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes,use_ssa_in_encoder=True)
    return net


if __name__ == '__main__':#3
    # net = get_transNet(1)
    # img = torch.randn((2, 3, 512, 512))
    # segments = net(img)
    # print(segments.size())
    # for edge in edges:
    #     print(edge.size())

    # Create the TransUnet model with n_classes=1 (as in the example in TransUnet.py)
    model = get_transNet(n_classes=2)

    # Move model to CPU (can change to 'cuda' if GPU is available)
    device = torch.device("cpu")
    model = model.to(device)

    # Print model structure using torchsummary
    # Input size: (channels, height, width) = (3, 256, 256)
    summary(model, input_size=(3, 256, 256))

    



