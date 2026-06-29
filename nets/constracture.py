import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'nets')))
import torch
from torchsummary import summary
from .TransUnet import get_transNet
from torch.nn import Module
from nets.vit_seg_modeling_resnet_skip import ResNetV2


# 创建 TransUnet 模型
model = get_transNet(n_classes=2)
# 移到适当设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(3, 256, 256), device="cuda" if torch.cuda.is_available() else "cpu")
