# models/modules/irb.py
import torch, torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual
def IRB(inp, oup, stride, expand_ratio):
    return InvertedResidual(inp, oup, stride, expand_ratio)