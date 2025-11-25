#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: irb.py
====================================
Inverted Residual Block (IRB) Module

Purpose:
- Provide standardized Inverted Residual Block implementation for PetNet
- Support efficient MobileNetV2-style bottleneck architecture
- Enable configurable expansion ratios and stride options

Features:
1. Standard Implementation: Based on torchvision's InvertedResidual
2. Lightweight Design: Efficient computation with depthwise separable convolutions
3. Flexible Configuration: Support for different expansion ratios and strides
4. Integration Ready: Seamless integration with PetNet architecture
5. Performance Optimized: Optimized for both training and inference
"""

import torch, torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual

def IRB(inp, oup, stride, expand_ratio):

        return InvertedResidual(inp, oup, stride, expand_ratio)
