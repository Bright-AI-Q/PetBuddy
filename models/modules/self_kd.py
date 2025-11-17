#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: self_kd.py
====================================
Self Knowledge Distillation Module

Purpose:
- Implement self-knowledge distillation for pet recognition model training
- Provide feature-level distillation between teacher and student networks
- Support temperature-scaled KL divergence for soft target learning

Features:
1. Stage-wise Distillation: Separate distillation for each model stage
2. EMA Teacher: Exponentially moving average teacher feature updates
3. Temperature Scaling: Configurable temperature for soft target smoothing
4. Batch Consistency: Automatic handling of batch size changes
5. Lightweight Design: Minimal overhead with efficient KL divergence computation
"""

import torch.nn as nn
import torch.nn.functional as F


class SelfKD(nn.Module):
    def __init__(self, channels_list, T=4, alpha=0.999):
        """
        Self Knowledge Distillation module for stage features

        Args:
            channels_list: List of channel numbers for each stage
            T: Temperature parameter for KL divergence
            alpha: EMA update coefficient for teacher features
        """
        super().__init__()
        self.T = T
        self.alpha = alpha
        # Create adaptive average pooling for each stage
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in channels_list])
        self.teacher_features = None  # Store teacher model features

    def forward(self, student_features, stage_idx):
        """
        Forward pass for self knowledge distillation

        Args:
            student_features: Feature maps from student model (B, C, H, W)
            stage_idx: Index of the current stage (0, 1, 2)

        Returns:
            KL divergence loss for the given stage
        """
        # Global average pooling for student features
        student_pooled = self.pools[0](student_features)  # (B, C, 1, 1)
        student_logits = student_pooled.squeeze(-1).squeeze(-1)  # (B, C)

        if self.teacher_features is None:
            # Initialize teacher features
            self.teacher_features = student_logits.detach().clone()
        else:
            # Check batch size consistency
            if self.teacher_features.size(0) != student_logits.size(0):
                self.teacher_features = student_logits.detach().clone()
            else:
                # EMA update for teacher features
                self.teacher_features = (self.alpha * self.teacher_features +
                                       (1 - self.alpha) * student_logits.detach())

        # Calculate KL divergence loss
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(self.teacher_features / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        return kl_loss