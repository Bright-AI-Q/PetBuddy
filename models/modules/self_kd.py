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
    def __init__(self, in_channels: int, num_classes: int, T: float = 4.0):
        """
        Self Knowledge Distillation module for stage features

        Args:
            channels_list: List of channel numbers for each stage
            T: Temperature parameter for KL divergence
            alpha: EMA update coefficient for teacher features
        """

        super().__init__()
        self.T = T

        # This is a small auxiliary classifier
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),  # Add a little Dropout to prevent overfitting
            nn.Linear(in_channels, num_classes)
        )
    def forward(self, student_features, teacher_logits):
        # 1. Student (middle layer) attempts to predict classification
        student_logits = self.aux_head(student_features)

        # 2. Teacher (final layer) provides soft labels
        # Note: Teacher's logits need to be detached, we don't backprop through teacher, only update student
        teacher_probs = F.softmax(teacher_logits.detach() / self.T, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)

        # 3. Calculate KL divergence
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.T ** 2)

        return kl_loss