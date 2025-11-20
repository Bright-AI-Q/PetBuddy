#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: petnet.py
====================================
Unified Pet Recognition Network (PetNet)

Purpose:
- Provide unified architecture for both single-pet and multi-pet recognition
- Support modular design with configurable components
- Enable both classification and detection tasks

Key Features:
1. Modular Architecture: Interchangeable components (IRB, LDRE, SelfKD, DualAttn)
2. Multi-mode Support: Handles both single-pet and multi-pet inputs
3. Integrated Techniques: Self-knowledge distillation and dual attention mechanisms
4. Configurable Design: Flexible configuration for each module
5. Production Ready: Optimized for both training and inference
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from models.modules.irb import IRB
from models.modules.self_kd import SelfKD
from models.modules.ldre import LDRE
from models.modules.dual_attn import ECAPos

class PetNet(nn.Module):
    def __init__(self, num_classes: int = 144,
                 stage_repeats: List[int] = [2, 3, 4],
                 ldre_cfg: Optional[Dict] = None,
                 attn_cfg: Optional[Dict] = None,
                 selfkd_cfg: Optional[Dict] = None,
                 max_pets_per_image: int = 10):
        """
        Unified pet recognition network with modular design

        Args:
            num_classes: Number of pet categories
            stage_repeats: Block repetition counts for each stage [stage1, stage2, stage3]
            ldre_cfg: Configuration for Local Dropout Random Erasing (LDRE)
                - enable: bool - whether to enable LDRE
                - prob: float - probability of applying LDRE
                - grid_size: int - grid size for LDRE
                - drop_count: int - number of regions to drop
            attn_cfg: Configuration for dual attention mechanism
                - enable: bool - whether to enable attention
                - pos_enc: str - type of positional encoding ('relative'/'absolute')
                - eca_kernel: int - kernel size for ECAAttention
            selfkd_cfg: Configuration for self-knowledge distillation
                - enable: bool - whether to enable self-KD
                - T: float - temperature parameter
                - w: List[float] - loss weights for each stage
                - alpha: float - interpolation weight
            max_pets_per_image: Maximum number of pets to process per image
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_pets_per_image = max_pets_per_image

        # Initialize SelfKD module list
        self.selfkd_modules = nn.ModuleList([None, None, None])  # For 3 stages

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6()
        )

        # 3 Stages
        self.stage1 = self._make_stage(32, 48, stage_repeats[0], 2, ldre_cfg, attn_cfg, selfkd_cfg, 0)
        self.stage2 = self._make_stage(48, 96, stage_repeats[1], 2, ldre_cfg, attn_cfg, selfkd_cfg, 1)
        self.stage3 = self._make_stage(96, 192, stage_repeats[2], 2, ldre_cfg, attn_cfg, selfkd_cfg, 2)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes)
        )

    def _add_optional_modules(self, layers: List[nn.Module], out_c: int,
                            ldre_cfg: Optional[Dict],
                            attn_cfg: Optional[Dict],
                            selfkd_cfg: Optional[Dict],
                            stage_idx: int) -> None:
        """Add optional modules based on their configs"""
        if self._is_module_enabled(ldre_cfg):
            ldre_cfg_filtered = {k: v for k, v in ldre_cfg.items() if k != 'enable'}
            layers.append(LDRE(**ldre_cfg_filtered))
        if self._is_module_enabled(attn_cfg):
            attn_cfg_filtered = {k: v for k, v in attn_cfg.items() if k not in ['enable', 'pos_enc']}
            eca_kernel = attn_cfg_filtered.pop('eca_kernel', attn_cfg["eca_kernel"])
            layers.append(ECAPos(out_c, H=7, W=7, eca_kernel=eca_kernel))
        # SelfKD is a knowledge distillation module, not used in the main forward path
        if self._is_module_enabled(selfkd_cfg) and stage_idx < len(self.selfkd_modules):
            # Remove 'enable' and 'w' keys from config before passing to SelfKD
            # SelfKD only accepts channels_list, T, alpha parameters
            selfkd_cfg_filtered = {k: v for k, v in selfkd_cfg.items()
                                 if k not in ['enable', 'w']}
            self.selfkd_modules[stage_idx] = SelfKD([out_c], **selfkd_cfg_filtered)

    def _is_module_enabled(self, cfg: Optional[Dict]) -> bool:
        """Check if a module is enabled in config"""
        return cfg is not None and cfg.get('enable', False)

    def _make_stage(self, in_c: int, out_c: int, n: int, stride: int,
                   ldre_cfg: Optional[Dict],
                   attn_cfg: Optional[Dict],
                   selfkd_cfg: Optional[Dict],
                   stage_idx: int) -> nn.Sequential:
        """Create a stage with optional modules"""
        layers = []
        for i in range(n):
            s = stride if i == 0 else 1
            layers.append(IRB(in_c if i == 0 else out_c, out_c, stride=s, expand_ratio=6))
            self._add_optional_modules(layers, out_c, ldre_cfg, attn_cfg, selfkd_cfg, stage_idx)
        return nn.Sequential(*layers)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process single pet input"""
        x = self.stem(x)

        # Collect stage outputs for SelfKD if needed
        stage_outs = []
        if self.training and any(self.selfkd_modules):
            for i, stage in enumerate([self.stage1, self.stage2, self.stage3]):
                x = stage(x)
                # Collect each stage output for SelfKD
                stage_outs.append(x)
        else:
            # Normal forward propagation
            for stage in [self.stage1, self.stage2, self.stage3]:
                x = stage(x)

        x = self.head(x)
        return (x, stage_outs) if stage_outs else x

    def forward_multi(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, List]:
        """Process multiple pet inputs"""
        batch_logits = []
        batch_stage_outs = [] if self.training else None

        for img in images:
            # Process each pet image
            logits, stage_outs = self.forward_single(img.unsqueeze(0))
            batch_logits.append(logits.squeeze(0))

            if batch_stage_outs is not None:
                batch_stage_outs.append(stage_outs)

        # Stack all results
        stacked_logits = torch.stack(batch_logits)

        if batch_stage_outs is not None:
            # Reorganize stage outputs
            stage_outputs = []
            for i in range(len(batch_stage_outs[0])):
                stage_outputs.append(torch.stack([outs[i] for outs in batch_stage_outs]))
            return stacked_logits, stage_outputs

        return stacked_logits, []

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, Tuple]:
        """
        Unified forward propagation

        Args:
            x: Can be single image tensor (B, 3, H, W) or list of multi-pet images [N, 3, H, W]

        Returns:
            Single pet: logits (B, num_classes) or (logits, stage_logits)
            Multi-pet: stacked logits (N, num_classes) or (stacked_logits, stage_logits)
        """
        if isinstance(x, list):
            # Multi-pet mode
            return self.forward_multi(x)
        else:
            # Single-pet mode
            return self.forward_single(x)

