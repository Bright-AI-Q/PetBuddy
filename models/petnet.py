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
1. Modular Architecture: Interchangeable components (IRB, SelfKD, DualAttn)
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
from models.modules.dual_attn import ECAPos
from pathlib import Path

class PetNet(nn.Module):
    def __init__(self, num_classes: int = 144,
                 stage_repeats: List[int] = [2, 3, 4],
                 attn_cfg: Optional[Dict] = None,
                 selfkd_cfg: Optional[Dict] = None,
                 max_pets_per_image: int = 10,
            ):
        """
        Unified pet recognition network with modular design

        Args:
            num_classes: Number of pet categories
            stage_repeats: Block repetition counts for each stage [stage1, stage2, stage3]
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

        self.selfkd_cfg = selfkd_cfg  # Save selfkd_cfg as instance attribute

        # Initialize SelfKD module list
        self.selfkd_modules = nn.ModuleList([None, None, None])  # For 3 stages

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6()
        )

        # 3 Stages
        self.stage1 = self._make_stage(32, 48, stage_repeats[0], 2, attn_cfg, selfkd_cfg, 0)
        self.stage2 = self._make_stage(48, 96, stage_repeats[1], 2, attn_cfg, selfkd_cfg, 1)
        self.stage3 = self._make_stage(96, 192, stage_repeats[2], 2, attn_cfg, selfkd_cfg, 2)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
        if self._is_module_enabled(selfkd_cfg):

            stage_out_channels = [48, 96, 192]

            selfkd_cfg_filtered = {k: v for k, v in selfkd_cfg.items()
                                   if k not in ['enable', 'w', 'alpha']}


            for idx, out_c in enumerate(stage_out_channels):

                if idx < len(self.selfkd_modules):
                    self.selfkd_modules[idx] = SelfKD(
                        in_channels=out_c,
                        num_classes=self.num_classes,
                        **selfkd_cfg_filtered
                    )

    def _add_optional_modules(self, layers: List[nn.Module], out_c: int,
                            attn_cfg: Optional[Dict],
                            selfkd_cfg: Optional[Dict],
                            stage_idx: int) -> None:
        """Add optional modules based on their configs"""
        if self._is_module_enabled(attn_cfg):
            attn_cfg_filtered = {k: v for k, v in attn_cfg.items() if k not in ['enable', 'pos_enc']}
            eca_kernel = attn_cfg_filtered.pop('eca_kernel', attn_cfg["eca_kernel"])
            layers.append(ECAPos(out_c, H=7, W=7, eca_kernel=eca_kernel))
        # SelfKD is a knowledge distillation module, not used in the main forward path
        if self._is_module_enabled(selfkd_cfg) and stage_idx < len(self.selfkd_modules):
            selfkd_cfg_filtered = {k: v for k, v in selfkd_cfg.items()
                                   if k not in ['enable', 'w', 'alpha']}  # Filter out unnecessary keys



            self.selfkd_modules[stage_idx] = SelfKD(
                in_channels=out_c,
                num_classes=self.num_classes,
                **selfkd_cfg_filtered
            )

    def _is_module_enabled(self, cfg: Optional[Dict]) -> bool:
        """Check if a module is enabled in config"""
        return cfg is not None and cfg.get('enable', False)

    def _make_stage(self, in_c: int, out_c: int, n: int, stride: int,
                   attn_cfg: Optional[Dict],
                   selfkd_cfg: Optional[Dict],
                   stage_idx: int) -> nn.Sequential:
        """Create a stage with optional modules"""
        layers = []


        for i in range(n):
            s = stride if i == 0 else 1
            current_in_c = in_c if i == 0 else out_c
            layers.append(IRB(current_in_c, out_c, stride=s, expand_ratio=6))

        if self._is_module_enabled(attn_cfg):
            eca_kernel = attn_cfg.get('eca_kernel', 3)
            layers.append(ECAPos(out_c, H=7, W=7, eca_kernel=eca_kernel))
        return nn.Sequential(*layers)

    def forward_single(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process single pet input"""
        s1_out = self.stage1(self.stem(x))
        s2_out = self.stage2(s1_out)
        s3_out = self.stage3(s2_out)
        final_logits = self.head(s3_out)

        if not self.training or not self._is_module_enabled(self.selfkd_cfg):
            return final_logits

        total_kd_loss = 0.0
        stage_outputs = [s1_out, s2_out, s3_out]
        loss_weights = self.selfkd_cfg.get('w', [1.0, 1.0, 1.0])

        for i, stage_out in enumerate(stage_outputs):
            if i < len(self.selfkd_modules) and self.selfkd_modules[i] is not None:
                kd_loss = self.selfkd_modules[i](stage_out, final_logits)
                total_kd_loss += loss_weights[i] * kd_loss
        return final_logits, total_kd_loss

    def forward_multi(self, images: List[torch.Tensor])-> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process multiple pet inputs"""
        """
               Processes multiple pet inputs by iterating through them.
               """
        batch_logits = []
        total_kd_loss = 0.0 if self.training and self._is_module_enabled(self.selfkd_cfg) else None

        for img_tensor in images:
            output = self.forward_single(img_tensor.unsqueeze(0))

            if isinstance(output, tuple):
                logits, kd_loss = output
                if total_kd_loss is not None:
                    total_kd_loss += kd_loss
            else:
                logits = output

            batch_logits.append(logits.squeeze(0))

        stacked_logits = torch.stack(batch_logits)

        if total_kd_loss is not None:
            return stacked_logits, total_kd_loss / len(images)

        return stacked_logits

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



    def load_pretrained_weights(self, pretrained_path: str):
        """
        Load pre-trained weights from the official MobileNetV2 weight file.
        This enhanced method resolves layer name mismatches through programmatic mapping.
        """
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            print(f"⚠️ pre-trained weights don't exist: {pretrained_path}, will train from scratch.")
            return

        print(f"🔁 Loading ImageNet pre-trained weights: {pretrained_path}")

        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        model_dict = self.state_dict()

        # Create an empty state_dict, we only fill in matching weights
        new_state_dict = {}

        # Iterate through each layer of the pre-trained model
        for pretrained_key, pretrained_value in pretrained_dict.items():
            # We don't load the final classification head because our class numbers differ
            if pretrained_key.startswith('classifier'):
                continue

            # --- Smart Name Mapping ---
            # This is the core logic for translating official names 'features.X' to our 'stageY.Z'

            # Translate stem part
            if pretrained_key.startswith('features.0'):
                # 'features.0.0.weight' -> 'stem.0.weight'
                # 'features.0.1.weight' -> 'stem.1.weight'
                model_key = pretrained_key.replace('features.0', 'stem', 1)

            # Translate stages part
            # Official MobileNetV2 IRB blocks range from features.1 to features.17
            # Our model is divided into stage1, stage2, stage3
            else:
                try:
                    # Extract block index from name, e.g., 'features.1.conv.0.0.weight' -> 1
                    block_idx = int(pretrained_key.split('.')[1])

                    if 1 <= block_idx <= 2:  # Official blocks 1-2 correspond to our stage1
                        # 'features.1.XXX' -> 'stage1.0.XXX'
                        # 'features.2.XXX' -> 'stage1.1.XXX'
                        model_key = pretrained_key.replace(f'features.{block_idx}', f'stage1.{block_idx - 1}', 1)
                    elif 3 <= block_idx <= 6:  # Official blocks 3-6 correspond to our stage2
                        # 'features.3.XXX' -> 'stage2.0.XXX'
                        model_key = pretrained_key.replace(f'features.{block_idx}', f'stage2.{block_idx - 3}', 1)
                    elif 7 <= block_idx <= 13:  # Official blocks 7-13 correspond to our stage3
                        # 'features.7.XXX' -> 'stage3.0.XXX'
                        model_key = pretrained_key.replace(f'features.{block_idx}', f'stage3.{block_idx - 7}', 1)
                    else:
                        # Official model has additional layers that our model structure doesn't have, ignore them directly
                        continue
                except (ValueError, IndexError):
                    # If key format doesn't match expected pattern, skip directly
                    continue

            # Check if translated key exists in our model and if shapes match
            if model_key in model_dict and model_dict[model_key].shape == pretrained_value.shape:
                new_state_dict[model_key] = pretrained_value

        if not new_state_dict:
            print("❌ Error: No matching layers found after name mapping. Please check the structural differences between PetNet and MobileNetV2.")
            return

        print(f"✅ Successfully mapped and prepared to load {len(new_state_dict)} layers.")

        # Update model with loaded weights
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)
        print("✅ Pre-trained weights loading completed.")