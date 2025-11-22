from typing import List, Union

import timm
import torch
from torch import nn


class Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=0.2,
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process single pet input"""
        x = self.backbone(x)
        return x

    def forward_multi(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Process multiple pet inputs"""
        batch_logits = []
        for img in images:
            logits = self.forward_single(img.unsqueeze(0))
            batch_logits.append(logits.squeeze(0))
        stacked_logits = torch.stack(batch_logits)
        return stacked_logits

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
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


