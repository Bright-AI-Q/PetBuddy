import torch, cv2
import torch.nn as nn
import torch.nn.functional as F

class ECAPos(nn.Module):
    def __init__(self, C, H=7, W=7, eca_kernel=3):
        super().__init__()
        # Ensure kernel size is odd for symmetric padding
        if eca_kernel % 2 == 0:
            eca_kernel += 1
        padding = (eca_kernel - 1) // 2
        self.eca = nn.Conv1d(C, C, kernel_size=eca_kernel, padding=padding, groups=C)
        # Position encoding size is now dynamically adjusted in forward()
        self.base_H = H
        self.base_W = W
        self.pos_base = nn.Parameter(torch.randn(2, C, H, W) * 0.02)  # 2=xy

    def forward(self, x):               # (B,C,H,W)
        # ECA attention
        w = x.mean(dim=(2,3))           # (B,C)
        w = self.eca(w.unsqueeze(-1)).squeeze(-1).sigmoid()
        x = x * w.view(x.size(0), -1, 1, 1)

        # Dynamically adjust position encoding size to match input
        B, C, H, W = x.shape
        if H != self.base_H or W != self.base_W:
            # Resize position encoding using bilinear interpolation
            pos = F.interpolate(self.pos_base, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos = self.pos_base

        # Position encoding
        x = x + pos[0] + pos[1]
        return x