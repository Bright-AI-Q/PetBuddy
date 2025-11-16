import torch, cv2
import torch.nn as nn
import torch.nn.functional as F

class ECAPos(nn.Module):
    def __init__(self, C, H=7, W=7, eca_kernel=3):
        super().__init__()
        # 确保卷积核大小为奇数，以便对称填充
        if eca_kernel % 2 == 0:
            eca_kernel += 1
        padding = (eca_kernel - 1) // 2
        self.eca = nn.Conv1d(C, C, kernel_size=eca_kernel, padding=padding, groups=C)
        # 不再固定位置编码的尺寸，改为在forward中动态调整
        self.base_H = H
        self.base_W = W
        self.pos_base = nn.Parameter(torch.randn(2, C, H, W) * 0.02)  # 2=xy

    def forward(self, x):               # B,C,H,W
        # ECA
        w = x.mean(dim=(2,3))           # B,C
        w = self.eca(w.unsqueeze(-1)).squeeze(-1).sigmoid()
        x = x * w.view(x.size(0), -1, 1, 1)

        # 动态调整位置编码尺寸以匹配输入
        B, C, H, W = x.shape
        if H != self.base_H or W != self.base_W:
            # 使用双线性插值调整位置编码尺寸
            pos = F.interpolate(self.pos_base, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos = self.pos_base

        # PosEnc
        x = x + pos[0] + pos[1]
        return x