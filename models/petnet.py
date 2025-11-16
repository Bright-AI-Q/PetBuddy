"""
统一宠物识别网络 - 支持单宠物和多宠物模式
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from models.modules.irb import IRB
from models.modules.self_kd import SelfKD
from models.modules.ldre import LDRE
from models.modules.dual_attn import ECAPos

class PetNet(nn.Module):
    def __init__(self, num_classes: int = 37,
                 stage_repeats: List[int] = [2, 3, 4],
                 ldre_cfg: Optional[Dict] = None,
                 attn_cfg: Optional[Dict] = None,
                 selfkd_cfg: Optional[Dict] = None,
                 max_pets_per_image: int = 10):
        """
        统一的宠物识别网络

        Args:
            num_classes: 类别数量
            stage_repeats: 各阶段重复块数
            ldre_cfg: LDRE配置
            attn_cfg: 注意力配置
            selfkd_cfg: 自知识蒸馏配置
            max_pets_per_image: 每张图像最大宠物数
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_pets_per_image = max_pets_per_image

        # 初始化SelfKD模块列表
        self.selfkd_modules = nn.ModuleList([None, None, None])  # 对应3个stage

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6()
        )

        # 3 Stages
        self.stage1 = self._make_stage(24, 40, stage_repeats[0], 2, ldre_cfg, attn_cfg, selfkd_cfg, 0)
        self.stage2 = self._make_stage(40, 80, stage_repeats[1], 2, ldre_cfg, attn_cfg, selfkd_cfg, 1)
        self.stage3 = self._make_stage(80, 160, stage_repeats[2], 2, ldre_cfg, attn_cfg, selfkd_cfg, 2)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(160, num_classes)
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
            eca_kernel = attn_cfg_filtered.pop('eca_kernel', 3)
            layers.append(ECAPos(out_c, H=7, W=7, eca_kernel=eca_kernel))
        # SelfKD是知识蒸馏模块，不在前向传播的主路径中使用
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
        """处理单宠物输入"""
        x = self.stem(x)

        # 如果需要stage输出用于SelfKD
        stage_outs = []
        if self.training and any(self.selfkd_modules):
            for i, stage in enumerate([self.stage1, self.stage2, self.stage3]):
                x = stage(x)
                # 收集每个stage的输出用于SelfKD
                stage_outs.append(x)
        else:
            # 正常前向传播
            for stage in [self.stage1, self.stage2, self.stage3]:
                x = stage(x)

        x = self.head(x)
        return (x, stage_outs) if stage_outs else x

    def forward_multi(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, List]:
        """处理多宠物输入"""
        batch_logits = []
        batch_stage_outs = [] if self.training else None

        for img in images:
            # 处理每个宠物图像
            logits, stage_outs = self.forward_single(img.unsqueeze(0))
            batch_logits.append(logits.squeeze(0))

            if batch_stage_outs is not None:
                batch_stage_outs.append(stage_outs)

        # 堆叠所有结果
        stacked_logits = torch.stack(batch_logits)

        if batch_stage_outs is not None:
            # 重组stage输出
            stage_outputs = []
            for i in range(len(batch_stage_outs[0])):
                stage_outputs.append(torch.stack([outs[i] for outs in batch_stage_outs]))
            return stacked_logits, stage_outputs

        return stacked_logits

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, Tuple]:
        """
        统一的前向传播

        Args:
            x: 可以是单张图像 (B, 3, H, W) 或多宠物图像列表 [N, 3, H, W]

        Returns:
            单宠物: logits (B, num_classes) 或 (logits, stage_logits)
            多宠物: 堆叠的logits (N, num_classes) 或 (stacked_logits, stage_logits)
        """
        if isinstance(x, list):
            # 多宠物模式
            return self.forward_multi(x)
        else:
            # 单宠物模式
            return self.forward_single(x)

