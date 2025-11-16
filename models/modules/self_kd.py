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
        # 为每个stage创建自适应平均池化层
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in channels_list])
        self.teacher_features = None  # 存储教师模型的特征

    def forward(self, student_features, stage_idx):
        """
        Forward pass for self knowledge distillation

        Args:
            student_features: Feature maps from student model (B, C, H, W)
            stage_idx: Index of the current stage (0, 1, 2)

        Returns:
            KL divergence loss for the given stage
        """
        # 全局平均池化学生特征
        student_pooled = self.pools[0](student_features)  # (B, C, 1, 1)
        student_logits = student_pooled.squeeze(-1).squeeze(-1)  # (B, C)

        if self.teacher_features is None:
            # 初始化教师特征
            self.teacher_features = student_logits.detach().clone()
        else:
            # 检查批次大小是否匹配
            if self.teacher_features.size(0) != student_logits.size(0):
                self.teacher_features = student_logits.detach().clone()
            else:
                # 指数移动平均更新教师特征
                self.teacher_features = (self.alpha * self.teacher_features +
                                       (1 - self.alpha) * student_logits.detach())

        # 计算KL散度损失
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(self.teacher_features / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        return kl_loss