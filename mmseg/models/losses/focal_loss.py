import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from torch.autograd import Variable

@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal Loss for semantic segmentation.

    Args:
        gamma (float): focusing parameter. Default: 2.0
        alpha (float): balance factor. Default: 1.0
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'
        loss_weight (float): weight multiplier for the loss. Default: 1.0
        ignore_index (int): label to ignore. Default: 255
    """
    def __init__(self,
                 use_sigmoid=False,
                 gamma=2.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = reduction_override if reduction_override else self.reduction

        ce_loss = F.cross_entropy(
            cls_score,
            label,
            reduction='none',
            ignore_index=kwargs['ignore_index'])

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply external weight if provided (per-pixel)
        if weight is not None:
            focal_loss = focal_loss * weight

        if reduction == 'mean':
            loss = focal_loss.mean()
        elif reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss

        return self.loss_weight * loss
