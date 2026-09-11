import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


def bce_iou_loss(ptr, atr, pos_weight=1.0, iou_weight=1.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(
        ptr, atr, pos_weight=torch.tensor([pos_weight], device=ptr.device), reduction=reduction
    )

    pred = torch.sigmoid(ptr)
    pred_flat = pred.view(pred.size(0), -1)  # [B, H*W]
    atr_flat = atr.view(atr.size(0), -1)  # [B, H*W]

    intersection = (pred_flat * atr_flat).sum(dim=1)
    iou_union = pred_flat.sum(dim=1) + atr_flat.sum(dim=1)- intersection

    iou = 1 - (intersection + 1e-5) / (iou_union + 1e-5)
    iou = iou.mean() if reduction == 'mean' else iou.sum()
    total_loss = bce + iou_weight * iou
    return total_loss


def bce_dice_iou_loss(
    ptr,
    atr,
    pos_weight=1.0,
    dice_weight=1.0,
    iou_weight=1.0,
    reduction='mean',
):
    """BCE + Dice + IoU loss used by the reference PatchSelector run.py."""
    target = atr.to(dtype=ptr.dtype)
    positive_weight = torch.as_tensor(
        pos_weight, dtype=ptr.dtype, device=ptr.device
    ).reshape(1)
    bce = F.binary_cross_entropy_with_logits(
        ptr,
        target,
        pos_weight=positive_weight,
        reduction=reduction,
    )

    pred = torch.sigmoid(ptr)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)

    dice_union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1e-5) / (dice_union + 1e-5)

    iou_union = dice_union - intersection
    iou = 1.0 - (intersection + 1e-5) / (iou_union + 1e-5)

    if reduction == 'mean':
        dice, iou = dice.mean(), iou.mean()
    elif reduction == 'sum':
        dice, iou = dice.sum(), iou.sum()
    elif reduction != 'none':
        raise ValueError(f"Unsupported reduction: {reduction}")

    return bce + dice_weight * dice + iou_weight * iou






