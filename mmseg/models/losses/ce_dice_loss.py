import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def multiclass_dice_loss(pred,
                         label,
                         smooth=1.0,
                         exponent=2.0,
                         class_weight=None,
                         ignore_index=255,
                         reduction='mean'):
    """Calculate multi-class dice loss from logits.

    Args:
        pred (torch.Tensor): Logits with shape [N, C, H, W].
        label (torch.Tensor): Labels with shape [N, H, W].
        smooth (float): Smoothing factor.
        exponent (float): Exponent value for denominator terms.
        class_weight (torch.Tensor | None): Per-class weights.
        ignore_index (int): Ignore label id.
        reduction (str): Reduction type in {'none', 'mean', 'sum'}.

    Returns:
        torch.Tensor: Reduced dice loss.
    """
    num_classes = pred.shape[1]
    pred_prob = F.softmax(pred, dim=1)

    valid_mask = (label != ignore_index)
    label_for_onehot = label.clone()
    label_for_onehot[~valid_mask] = 0
    target_onehot = F.one_hot(label_for_onehot, num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()
    pred_prob = pred_prob * valid_mask
    target_onehot = target_onehot * valid_mask

    dims = (0, 2, 3)
    intersection = (pred_prob * target_onehot).sum(dims)
    denominator = (pred_prob.pow(exponent) + target_onehot.pow(exponent)).sum(dims)
    dice_score = (2 * intersection + smooth) / (denominator + smooth)
    loss = 1 - dice_score

    if class_weight is not None:
        loss = loss * class_weight

    return weight_reduce_loss(loss, weight=None, reduction=reduction, avg_factor=None)

def ohem_cross_entropy(cls_score,
                       label,
                       class_weight=None,
                       ignore_index=255,
                       thresh=None,
                       min_kept=100000):
    """OHEM cross entropy for semantic segmentation.

    Args:
        cls_score (Tensor): [N, C, H, W]
        label (Tensor): [N, H, W]
        class_weight (Tensor | None): class weights for CE
        ignore_index (int): ignore label id
        thresh (float | None): loss threshold for keeping hard pixels
        min_kept (int): minimum number of hard pixels to keep

    Returns:
        Tensor: scalar CE loss after OHEM
    """
    # per-pixel CE loss
    loss = F.cross_entropy(
        cls_score,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    valid_mask = (label != ignore_index)
    valid_loss = loss[valid_mask]

    if valid_loss.numel() == 0:
        return cls_score.sum() * 0.0

    # sort descending: hardest first
    sorted_loss, _ = torch.sort(valid_loss, descending=True)

    if thresh is not None:
        kept_mask = sorted_loss > thresh
        kept_loss = sorted_loss[kept_mask]

        if kept_loss.numel() < min_kept:
            num_kept = min(min_kept, sorted_loss.numel())
            kept_loss = sorted_loss[:num_kept]
    else:
        num_kept = min(min_kept, sorted_loss.numel())
        kept_loss = sorted_loss[:num_kept]

    if kept_loss.numel() == 0:
        kept_loss = sorted_loss[:1]

    return kept_loss.mean()

@LOSSES.register_module()
class CEDiceLoss(nn.Module):
    """CrossEntropy + Dice loss for multi-class semantic segmentation."""

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 smooth=1.0,
                 exponent=2.0,
                 loss_weight=1.0):
        super(CEDiceLoss, self).__init__()
        assert use_sigmoid is False, 'CEDiceLoss only supports softmax multi-class segmentation.'
        assert use_mask is False, 'CEDiceLoss does not support mask loss.'
        self.reduction = reduction
        self.class_weight = class_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        ce_loss = F.cross_entropy(
            cls_score,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

        if weight is not None:
            weight = weight.float()
        ce_loss = weight_reduce_loss(
            ce_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        dice_loss = multiclass_dice_loss(
            cls_score,
            label,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=ignore_index,
            reduction=reduction)

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return self.loss_weight * loss

@LOSSES.register_module()
class CEDiceOHEMLoss(nn.Module):
    """CrossEntropy(OHEM) + Dice loss for multi-class semantic segmentation."""

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 smooth=1.0,
                 exponent=2.0,
                 loss_weight=1.0,
                 thresh=None,
                 min_kept=100000):
        super(CEDiceOHEMLoss, self).__init__()
        assert use_sigmoid is False, 'CEDiceOHEMLoss only supports softmax multi-class segmentation.'
        assert use_mask is False, 'CEDiceOHEMLoss does not support mask loss.'
        self.reduction = reduction
        self.class_weight = class_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight

        self.thresh = thresh
        self.min_kept = min_kept

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if reduction != 'mean':
            raise NotImplementedError('CEDiceOHEMLoss currently only supports reduction="mean".')

        if weight is not None:
            raise NotImplementedError('CEDiceOHEMLoss does not support per-pixel weight together with OHEM.')

        if avg_factor is not None:
            raise NotImplementedError('CEDiceOHEMLoss does not support avg_factor together with OHEM.')

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        ce_loss = ohem_cross_entropy(
            cls_score,
            label,
            class_weight=class_weight,
            ignore_index=ignore_index,
            thresh=self.thresh,
            min_kept=self.min_kept)

        dice_loss = multiclass_dice_loss(
            cls_score,
            label,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=ignore_index,
            reduction=reduction)

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return self.loss_weight * loss