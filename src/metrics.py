from __future__ import annotations

import torch


def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = targets.view(targets.size(0), -1)

    intersection = (pred_flat * tgt_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    tgt_sum = tgt_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (pred_sum + tgt_sum + smooth)
    union = pred_sum + tgt_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice.mean(), iou.mean()


def pixel_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    pred_flat = pred.view(-1)
    tgt_flat = targets.view(-1)

    tp = (pred_flat * tgt_flat).sum()
    fp = (pred_flat * (1.0 - tgt_flat)).sum()
    fn = ((1.0 - pred_flat) * tgt_flat).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall
