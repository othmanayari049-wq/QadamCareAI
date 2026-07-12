import torch


def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    return (2 * intersection + eps) / (union + eps)


def iou_score(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    return (intersection + eps) / (union + eps)