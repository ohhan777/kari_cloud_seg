import torch
import torch.nn.functional as F


def bce_loss(preds, targets, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        preds: a tensor of shape [B, C, H, W]. For binary classification, C should be 1.
        targets: a tensor of shape [B, H, W].
        pos_weight: a tensor of shape [C] for class-wise loss weighting.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    # Ensure preds has the correct shape [B, 1, H, W] for binary classification
    if preds.shape[1] != 1:
        raise ValueError("For binary classification, preds should have shape [B, 1, H, W]")
    
    # Reshape targets to match preds
    targets = targets.unsqueeze(1)  # [B, H, W] to [B, 1, H, W]
    
    bce_loss = F.binary_cross_entropy_with_logits(
        preds.float(),
        targets.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(preds, targets, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        preds.float(),
        targets.long(),    # [B, H, W]
        ignore_index=ignore,
    )
    return ce_loss


def dice_loss(preds, targets):
    """Computes the Sørensen-Dice loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = preds.shape[1]
    
    # Ensure targets are long integers
    targets = targets.long()
    
    # Check if targets are already in the correct range
    if targets.max() >= num_classes or targets.min() < 0:
        raise ValueError(f"Target values should be in the range [0, {num_classes-1}]")
    
    true_1_hot = F.one_hot(targets, num_classes=num_classes)   # [B, H, W] to [B, H, W, C]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)                # [B, H, W, C] to [B, C, H, W]
    
    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()
    
    dims = (0,) + tuple(range(2, targets.ndimension() + 1))    # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)        # intersection w.r.t. the class
    cardinality = torch.sum(probas + true_1_hot, dims)         # cardinality w.r.t. the class
    
    dice_loss = (2. * intersection / (cardinality + 1e-7)).mean()
    return (1 - dice_loss)


def jaccard_loss(preds, targets):
    """Computes the Jaccard loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
    Returns:
        Jaccard loss
    """
    num_classes = preds.shape[1]
    
    # targets is already [B, H, W], so we don't need to squeeze
    true_1_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W] to [B, H, W, C]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)  # [B, H, W, C] to [B, C, H, W]
    
    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()
    
    dims = (0,) + tuple(range(2, targets.ndimension() + 1))  # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    
    jacc_loss = (intersection / (union + 1e-7)).mean()
    return (1 - jacc_loss)

