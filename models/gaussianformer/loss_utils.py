"""Lovasz-Softmax loss.

Trimmed down from the reference implementation by Maxim Berman (MIT License)
to the flat multi-class case GaussianFormer uses.
"""

import torch


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t. the sorted errors (Alg. 1)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, ignore=None):
    """Multi-class Lovasz-Softmax over the classes present in ``labels``.

    Args:
        probas (Tensor): ``[P, C]`` class probabilities.
        labels (Tensor): ``[P]`` ground truth labels in ``[0, C)``.
        ignore (int, optional): label id excluded from the loss.
    """
    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]
    if probas.numel() == 0:
        return probas.sum() * 0.0

    losses = []
    for c in range(probas.shape[1]):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()
