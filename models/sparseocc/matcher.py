import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast


def focal_loss_cost(cls_pred, gt_labels, weight=2.0, alpha=0.25, gamma=2.0, eps=1e-12):
    """Compute focal loss cost between predictions and ground truth labels.

    This is a direct implementation replacing mmdet's FocalLossCost which
    requires InstanceData objects in mmdet 3.x.

    Args:
        cls_pred: [num_queries, num_classes] predicted logits
        gt_labels: [num_gt] ground truth class indices
        weight: cost weight
        alpha: focal loss alpha
        gamma: focal loss gamma
        eps: epsilon for numerical stability
    Returns:
        cost: [num_queries, num_gt] cost matrix
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cost * weight


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


# modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py#L70
class HungarianMatcher(BaseModule):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, mask_pred, class_pred, mask_gt, class_gt):
        """
        Args:
            mask_pred: [bs, num_query, num_voxel]
            class_pred: [bs, num_query, num_query_classes]
            mask_gt: [bs, num_voxel], value in range [0, num_obj - 1]
            class_gt: [[bs0_num_obj], [bs1_num_obj], ...], value in range [0, num_cls - 1]
        """
        bs, num_queries = class_pred.shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            tgt_ids = class_gt[b]
            num_instances = tgt_ids.shape[0]  # must be here, cause num of instances may change after masking

            # Handle no instances (e.g., semantic-only datasets)
            if num_instances == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # Compute the classification cost. We use focal loss provided by mmdet as sparsebev does
            out_prob = class_pred[b]
            cost_class = focal_loss_cost(out_prob, tgt_ids.long())

            out_mask = mask_pred[b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = mask_gt[b]

            tgt_mask = (tgt_mask.unsqueeze(-1) == torch.arange(num_instances).to(mask_gt.device))
            tgt_mask = tgt_mask.permute(1, 0)  # [Q, N]

            # all masks share the same set of points for efficient matching!
            tgt_mask = tgt_mask.view(tgt_mask.shape[0], -1)
            out_mask = out_mask.view(out_mask.shape[0], -1)

            with autocast('cuda', enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
