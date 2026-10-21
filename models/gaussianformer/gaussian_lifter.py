import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from models.gaussianformer.utils import safe_inverse_sigmoid
from utils.labels import get_dataset_meta


@MODELS.register_module()
class GaussianLifter(BaseModule):
    """Initialise the learnable 3D Gaussian queries consumed by the encoder.

    Each anchor is ``[xyz(3), scale(3), rotation(4), opacity(0|1), semantics]``
    where the semantic channels cover every class except ``free``: emptiness is
    modelled by a separate always-on Gaussian inside the head.
    """

    def __init__(self,
                 dataset_type,
                 occ_variant,
                 num_anchor,
                 embed_dims,
                 anchor_grad=True,
                 feat_grad=True,
                 include_opa=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        semantic_dim = int(meta.num_classes) - 1

        self.num_anchor = int(num_anchor)
        self.embed_dims = int(embed_dims)

        xyz = safe_inverse_sigmoid(torch.rand(self.num_anchor, 3, dtype=torch.float))
        scale = safe_inverse_sigmoid(torch.rand_like(xyz))
        rots = torch.zeros(self.num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((self.num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((self.num_anchor, 0), dtype=torch.float)
        semantic = torch.randn(self.num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.anchor = nn.Parameter(anchor.clone(), requires_grad=anchor_grad)
        self.instance_feature = nn.Parameter(
            torch.zeros([self.num_anchor, self.embed_dims]), requires_grad=feat_grad)
        self.register_buffer('anchor_init', anchor.clone(), persistent=False)

    def init_weights(self):
        self.anchor.data.copy_(self.anchor_init)
        if self.instance_feature.requires_grad:
            nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, img_feats):
        batch_size = img_feats[0].shape[0]
        return {
            'representation': self.anchor[None].tile((batch_size, 1, 1)),
            'rep_features': self.instance_feature[None].tile((batch_size, 1, 1)),
        }
