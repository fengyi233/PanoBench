import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from utils.labels import get_dataset_meta


@MODELS.register_module()
class GaussianOccEncoder(BaseModule):
    """Iteratively refine the Gaussian anchors produced by the lifter.

    ``operation_order`` names one entry of ``op_config_map`` per layer, so the
    decoder depth and the layout of each decoder block are fully described by
    the config.
    """

    def __init__(self,
                 dataset_type,
                 occ_variant,
                 anchor_encoder: dict,
                 norm_layer: dict,
                 ffn: dict,
                 deformable_model: dict,
                 refine_layer: dict,
                 spconv_layer: dict = None,
                 operation_order=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        semantic_dim = int(meta.num_classes) - 1
        pc_range = list(meta.point_cloud_range)

        anchor_encoder = dict(anchor_encoder, semantic_dim=semantic_dim)
        refine_layer = dict(refine_layer, semantic_dim=semantic_dim, pc_range=pc_range)
        deformable_model = dict(deformable_model)
        deformable_model['kps_generator'] = dict(
            deformable_model['kps_generator'], pc_range=pc_range)
        if spconv_layer is not None:
            spconv_layer = dict(
                spconv_layer, pc_range=pc_range, grid_size=list(meta.voxel_size_3d))

        self.operation_order = list(operation_order)
        self.op_config_map = {
            'norm': norm_layer,
            'ffn': ffn,
            'deformable': deformable_model,
            'refine': refine_layer,
            'spconv': spconv_layer,
        }
        unknown = set(self.operation_order) - set(self.op_config_map)
        assert not unknown, f'Unsupported operations in operation_order: {sorted(unknown)}'

        self.anchor_encoder = MODELS.build(anchor_encoder)
        self.layers = nn.ModuleList(
            [MODELS.build(self.op_config_map[op]) for op in self.operation_order])

    def init_weights(self):
        for op, layer in zip(self.operation_order, self.layers):
            if op == 'refine':
                continue
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, 'init_weight'):
                m.init_weight()

    def forward(self, anchor, instance_feature, ms_img_feats, metas):
        anchor_embed = self.anchor_encoder(anchor)

        gaussians = []
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](instance_feature, anchor)
            elif op in ('norm', 'ffn'):
                instance_feature = self.layers[i](instance_feature)
            elif op == 'deformable':
                instance_feature = self.layers[i](
                    instance_feature, anchor, anchor_embed, ms_img_feats, metas)
            else:
                anchor, gaussian = self.layers[i](instance_feature, anchor, anchor_embed)
                gaussians.append(gaussian)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
        return gaussians
