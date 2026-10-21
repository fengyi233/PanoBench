import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from models.csrc.wrapper import local_aggregate
from models.gaussianformer.loss_utils import lovasz_softmax
from models.gaussianformer.utils import get_rotation_matrix
from models.sparseocc.loss_utils import GeoScalLoss, SemScalLoss
from utils.data_converter import occ_dense2sparse
from utils.labels import get_dataset_meta


class LocalAggregator(nn.Module):
    """Splat the Gaussians onto the query points with the ``local_aggregate`` kernel."""

    def __init__(self, volume_size, voxel_origin, voxel_size, scale_multiplier):
        super().__init__()
        self.volume_size = tuple(int(v) for v in volume_size)
        self.voxel_size = float(voxel_size)
        self.scale_multiplier = float(scale_multiplier)
        self.register_buffer(
            'voxel_origin', torch.tensor(voxel_origin, dtype=torch.float).unsqueeze(0))

    def forward(self, pts, means3D, opacities, semantics, scales, cov3D):
        """
        Args:
            pts (Tensor): ``[1, N, 3]`` query points, in metric occupancy coordinates.
            means3D (Tensor): ``[1, G, 3]`` Gaussian centres.
            opacities (Tensor): ``[1, G]`` Gaussian opacities.
            semantics (Tensor): ``[1, G, C]`` per-Gaussian class logits.
            scales (Tensor): ``[1, G, 3]`` Gaussian scales, used for the cutoff radius.
            cov3D (Tensor): ``[1, G, 3, 3]`` inverse covariance matrices.

        Returns:
            Tensor: ``[1, N, C]`` aggregated class logits.
        """
        assert pts.shape[0] == 1, 'local_aggregate only supports batch size 1'
        assert not pts.requires_grad
        pts = pts.squeeze(0)
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        points_int = ((pts - self.voxel_origin) / self.voxel_size).to(torch.int)
        means3D_int = ((means3D.detach() - self.voxel_origin) / self.voxel_size).to(torch.int)
        for name, idx in (('query points', points_int), ('Gaussian centres', means3D_int)):
            assert idx.min() >= 0 and all(
                int(idx[:, i].max()) < self.volume_size[i] for i in range(3)
            ), f'{name} fall outside the occupancy grid'
        radii = torch.ceil(
            scales.max(dim=-1)[0] * self.scale_multiplier / self.voxel_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        logits = local_aggregate(pts, points_int, means3D, means3D_int, opacities, semantics,
                                 radii, cov3D, self.volume_size)
        return logits[None]


@MODELS.register_module()
class GaussianFormerV1Head(BaseModule):
    """Turn the refined Gaussians into dense occupancy logits and supervise them.

    The Gaussians only carry the ``num_classes - 1`` occupied classes; free space
    comes from a single always-on Gaussian covering the whole scene, whose logit
    is written into the ``free`` channel. That Gaussian has a fixed pose and
    opacity, so its per-voxel response is a constant (see ``_build_empty_weight``)
    and it never enters the aggregation kernel.
    """

    def __init__(self,
                 dataset_type,
                 occ_variant,
                 empty_mean,
                 empty_scale,
                 scale_multiplier=3,
                 apply_loss_type='random_1',
                 cls_weights=None,
                 loss_ce_weight=10.0,
                 loss_lovasz_weight=1.0,
                 use_sem_geo_scal_loss=False,
                 loss_geo_weight=1.0,
                 loss_sem_weight=1.0,
                 init_cfg=None):
        super().__init__(init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        self.volume_size = tuple(int(v) for v in meta.volume_size)
        self.num_classes = int(meta.num_classes)
        self.free_id = int(meta.free_id)

        self.loss_ce_weight = float(loss_ce_weight)
        self.loss_lovasz_weight = float(loss_lovasz_weight)
        self.apply_loss_type = apply_loss_type

        self.aggregator = LocalAggregator(
            volume_size=meta.volume_size,
            voxel_origin=meta.voxel_origin,
            voxel_size=meta.voxel_size,
            scale_multiplier=scale_multiplier)

        self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
        self.register_buffer('empty_mean', torch.tensor(empty_mean, dtype=torch.float)[None, None])
        self.register_buffer('empty_scale', torch.tensor(empty_scale, dtype=torch.float)[None, None])
        self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None])
        self.register_buffer('empty_opa', torch.ones(1)[None, None])

        if cls_weights is not None:
            assert len(cls_weights) == self.num_classes
            self.register_buffer('class_weights', torch.tensor(cls_weights, dtype=torch.float))
        else:
            self.class_weights = None

        if use_sem_geo_scal_loss:
            self.geo_loss = GeoScalLoss(
                self.num_classes, free_id=self.free_id, loss_weight=loss_geo_weight)
            self.sem_loss = SemScalLoss(
                self.num_classes, class_weights=cls_weights, loss_weight=loss_sem_weight)
        else:
            self.geo_loss = self.sem_loss = None

        self.register_buffer('occ_xyz', self._build_occ_xyz(meta), persistent=False)
        self.register_buffer('empty_weight', self._build_empty_weight(), persistent=False)

    @staticmethod
    def _build_occ_xyz(meta):
        """Metric coordinates of every voxel centre, flattened to ``[X * Y * Z, 3]``."""
        grids = [
            (torch.arange(size, dtype=torch.float) + 0.5) * meta.voxel_size + origin
            for size, origin in zip(meta.volume_size, meta.voxel_origin)
        ]
        return torch.stack(torch.meshgrid(*grids, indexing='ij'), dim=-1).flatten(0, 2)

    def _build_empty_weight(self):
        """Response of the free-space Gaussian at every voxel centre, ``[X * Y * Z]``.

        Its mean, scale, rotation and opacity are fixed buffers, so the response
        is a constant and only ``empty_scalar`` needs a gradient. Keeping it out
        of ``local_aggregate`` matters a lot: the backward kernel gives one thread
        to each Gaussian, and this one spans the whole grid.
        """
        m = torch.matmul(
            torch.diag_embed(self.empty_scale[0]), get_rotation_matrix(self.empty_rot[0]))
        cov_inv = torch.matmul(m.transpose(-1, -2), m).inverse()
        offset = self.occ_xyz - self.empty_mean.reshape(1, 3)
        power = -0.5 * (offset * torch.matmul(offset, cov_inv[0])).sum(dim=-1)
        return self.empty_opa.reshape(()) * power.exp()

    def _supervised_layers(self, num_layers):
        if not self.training or self.apply_loss_type == 'last':
            return [num_layers - 1]
        if self.apply_loss_type == 'all':
            return list(range(num_layers))
        if self.apply_loss_type.startswith('random_'):
            num_extra = int(self.apply_loss_type.split('_')[1]) - 1
            if num_extra < 1:
                return [num_layers - 1]
            extra = np.random.choice(num_layers - 1, num_extra, replace=False)
            return extra.tolist() + [num_layers - 1]
        raise NotImplementedError(f'Unsupported apply_loss_type: {self.apply_loss_type}')

    def prepare_gaussian_args(self, gaussian):
        means = gaussian.means
        scales = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics
        opacities = gaussian.opacities
        if opacities.numel() == 0:
            opacities = torch.ones_like(semantics[..., :1], requires_grad=False)

        assert semantics.shape[-1] == self.num_classes - 1, (
            f'Gaussians carry {semantics.shape[-1]} semantic channels, '
            f'expected {self.num_classes - 1} (all classes except free)')
        # The free channel stays empty here; it is filled analytically in `forward`.
        # The kernel is only instantiated for the class counts in `config.h`, so we
        # keep the padded channel rather than running it with `num_classes - 1`.
        zeros = torch.zeros_like(semantics[..., :1])
        semantics = torch.cat(
            [semantics[..., :self.free_id], zeros, semantics[..., self.free_id:]], dim=-1)

        bs, g = means.shape[:2]
        s = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        s[..., 0, 0] = scales[..., 0]
        s[..., 1, 1] = scales[..., 1]
        s[..., 2, 2] = scales[..., 2]
        m = torch.matmul(s, get_rotation_matrix(rotations))
        cov_inv = torch.matmul(m.transpose(-1, -2), m).float().inverse()
        return means, opacities.reshape(bs, g), semantics, scales, cov_inv

    def forward(self, gaussians):
        """
        Args:
            gaussians (List[GaussianPrediction]): one entry per refinement layer.

        Returns:
            dict: ``pred_occ`` holds ``[B, num_classes, X * Y * Z]`` logits for
            every supervised layer, ordered so that the last entry is the final one.
        """
        query = self.occ_xyz[None]
        free = (self.empty_scalar * self.empty_weight).view(1, -1, 1)
        preds = []
        for idx in self._supervised_layers(len(gaussians)):
            means, opacities, semantics, scales, cov_inv = self.prepare_gaussian_args(gaussians[idx])
            logits = self.aggregator(query, means, opacities, semantics, scales, cov_inv)
            logits = torch.cat([
                logits[..., :self.free_id],
                free.expand(logits.shape[0], -1, -1),
                logits[..., self.free_id + 1:],
            ], dim=-1)
            preds.append(logits.transpose(1, 2))
        return dict(pred_occ=preds)

    def loss(self, voxel_semantics, valid_mask, outs):
        target = voxel_semantics.flatten(1).long()
        preds = outs['pred_occ']
        if valid_mask is not None:
            mask = valid_mask.flatten(1).bool()
            target = target[mask][None]
            preds = [pred.transpose(1, 2)[mask][None].transpose(1, 2) for pred in preds]

        weight = None if self.class_weights is None else self.class_weights.to(preds[0].dtype)
        losses = {k: preds[0].new_zeros(()) for k in ('loss_occ_ce', 'loss_occ_lovasz')}
        if self.geo_loss is not None:
            losses['loss_occ_geo_scal'] = preds[0].new_zeros(())
            losses['loss_occ_sem_scal'] = preds[0].new_zeros(())

        for pred in preds:
            losses['loss_occ_ce'] = losses['loss_occ_ce'] + F.cross_entropy(pred, target, weight=weight)
            losses['loss_occ_lovasz'] = losses['loss_occ_lovasz'] + lovasz_softmax(
                pred.softmax(dim=1).transpose(1, 2).flatten(0, 1),
                target.flatten(),
                ignore=self.free_id)
            if self.geo_loss is not None:
                losses['loss_occ_geo_scal'] = losses['loss_occ_geo_scal'] + self.geo_loss(pred, target)
                losses['loss_occ_sem_scal'] = losses['loss_occ_sem_scal'] + self.sem_loss(pred, target)

        losses['loss_occ_ce'] = losses['loss_occ_ce'] * (self.loss_ce_weight / len(preds))
        losses['loss_occ_lovasz'] = losses['loss_occ_lovasz'] * (self.loss_lovasz_weight / len(preds))
        if self.geo_loss is not None:
            losses['loss_occ_geo_scal'] = losses['loss_occ_geo_scal'] / len(preds)
            losses['loss_occ_sem_scal'] = losses['loss_occ_sem_scal'] / len(preds)
        return losses

    def get_occ(self, outs):
        dense = outs['pred_occ'][-1].argmax(dim=1).reshape(-1, *self.volume_size)
        results = []
        for pred in dense:
            sparse = occ_dense2sparse(pred, free_id=self.free_id)
            results.append(dict(
                occ_loc=sparse[:, :3].detach().cpu().numpy(),
                sem_pred=sparse[:, 3].detach().cpu().numpy(),
            ))
        return results
