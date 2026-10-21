import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

from utils.labels import get_dataset_meta


@TRANSFORMS.register_module()
class RandomTransformImage(BaseTransform):
    def __init__(self, ida_aug_conf=None, test_mode=False):
        super().__init__()
        self.ida_aug_conf = ida_aug_conf
        self.test_mode = test_mode

    def transform(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

        if len(results['occ2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = np.asarray(results['img'][i], dtype=np.uint8)

                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = img
                results['occ2img'][i] = ida_mat @ results['occ2img'][i]

        elif len(results['occ2img']) % len(results['img']) == 0:
            for i in range(len(results['img'])):
                img = np.asarray(results['img'][i], dtype=np.uint8)

                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = img

            for i in range(len(results['occ2img'])):
                results['occ2img'][i] = ida_mat @ results['occ2img'][i]

        else:
            raise ValueError()

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        def crop_with_padding(image, box):
            x1, y1, x2, y2 = box
            out_h = y2 - y1
            out_w = x2 - x1
            channels = image.shape[2] if image.ndim == 3 else 1
            cropped = np.zeros((out_h, out_w, channels), dtype=image.dtype)

            src_x1 = max(x1, 0)
            src_y1 = max(y1, 0)
            src_x2 = min(x2, image.shape[1])
            src_y2 = min(y2, image.shape[0])

            if src_x1 < src_x2 and src_y1 < src_y2:
                dst_x1 = src_x1 - x1
                dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                cropped[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

            return cropped if image.ndim == 3 else cropped[..., 0]

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = cv2.resize(img, resize_dims, interpolation=cv2.INTER_LINEAR)
        img = crop_with_padding(img, crop)
        if flip:
            img = cv2.flip(img, 1)
        if rotate != 0:
            h, w = img.shape[:2]
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
            rot_mat = cv2.getRotationMatrix2D(center, rotate, 1.0)
            img = cv2.warpAffine(
                img,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])

        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b

        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return np.ascontiguousarray(img), ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if not self.test_mode:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


# Modified from BEVAug from https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/mmdet3d/datasets/pipelines/loading.py
@TRANSFORMS.register_module()
class RandomTransformOcc(BaseTransform):
    def __init__(self, bda_aug_conf):
        self.bda_aug_conf = bda_aug_conf

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        rotate_bda = np.random.uniform(*self.bda_aug_conf.get('rot_lim', (0, 0)))
        scale_bda = np.random.uniform(*self.bda_aug_conf.get('scale_lim', (1, 1)))
        flip_dx = np.random.uniform() < self.bda_aug_conf.get('flip_dx_ratio', 0)
        flip_dy = np.random.uniform() < self.bda_aug_conf.get('flip_dy_ratio', 0)
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = rotate_angle / 180 * np.pi
        rot_sin, rot_cos = np.sin(rotate_angle), np.cos(rotate_angle)

        # construct matrix
        rot_mat = np.array([[rot_cos, -rot_sin, 0],
                            [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
        scale_mat = np.array([[scale_ratio, 0, 0],
                              [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
        flip_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ np.array([
                [-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ np.array([
                [1, 0, 0], [0, -1, 0], [0, 0, 1]])

        return flip_mat @ scale_mat @ rot_mat

    def affine(self, block, angle, scale, fill=-1):
        raise NotImplementedError

    def transform(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        M = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)

        bda_mat = np.eye(4)
        bda_mat[:3, :3] = M

        semantics = results['voxel_semantics']
        meta = get_dataset_meta(results['dataset_type'], results['occ_variant'])
        if rotate_bda != 0 or scale_bda != 1.0:
            semantics = self.affine(
                semantics, rotate_bda, scale_bda, fill=meta.free_id)
        if flip_dx:
            semantics = semantics[::-1, ...]
        if flip_dy:
            semantics = semantics[:, ::-1, ...]

        results['voxel_semantics'] = semantics.copy()
        bda_inv = np.linalg.inv(bda_mat).astype(np.float32)
        results['occ2img'] = [
            (occ2img @ bda_inv).astype(np.float32) for occ2img in results['occ2img']
        ]

        if 'valid_mask' in results:
            valid_mask = results['valid_mask']
            if rotate_bda != 0 or scale_bda != 1.0:
                valid_mask = self.affine(valid_mask, rotate_bda, scale_bda, fill=False)
            if flip_dx:
                valid_mask = valid_mask[::-1, ...]
            if flip_dy:
                valid_mask = valid_mask[:, ::-1, ...]

            results['valid_mask'] = valid_mask.copy()

        if 'voxel_instances' in results:
            voxel_instances = results['voxel_instances']
            if rotate_bda != 0 or scale_bda != 1.0:
                voxel_instances = self.affine(
                    voxel_instances, rotate_bda, scale_bda, fill=255)
            if flip_dx:
                voxel_instances = voxel_instances[::-1, ...]
            if flip_dy:
                voxel_instances = voxel_instances[:, ::-1, ...]

            results['voxel_instances'] = voxel_instances.copy()

        return results
