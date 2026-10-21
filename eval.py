"""PanoBench evaluation.

Example:
    python eval.py --config-name experiments/opusv1/panobench_carlaocc data.occ_variant=vs_0.1_surround_view checkpoint=work_dirs/latest.pth
"""

import warnings
from pathlib import Path

import hydra
import numpy as np
from mmengine.config import Config
from mmengine.dist import infer_launcher
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")


def _field(data, key: str):
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


def _optional_field(data, key: str):
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _meta_from_sample(sample) -> dict:
    if hasattr(sample, "metainfo"):
        return dict(sample.metainfo)
    if not isinstance(sample, dict):
        return {}
    meta = sample.get("metainfo") or sample.get("img_metas") or sample.get("img_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _pred_occ_from_sample(sample):
    if hasattr(sample, "pred_occ"):
        return sample.pred_occ
    if isinstance(sample, dict) and "pred_occ" in sample:
        return sample["pred_occ"]
    raise KeyError("Expected pred_occ in each validation output sample.")


def _sample_filename(sample, batch_idx: int, sample_idx: int) -> str:
    meta = _meta_from_sample(sample)
    filenames = meta.get("filename")
    if isinstance(filenames, (list, tuple)) and filenames:
        stem = Path(filenames[0]).stem
    elif isinstance(filenames, str):
        stem = Path(filenames).stem
    else:
        stem = str(meta.get("frame_token") or meta.get("token") or f"{batch_idx:06d}_{sample_idx:02d}")

    token = meta.get("frame_token") or meta.get("token")
    if token is not None and str(token) not in stem:
        stem = f"{stem}_{token}"
    return stem


class PredOccSaveHook(Hook):
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if outputs is None:
            return
        samples = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        for sample_idx, sample in enumerate(samples):
            pred_occ = _pred_occ_from_sample(sample)
            arrays = {
                "occ_loc": _to_numpy(_field(pred_occ, "occ_loc")),
                "semantics": _to_numpy(_field(pred_occ, "semantics")),
            }
            for key in ("pano_sem", "pano_inst"):
                value = _optional_field(pred_occ, key)
                if value is not None:
                    arrays[key] = _to_numpy(value)

            stem = _sample_filename(sample, batch_idx, sample_idx)
            save_path = self.save_dir / f"{stem}.npz"
            if save_path.exists():
                save_path = self.save_dir / f"{stem}_{batch_idx:06d}_{sample_idx:02d}.npz"
            np.savez_compressed(save_path, **arrays)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="experiments/opusv1/panobench_carlaocc",
)
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Set checkpoint=/path/to/checkpoint.pth for evaluation.")
    pred_occ_save_dir = cfg.get("pred_occ_save_dir")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("data", None)
    cfg_dict.pop("checkpoint", None)

    ckpt_path = Path(checkpoint)
    if cfg_dict.get("work_dir") is None:
        cfg_dict["work_dir"] = str(ckpt_path.parent / f"{ckpt_path.stem}_eval")
    cfg_dict["load_from"] = checkpoint
    cfg_dict["launcher"] = infer_launcher()
    if cfg_dict.get("custom_imports") is not None:
        import_modules_from_strings(**cfg_dict["custom_imports"])

    runner = Runner.from_cfg(Config(cfg_dict))
    if pred_occ_save_dir is not None:
        runner.register_hook(PredOccSaveHook(pred_occ_save_dir))
    runner.val()


if __name__ == "__main__":
    main()
