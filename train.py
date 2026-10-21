"""Hydra entrypoint for PanoBench training.

Examples:
    python train.py --config-name experiments/opusv1/panobench_carlaocc data.occ_variant=vs_0.1_surround_view
    torchrun --nproc_per_node=8 train.py --config-name experiments/opusv1/panobench_carlaocc data.occ_variant=vs_0.1_surround_view
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from mmengine.config import Config
from mmengine.dist import infer_launcher
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="experiments/opusv1/panobench_carlaocc",
)
def main(cfg: DictConfig):
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_dict.pop("data", {})
    cfg_dict.pop("checkpoint", None)

    cfg_name = HydraConfig.get().job.config_name
    variant = data_cfg.get("occ_variant")
    variant_name = "default" if variant is None else str(variant)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dict["work_dir"] = str(Path("outputs") / cfg_name / variant_name / timestamp)

    cfg_dict["launcher"] = infer_launcher()
    if cfg_dict.get("custom_imports") is not None:
        import_modules_from_strings(**cfg_dict["custom_imports"])
    runner = Runner.from_cfg(Config(cfg_dict))
    runner.train()


if __name__ == "__main__":
    main()
