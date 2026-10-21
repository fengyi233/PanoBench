from __future__ import annotations

import re

from mmengine.registry import LOG_PROCESSORS
from mmengine.runner.log_processor import LogProcessor

# ---------------------------------------------------------------------------
# ConciseLogProcessor（LOG_PROCESSORS）
# ---------------------------------------------------------------------------
# OPUS 子 loss 形如 `d0.loss_cls`、`d1.loss_pts`、`init_loss_pts`，需一并过滤。
_LOSS_COMPONENT_RE = re.compile(r"^(?:loss_|init_loss|d\d+\.(?:loss|init_loss))")


def _is_loss_component_field(name: str) -> bool:
    return bool(_LOSS_COMPONENT_RE.match(name)) and name != "loss"


def _drop_loss_component_segments(log_str: str) -> str:
    parts = []
    for segment in log_str.split("  "):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            parts.append(segment)
            continue
        field_name = segment.split(":", 1)[0].strip()
        if _is_loss_component_field(field_name):
            continue
        parts.append(segment)
    return "  ".join(parts)


@LOG_PROCESSORS.register_module()
class ConciseLogProcessor(LogProcessor):
    """训练时不在终端逐条打印 d0.loss_cls 等，总 loss 仍写 tag（TensorBoard 不受影响）。"""

    def get_log_after_iter(self, runner, batch_idx: int, mode: str):
        tag, log_str = super().get_log_after_iter(runner, batch_idx, mode)
        if mode != "train":
            return tag, log_str
        return tag, _drop_loss_component_segments(log_str)
