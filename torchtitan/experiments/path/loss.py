from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import CompileConfig
from torchtitan.components.loss import BaseLoss
from torchtitan.tools.logging import logger
from xx.training.lib.driving import DrivingLoss, DrivingMetric


class PathLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        del config
        self.fn = None
        self.loss_fn = DrivingLoss()
        self.metric_fn = DrivingMetric()
        if compile_config is not None and compile_config.enable and "loss" in compile_config.components:
            logger.info("Compiling the path loss and metric functions with torch.compile")
            self.loss_fn = torch.compile(self.loss_fn, backend=compile_config.backend)
            self.metric_fn = torch.compile(self.metric_fn, backend=compile_config.backend)

    def to(self, device: torch.device) -> PathLoss:
        self.loss_fn.to(device)
        self.metric_fn.to(device)
        return self

    def __call__(
        self,
        pred: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        global_valid_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del global_valid_tokens
        pred = {k: v.float() if v.is_floating_point() else v for k, v in pred.items()}
        loss, losses = self.loss_fn(pred, targets)
        return loss, losses | self.metric_fn(pred, targets)
