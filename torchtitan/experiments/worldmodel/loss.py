from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig


WorldModelLossFunction = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]],
    tuple[torch.Tensor, dict[str, torch.Tensor]],
]


def laplacian_density_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    std_clamp: float = 1e-3,
    loss_clamp: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_values = y_pred.shape[-1] // 2
    mu_true = y_true[..., :n_values]
    mu_pred = y_pred[..., :n_values]
    mask = ~torch.isnan(mu_true)
    mu_true = mu_true.masked_fill(~mask, 0).detach()
    log_sigma_raw = y_pred[..., n_values:]
    err = torch.abs(mu_true - mu_pred)
    log_sigma_min = torch.clamp(log_sigma_raw, min=math.log(std_clamp))
    log_sigma = torch.max(log_sigma_raw, torch.log(1e-6 + err / loss_clamp))
    return mask * (err * torch.exp(-log_sigma) + log_sigma_min), err, mask


def compute_worldmodel_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    plan_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss: torch.Tensor | None = None
    terms: dict[str, torch.Tensor] = {}

    if "sample" in outputs:
        pred = outputs["sample"]
        target = targets["v"].to(device=pred.device, dtype=pred.dtype)
        mask = targets["mask"].to(device=pred.device).flatten(1).float()
        mse = F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1)
        diffusion_loss = (mse * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = diffusion_loss
        terms["diffusion_loss"] = diffusion_loss.detach()

    if "plan" in outputs and "plan" in targets:
        pred = outputs["plan"]
        target = targets["plan"].to(device=pred.device, dtype=pred.dtype)
        plan_loss_values, plan_err, plan_mask = laplacian_density_loss(target.float(), pred.float())
        plan_loss = plan_loss_values.flatten(1).mean(dim=1)
        flat_mask = plan_mask.flatten(1).float()
        plan_mse = (plan_err.square().flatten(1) * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1.0)
        weighted_plan_loss = (plan_loss_weight if "sample" in outputs else 1.0) * plan_loss
        loss = weighted_plan_loss if loss is None else loss + weighted_plan_loss
        terms["plan_loss"] = plan_loss.detach()
        terms["plan_mse"] = plan_mse.detach()

    if loss is None:
        raise RuntimeError("worldmodel produced no trainable outputs")
    terms["loss"] = loss.detach()
    return loss, terms


class WorldModelLoss(BaseLoss):
    fn: WorldModelLossFunction

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        plan_loss_weight: float

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        plan_loss_weight = config.plan_loss_weight

        def loss_fn(
            outputs: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return compute_worldmodel_losses(
                outputs,
                targets,
                plan_loss_weight=plan_loss_weight,
            )

        self.fn = loss_fn
        self._maybe_compile(compile_config)

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        global_valid_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del global_valid_tokens
        return self.fn(outputs, targets)
