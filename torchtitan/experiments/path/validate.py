from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import BaseValidator
from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils

from .loss import PathLoss

ValidationContext = Callable[[], AbstractContextManager[None]]


class PathValidator(BaseValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool
        steps: int
        dataloader: BaseDataLoader.Config
        mixed_precision_param: str

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: PathLoss,
        validation_context: ValidationContext,
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
        **kwargs: Any,
    ) -> None:
        del parallelism, kwargs
        super().__init__(config=config)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.dataloader = self.config.dataloader.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
            validation_steps=self.config.steps,
        )

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        for model in model_parts:
            model.eval()

        device = next(model_parts[0].parameters()).device
        amp_dtype = TORCH_DTYPE_MAP[self.config.mixed_precision_param]
        amp_enabled = amp_dtype != torch.float32 and not self.parallel_dims.fsdp_enabled

        try:
            total_loss = torch.zeros((), device=device)
            total_samples = torch.zeros((), device=device)
            metric_sums: dict[str, torch.Tensor] = {}
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            for num_steps, (input_dict, targets) in enumerate(self.dataloader):
                if self.config.steps != -1 and num_steps >= self.config.steps:
                    break
                self.metrics_processor.ntokens_since_last_log += next(iter(input_dict.values())).shape[0]
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                targets = {k: v.to(device) for k, v in targets.items()}
                local_samples = torch.tensor(next(iter(input_dict.values())).shape[0], dtype=torch.float32, device=device)
                global_samples = (
                    dist_utils.dist_sum(local_samples, self.parallel_dims.get_mesh("batch"))
                    if self.parallel_dims.dp_enabled
                    else local_samples
                )

                with self.validation_context(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    pred = model_parts[0](input_dict)
                    loss_vec, metrics = self.loss_fn(pred, targets)

                loss_sum = loss_vec.float().sum()
                batch_metric_sums = {k: v.float().sum() for k, v in metrics.items() if k != "loss"}
                if self.parallel_dims.dp_cp_enabled:
                    loss_sum = dist_utils.dist_sum(loss_sum, loss_mesh)
                    batch_metric_sums = {k: dist_utils.dist_sum(v, loss_mesh) for k, v in batch_metric_sums.items()}
                total_loss += loss_sum
                total_samples += global_samples
                for name, value in batch_metric_sums.items():
                    metric_sums[name] = metric_sums.get(name, torch.zeros((), device=device)) + value

            samples = torch.as_tensor(total_samples, dtype=torch.float32, device=device)
            loss = float((torch.as_tensor(total_loss, dtype=torch.float32, device=device) / samples).item())
            extra_metrics = {
                f"validation_metrics/path/{k}": float((torch.as_tensor(v, dtype=torch.float32, device=device) / samples).item())
                for k, v in metric_sums.items()
            }
            self.metrics_processor.log_validation(loss=loss, step=step, extra_metrics=extra_metrics)
        finally:
            for model in model_parts:
                model.train()

    def close(self) -> None:
        self.dataloader.close()
