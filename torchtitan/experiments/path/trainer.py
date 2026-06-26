from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .loss import PathLoss
from .onnx_checkpoint import PathOnnxCheckpointManager
from .validate import PathValidator, segment_names_from_info


class PathTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        loss: PathLoss.Config
        validator: PathValidator.Config
        checkpoint: PathOnnxCheckpointManager.Config
        miniray: dict[str, Any] = field(default_factory=dict)
        fps: int

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.codedir:
                self.miniray = {**self.miniray, "codedir": self.codedir}
                self.validator.miniray = {**self.validator.miniray, "codedir": self.codedir}

    def __init__(self, config: Config):
        super().__init__(config)
        training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{training_id}:path:train")
        self.loss_fn.to(self.device)

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                input_dict, targets = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.metrics_processor.ntokens_since_last_log += next(iter(input_dict.values())).shape[0]
            self.metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)
            yield input_dict, targets

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.ntokens_seen += next(iter(input_dict.values())).shape[0]
        return input_dict, labels

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        local_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, labels = self.post_dataloading_process(input_dict, labels)
        assert len(self.model_parts) == 1
        with self.train_context():
            pred = self.model_parts[0](inputs)
            loss_vec, metrics = self.loss_fn(pred, labels)
            loss = loss_vec.sum() / local_samples
            del pred
            loss.backward()
        return loss, metrics

    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> None:
        self.optimizers.zero_grad()
        lr_metrics = self.lr_schedulers.get_metrics()
        parallel_dims = self.parallel_dims
        batch_mesh = parallel_dims.get_optional_mesh("batch")

        microbatches = []
        step_segment_names: set[str] = set()
        local_samples = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, targets = next(data_iterator)
                local_samples += next(iter(input_dict.values())).shape[0]
                if "info" in input_dict:
                    step_segment_names.update(segment_names_from_info(input_dict["info"]))
                microbatches.append((input_dict, targets))
        sl.log_trace_scalar({"local_samples": int(local_samples)})

        local_samples = local_samples.to(self.device)
        if batch_mesh is not None:
            global_samples = dist_utils.dist_sum(local_samples, batch_mesh)
        else:
            global_samples = local_samples.float()
        global_samples = torch.as_tensor(global_samples, dtype=torch.float32, device=self.device)
        global_samples_value = float(global_samples.item())

        accumulated_losses = []
        metric_sums: dict[str, torch.Tensor] = {}
        for input_dict, targets in microbatches:
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            loss, metrics = self.forward_backward_step(
                input_dict=input_dict,
                labels=targets,
                local_samples=local_samples,
            )
            accumulated_losses.append(loss.detach())
            for name, value in metrics.items():
                if name == "loss":
                    continue
                metric_sums[name] = metric_sums.get(name, torch.zeros((), device=self.device)) + value.float().sum()

        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        self.unique_segment_counter.update(step_segment_names)

        loss = torch.sum(torch.stack(accumulated_losses))
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_loss_sum = loss * local_samples
            global_avg_loss, global_max_loss, global_samples_seen = (
                dist_utils.dist_sum(local_loss_sum.detach(), loss_mesh) / global_samples_value,
                dist_utils.dist_max(loss.detach(), loss_mesh),
                dist_utils.dist_sum(torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device), loss_mesh),
            )
            metric_sums = {k: dist_utils.dist_sum(v, loss_mesh) for k, v in metric_sums.items()}
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_samples_seen = self.ntokens_seen

        path_metrics = {
            f"path/{k}": float((torch.as_tensor(v, dtype=torch.float32, device=self.device) / global_samples).item())
            for k, v in metric_sums.items()
        }
        unique_segments_seen = (
            self.unique_segment_counter.global_count(batch_mesh.get_group())
            if batch_mesh is not None
            else self.unique_segment_counter.local_count()
        )
        dataset_metrics = {
            "dataset/unique_segments_seen": unique_segments_seen,
        }
        extra_metrics = {"n_samples_seen": global_samples_seen, **lr_metrics, **path_metrics, **dataset_metrics}
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def close(self) -> None:
        self.dataloader.close()
        if self.config.validator.enable:
            self.validator.close()
        super().close()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["unique_segment_counter"] = self.unique_segment_counter.state_dict()
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None:
            state["validation_unique_segment_counter"] = validator_unique_segment_counter.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "unique_segment_counter" in state_dict:
            self.unique_segment_counter.load_state_dict(state_dict["unique_segment_counter"])
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None and "validation_unique_segment_counter" in state_dict:
            validator_unique_segment_counter.load_state_dict(state_dict["validation_unique_segment_counter"])
