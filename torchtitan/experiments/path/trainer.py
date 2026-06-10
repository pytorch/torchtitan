from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.onnx_checkpoint import OnnxCheckpointManager
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .loss import PathLoss
from .validate import PathValidator


class PathTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        loss: PathLoss.Config
        validator: PathValidator.Config
        checkpoint: OnnxCheckpointManager.Config

    def __init__(self, config: Config):
        super().__init__(config)
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
        global_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, labels = self.post_dataloading_process(input_dict, labels)
        assert len(self.model_parts) == 1
        amp_dtype = TORCH_DTYPE_MAP[self.config.training.mixed_precision_param]
        amp_enabled = amp_dtype != torch.float32 and not self.parallel_dims.fsdp_enabled
        with self.train_context(), torch.autocast(self.device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred = self.model_parts[0](inputs)
            loss_vec, metrics = self.loss_fn(pred, labels)
            loss = loss_vec.sum() / global_samples
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

        microbatches = []
        local_samples = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, targets = next(data_iterator)
                local_samples += next(iter(input_dict.values())).shape[0]
                microbatches.append((input_dict, targets))
        sl.log_trace_scalar({"local_samples": int(local_samples)})

        local_samples = local_samples.to(self.device)
        if parallel_dims.dp_enabled:
            global_samples = dist_utils.dist_sum(local_samples, parallel_dims.get_mesh("batch"))
        else:
            global_samples = local_samples.float()

        accumulated_losses = []
        metric_sums: dict[str, torch.Tensor] = {}
        for input_dict, targets in microbatches:
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            loss, metrics = self.forward_backward_step(
                input_dict=input_dict,
                labels=targets,
                global_samples=global_samples,
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

        loss = torch.sum(torch.stack(accumulated_losses))
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_avg_loss = loss * global_samples / local_samples
            global_avg_loss, global_max_loss, global_samples_seen = (
                dist_utils.dist_sum(loss.detach(), loss_mesh),
                dist_utils.dist_max(local_avg_loss.detach(), loss_mesh),
                dist_utils.dist_sum(torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device), loss_mesh),
            )
            metric_sums = {k: dist_utils.dist_sum(v, loss_mesh) for k, v in metric_sums.items()}
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_samples_seen = self.ntokens_seen

        path_metrics = {f"path/{k}": float((v / global_samples).detach().item()) for k, v in metric_sums.items()}
        extra_metrics = {"n_samples_seen": global_samples_seen, **lr_metrics, **path_metrics}
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
