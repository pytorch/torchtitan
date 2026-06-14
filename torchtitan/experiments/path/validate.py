from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.report_runner import ReportRunner, ReportSpec
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.components.validate import BaseValidator
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from xx.common.helpers import parse_info
from xx.release_tests.lib.base_report import ReportFormat
from xx.training.path.test import (
    DATASET_REPORTS,
    MODEL_REPORTS,
)

from .loss import PathLoss

ValidationContext = Callable[[], AbstractContextManager[None]]


def segment_names_from_info(info: torch.Tensor) -> list[str]:
    return [parse_info(x)["name"] for x in info.cpu().numpy()]


def global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class PathValidator(BaseValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool
        steps: int
        dataloader: BaseDataLoader.Config
        mixed_precision_param: str
        reports: dict[str, list[int]] = field(default_factory=dict)
        miniray: dict[str, Any] = field(default_factory=dict)

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
        self.miniray = dict(self.config.miniray)
        self.dataloader = self.config.dataloader.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
            validation_steps=self.config.steps,
        )
        #TODO centralize training_id
        self.training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{self.training_id}:path:validation")
        self.report_runner = ReportRunner(metrics_processor=self.metrics_processor, enabled=global_rank() == 0)

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        for model in model_parts:
            model.eval()

        device = next(model_parts[0].parameters()).device
        try:
            total_loss = torch.zeros((), device=device)
            total_samples = torch.zeros((), device=device)
            metric_sums: dict[str, torch.Tensor] = {}
            validation_segment_names: set[str] = set()
            batch_mesh = self.parallel_dims.get_optional_mesh("batch")
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            for num_steps, (input_dict, targets) in enumerate(self.dataloader):
                if self.config.steps != -1 and num_steps >= self.config.steps:
                    break
                self.metrics_processor.ntokens_since_last_log += next(iter(input_dict.values())).shape[0]
                if "info" in input_dict:
                    validation_segment_names.update(segment_names_from_info(input_dict["info"]))
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                targets = {k: v.to(device) for k, v in targets.items()}
                local_samples = torch.tensor(next(iter(input_dict.values())).shape[0], dtype=torch.float32, device=device)
                global_samples = (
                    dist_utils.dist_sum(local_samples, batch_mesh)
                    if batch_mesh is not None
                    else local_samples
                )

                with self.validation_context():
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

            self.unique_segment_counter.update(validation_segment_names)

            samples = torch.as_tensor(total_samples, dtype=torch.float32, device=device)
            loss = float((torch.as_tensor(total_loss, dtype=torch.float32, device=device) / samples).item())
            extra_metrics = {
                f"validation_metrics/path/{k}": float((torch.as_tensor(v, dtype=torch.float32, device=device) / samples).item())
                for k, v in metric_sums.items()
            }
            extra_metrics["validation_metrics/dataset/unique_segments_seen"] = (
                self.unique_segment_counter.global_count(batch_mesh.get_group())
                if batch_mesh is not None
                else self.unique_segment_counter.local_count()
            )
            self.metrics_processor.log_validation(loss=loss, step=step, extra_metrics=extra_metrics)
            self._submit_reports(step)
        finally:
            for model in model_parts:
                model.train()

    def close(self) -> None:
        try:
            self.report_runner.close()
        finally:
            self.dataloader.close()

    def _submit_reports(self, step: int) -> None:
        def _run_report(TestCls: type, test_config: Any) -> tuple[Any, ...]:
            return (TestCls(test_config).run_report(),)

        dataloader_config = self.config.dataloader
        report_specs: dict[str, ReportSpec] = {}
        for report_name, (TestCls, ReportConfigCls) in MODEL_REPORTS.items():
            report_config = ReportConfigCls(
                rollout={'agent': {'supercombo': f'{self.training_id}/{step}', 'model_trained_fps': dataloader_config.fps}},
                report_name=f'path_{report_name}',
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs[report_name] = ReportSpec(
                output_names=(report_name,),
                output_types=('html',),
                steps=self.config.reports.get(report_name, []),
                func=_run_report,
                arguments=[TestCls, report_config],
            )

        for report_name, (TestCls, ReportConfigCls) in DATASET_REPORTS.items():
            report_config = ReportConfigCls(
                route_list=dataloader_config.dataset,
                pipeline_dir=dataloader_config.pipeline_dir,
                report_name=f'path_{report_name}',
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs[report_name] = ReportSpec(
                output_names=(report_name,),
                output_types=('html',),
                steps=self.config.reports.get(report_name, []),
                func=_run_report,
                arguments=[TestCls, report_config],
            )

        self.report_runner.submit_due(step=step, report_specs=report_specs)
