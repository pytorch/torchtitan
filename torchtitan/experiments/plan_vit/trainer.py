# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin trainer for plan_vit: model(inputs_dict) -> pred dict, PathLoss(pred, targets), backward.

Mirrors PathTrainer's generic core without the path-specific ONNX export / driving validator / reports a
scaling study doesn't need. loss + dataloader come from the base Trainer.Config (set in config_registry).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer


class PlanViTTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: "PlanViTTrainer.Config"):
        self.ntokens_seen = 0
        self._metrics: dict[str, torch.Tensor] = {}
        super().__init__(config)
        self.loss_fn.to(self.device)

    def batch_generator(
        self,
        data_iterable: Iterable[
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        ],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        data_iterator = iter(data_iterable)
        while True:
            t0 = time.perf_counter()
            try:
                input_dict, targets = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.metrics_processor.ntokens_since_last_log += next(
                iter(input_dict.values())
            ).shape[0]
            self.metrics_processor.data_loading_times.append(time.perf_counter() - t0)
            yield input_dict, targets

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self, *, input_dict: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert len(self.model_parts) == 1
        with self.train_context():
            pred = self.model_parts[0](input_dict)
            # plan_vit is single-frame: it predicts the current (last) frame's plan, so supervise
            # against the last temporal position of the dense target (path trains all positions).
            labels = {**labels, "plan": labels["plan"][:, -1]}
            loss_vec, metrics = self.loss_fn(pred, labels)
            loss = loss_vec.mean()
            self._metrics = metrics
            loss.backward()
        return loss

    def train_step(
        self,
        data_iterator: Iterator[
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        ],
    ) -> None:
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        input_dict, targets = next(data_iterator)
        input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
        targets = {k: v.to(self.device) for k, v in targets.items()}
        self.ntokens_seen += next(iter(input_dict.values())).shape[0]
        loss = self.forward_backward_step(input_dict=input_dict, labels=targets)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        local_loss = loss.detach()
        if self.parallel_dims.dp_cp_enabled:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_mean(local_loss, loss_mesh)
            global_max_loss = dist_utils.dist_max(local_loss, loss_mesh)
        else:
            global_avg_loss = global_max_loss = float(local_loss.item())

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics={"metrics/lr/": lr},
        )
