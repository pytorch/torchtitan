# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Knowledge distillation trainer.

Trains a student model to match a frozen teacher model's output distribution
while also learning from hard labels. The teacher is loaded from a checkpoint
and kept frozen (no gradients, eval mode).

For quantization-aware distillation, apply fake quantization to the student
via ``post_model_init_fn`` -- the teacher stays bf16.
"""

import os
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.kd.loss import kd_loss
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class KDTrainer(Trainer):
    """Trainer that distills knowledge from a frozen teacher to a student.

    Extends :class:`~torchtitan.trainer.Trainer` by:

    1. Building a second (teacher) model from the same ``model_spec``,
       frozen and in eval mode.
    2. Overriding ``forward_backward_step`` to run both models on the
       same input and compute the KD loss.

    The student model is ``self.model_parts`` (the standard Trainer model).
    The teacher model is ``self.teacher_parts``.

    Pipeline parallelism is not supported -- both models must fit on the
    same set of GPUs with DP/TP/EP parallelism only.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        temperature: float = 2.0
        """Softmax temperature for soft targets in KD loss."""

        alpha: float = 0.5
        """Weight for distillation loss. 1.0 = pure KD, 0.0 = pure CE."""

        def __post_init__(self):
            if self.parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Knowledge distillation does not support pipeline parallelism."
                )
            if isinstance(self.loss, ChunkedLossWrapper.Config):
                raise ValueError(
                    "ChunkedLossWrapper is not compatible with knowledge distillation. "
                    "Use CrossEntropyLoss instead (loss=CrossEntropyLoss.Config())."
                )

    teacher_parts: list[nn.Module]

    def __init__(self, config: Config):
        super().__init__(config)

        assert config.model_spec is not None
        model_spec = config.model_spec

        logger.info("Building teacher model for knowledge distillation")
        teacher = self._build_teacher(model_spec, config)
        self.teacher_parts = [teacher]

        teacher_param_count = sum(p.numel() for p in teacher.parameters())
        logger.info(f"Teacher model size: {teacher_param_count:,} parameters (frozen)")

    def _build_teacher(
        self, model_spec: ModelSpec, config: "KDTrainer.Config"
    ) -> nn.Module:
        """Build, parallelize, and freeze the teacher model.

        The teacher is initialized from the same ``hf_assets_path`` as the
        student. For quantization-aware distillation, the student gets fake
        quant applied via ``post_model_init_fn`` while the teacher stays bf16.
        """
        model_config = model_spec.model
        device_type = utils.device_type

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            teacher = model_config.build()

        teacher = model_spec.parallelize_fn(
            teacher,
            parallel_dims=self.parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )

        if config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        teacher.to_empty(device=init_device)
        with torch.no_grad():
            cast(BaseModel, teacher).init_weights(buffer_device=buffer_device)

        # Load HF checkpoint weights if available (mirrors CheckpointManager logic)
        checkpoint_path = config.checkpoint.initial_load_path
        if checkpoint_path is None and config.checkpoint.initial_load_in_hf:
            checkpoint_path = config.hf_assets_path
        if (
            config.checkpoint.initial_load_in_hf
            and checkpoint_path
            and os.path.isdir(checkpoint_path)
            and model_spec.state_dict_adapter
        ):
            self._load_teacher_from_hf(
                teacher, model_spec, model_config, config, checkpoint_path
            )

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        return teacher

    def _load_teacher_from_hf(
        self,
        teacher: nn.Module,
        model_spec: ModelSpec,
        model_config,
        config: "KDTrainer.Config",
        checkpoint_path: str,
    ) -> None:
        """Load HF safetensors weights into the teacher model."""
        sd_adapter = model_spec.state_dict_adapter(model_config, config.hf_assets_path)
        wrapped = ModelWrapper(teacher)
        state_dict = wrapped.state_dict()

        hf_state_dict = sd_adapter.to_hf(state_dict)
        hf_storage_reader = sd_adapter.get_hf_storage_reader(checkpoint_path, False)
        dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

        state_dict = sd_adapter.from_hf(hf_state_dict)
        wrapped.load_state_dict(state_dict)
        logger.info(f"Loaded teacher weights from HF checkpoint: {checkpoint_path}")

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float,
    ) -> torch.Tensor:
        config = self.config

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)

        assert len(self.model_parts) == 1
        assert len(self.teacher_parts) == 1
        student = self.model_parts[0]
        teacher = self.teacher_parts[0]

        # KD computes the KL over the full vocab, so it needs unsharded logits.
        # It is only validated without tensor/loss parallelism (TP=1), where the
        # model outputs are already plain tensors; the DTensor branches below are
        # a no-op there and gather the logits otherwise.
        with self.train_context():
            with torch.no_grad():
                teacher_pred = teacher(inputs, **extra_kwargs)
                if isinstance(teacher_pred, DTensor):
                    teacher_pred = teacher_pred.full_tensor()

            student_pred = student(inputs, **extra_kwargs)
            if isinstance(student_pred, DTensor):
                student_pred = student_pred.full_tensor()

            loss = kd_loss(
                student_logits=student_pred,
                teacher_logits=teacher_pred,
                labels=labels,
                temperature=config.temperature,
                alpha=config.alpha,
            )
            loss = loss / global_valid_tokens

            del student_pred, teacher_pred
            loss.backward()

        return loss
