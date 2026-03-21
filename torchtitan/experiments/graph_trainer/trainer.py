# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from dataclasses import dataclass, field
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_utils import CompiledModule
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config: Config):
        if config.compile.fake_tensors:
            self._fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            original_to_empty = torch.nn.Module.to_empty
            fake_mode = self._fake_mode

            def to_empty_under_fake_mode(module, *, device, **kwargs):
                with fake_mode:
                    return original_to_empty(module, device=device, **kwargs)

            with (
                patch.object(torch.nn.Module, "to_empty", to_empty_under_fake_mode),
                patch.object(
                    CompiledModule, "init_weights", lambda self, **kwargs: None
                ),
            ):
                super().__init__(config)
        else:
            self._fake_mode = None
            super().__init__(config)

    def train(self):
        if self.config.compile.precompile and self._fake_mode is not None:
            self._precompile_with_fake_tensors()
        else:
            super().train()

    def _precompile_with_fake_tensors(self):
        """
        Trigger AOT compilation with fake tensors and save the artifact.
        We call the joint_graph_builder directly (not the full forward)
        because the compiled code cannot be executed with fake tensors
        (NCCL would try to communicate fake data). The on_compile
        callback in the builder automatically saves the artifact.
        """
        model = self.model_parts[0]

        data_iterator = self.batch_generator(self.dataloader)
        input_dict, labels = next(data_iterator)

        with self._fake_mode:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels
            )

        dt_args, dt_kwargs = model.parallelize_inputs(
            model.parallel_dims, (inputs,), {**extra_inputs, **extra_kwargs}
        )
        model.joint_graph_module = model.joint_graph_builder(
            model.inner, dt_args, dt_kwargs
        )

        logger.info(
            "Precompilation with fake tensors complete. "
            "Artifacts saved. Exiting — training cannot proceed "
            "with fake tensors."
        )

    def close(self) -> None:
        super().close()

        # Note [explicit cudagraph close]
        # cudagraph holds reference to nccl which prevents destroy nccl
        # group. so we need to explicitly delete cudagraph which is held
        # in joint_graph_module. An explicit gc.collect() is necessary
        # to clean up reference cycles.
        for part in self.model_parts:
            if hasattr(part, "joint_graph_module"):
                part.joint_graph_module = None
        gc.collect()
