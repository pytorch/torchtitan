# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.protocols import BaseModel


class FlexShardGraphTrainer(GraphTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config):
        pass

    def __init__(self, config: Config):
        compile_config = config.compile
        if compile_config.enable or compile_config.mode is not None:
            raise ValueError(
                "FlexShard currently supports eager execution only. "
                "Set --compile.mode None and leave --compile.enable false."
            )
        if compile_config.precompile_artifact_dir:
            raise ValueError("FlexShard does not support precompile artifacts yet.")
        if config.checkpoint.enable or config.checkpoint.initial_load_path is not None:
            raise ValueError("FlexShard checkpoint save/load is not supported yet.")
        super().__init__(config)

    def _get_model_build_device(self) -> str:
        """Build on CPU so FlexShard can shard real tensors before init."""
        return "cpu"

    def _init_model_weights(
        self,
        model: nn.Module,
        init_device: str,
        buffer_device: torch.device | None,
    ) -> None:
        """Initialize sharded params in-place and move non-param buffers."""
        has_flex_shard_state = any(
            getattr(module, "_dstorages", None) is not None
            for module in model.modules()
        )
        target_device = torch.device(init_device)
        if (
            has_flex_shard_state
            and buffer_device is None
            and target_device.type != "cpu"
        ):
            buffer_device = target_device

        with torch.no_grad():
            cast(BaseModel, model).init_weights(buffer_device=buffer_device)

        if target_device.type == "cpu":
            return

        if has_flex_shard_state:
            for module in model.modules():
                for name, buffer in list(module._buffers.items()):
                    if buffer is None or buffer.device.type == "meta":
                        continue
                    if buffer.device != target_device:
                        module._buffers[name] = buffer.to(target_device)
            return

        model.to(init_device)
