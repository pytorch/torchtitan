# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from torchtitan.components import fs
from torchtitan.components.checkpoint import CheckpointManager, MODEL
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module


OnnxInputDType = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
]

_ONNX_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_ARTIFACT_BARRIER_TIMEOUT = timedelta(minutes=5)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class OnnxCheckpointManager(CheckpointManager):
    """Checkpoint manager that can write rank-0 model artifacts alongside DCP."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        checkpoint_id_format: str = ""

        save_model_state_dict: bool = False
        """Save a gathered model state dict as one rank-0 torch file."""

        model_state_dict_file: str = "model_state_dict.pt"
        """File name for the gathered model state dict."""

        export_onnx: bool = False
        """Export a rank-0 ONNX model artifact alongside written checkpoints."""

        onnx_file: str = "model.onnx"
        """File name for the ONNX artifact."""

        input_names: list[str] = field(default_factory=list)
        """Input names passed to torch.onnx.export."""

        output_names: list[str] = field(default_factory=list)
        """Output names passed to torch.onnx.export."""

        dynamic_axes: dict[str, dict[int, str]] | None = None
        """Dynamic axes passed to torch.onnx.export."""

        input_shapes: list[list[int]] = field(default_factory=list)
        """Dummy input shapes used for ONNX export."""

        input_dtypes: list[OnnxInputDType] = field(default_factory=list)
        """Dummy input dtypes used for ONNX export."""

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.save_model_state_dict = config.save_model_state_dict
        self.model_state_dict_file = config.model_state_dict_file
        self.export_onnx = config.export_onnx
        self.onnx_file = config.onnx_file
        self.input_names = config.input_names
        self.output_names = config.output_names
        self.dynamic_axes = config.dynamic_axes
        self.input_shapes = config.input_shapes
        self.input_dtypes = config.input_dtypes

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        saved = super().save(curr_step, last_step)
        if saved and (self.save_model_state_dict or self.export_onnx):
            self._async_wait()
            self._save_rank0_artifacts(curr_step)
            self._wait_for_rank0()
        return saved

    def _wait_for_rank0(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        if dist.get_world_size() == 1:
            return

        barrier_group = dist.new_group(timeout=_ARTIFACT_BARRIER_TIMEOUT)
        try:
            dist.barrier(
                group=barrier_group,
                device_ids=[device_module.current_device()],
            )
        finally:
            dist.destroy_process_group(barrier_group)

    def _save_rank0_artifacts(self, curr_step: int) -> None:
        model_parts = self.states[MODEL].model
        if len(model_parts) != 1:
            raise ValueError("ONNX checkpoint artifacts do not support PP")

        state_dict = get_model_state_dict(
            model_parts[0],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if _rank() != 0:
            return

        checkpoint_id = self._create_checkpoint_id(curr_step)
        if self.save_model_state_dict:
            path = fs.join_path(checkpoint_id, self.model_state_dict_file)
            with fs.open_file(path, "wb") as f:
                torch.save(state_dict, f)
            logger.info("Saved gathered model state dict to %s", path)

        if self.export_onnx:
            model = self._build_cpu_model_from_state_dict(model_parts[0], state_dict)
            path = fs.join_path(checkpoint_id, self.onnx_file)
            self._export_onnx(model, path)
            logger.info("Exported ONNX model to %s", path)

    @staticmethod
    def _build_cpu_model_from_state_dict(
        source_model: nn.Module,
        state_dict: dict[str, Any],
    ) -> nn.Module:
        # The live training model may be FSDP-wrapped, compiled, or sharded. Export
        # a plain CPU model from the gathered state dict instead.
        model_config = getattr(source_model, "config", None)
        if model_config is None or not hasattr(model_config, "build"):
            raise ValueError(
                "ONNX export requires the model to expose a TorchTitan config "
                "with a build() method."
            )
        with torch.device("cpu"):
            model = model_config.build()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _export_onnx(self, model: nn.Module, path: str) -> None:
        inputs = self._build_onnx_inputs()
        input_names = self.input_names or None
        output_names = self.output_names or None
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            tuple(inputs),
            cast(Any, buffer),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=self._dynamic_axes(),
            external_data=False,
        )
        with fs.open_file(path, "wb") as f:
            f.write(buffer.getvalue())

    def _dynamic_axes(self) -> dict[str, dict[int, str]] | None:
        if self.dynamic_axes is not None:
            return self.dynamic_axes
        names = [*self.input_names, *self.output_names]
        return {name: {0: "b"} for name in names} if names else None

    def _build_onnx_inputs(self) -> list[torch.Tensor]:
        if not self.input_shapes:
            raise ValueError("checkpoint.input_shapes must be set for ONNX export")
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                "checkpoint.input_shapes and checkpoint.input_dtypes must have "
                f"the same length, got {len(self.input_shapes)} and "
                f"{len(self.input_dtypes)}."
            )
        if self.input_names and len(self.input_names) != len(self.input_shapes):
            raise ValueError(
                "checkpoint.input_names and checkpoint.input_shapes must have "
                f"the same length, got {len(self.input_names)} and "
                f"{len(self.input_shapes)}."
            )

        return [
            torch.zeros(shape, dtype=_ONNX_DTYPE_MAP[dtype])
            for shape, dtype in zip(self.input_shapes, self.input_dtypes, strict=True)
        ]
