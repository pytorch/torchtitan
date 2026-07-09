# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.package import PackageExporter

from torchtitan.components import fs
from torchtitan.components.torchpackage_checkpoint import (
    export_torch_package as export_recipe_torch_package,
    load_recipe_state,
    TorchPackageCheckpointManager,
)
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.model_for_inference import WorldModelForInference
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

PACKAGE_NAME = "model.torchpackage"
MODEL_CONFIG_FILE = "_torchpackage_model_config.pt"
STRUCTURED_LOG_DIR = os.getenv(
    "TORCHTITAN_STRUCTURED_LOG_DIR", "./outputs/worldmodel_torchpackage_checkpoint"
)
WORLD_MODEL_TORCH_PACKAGE_RECIPE = (
    "torchtitan.experiments.worldmodel.torchpackage_checkpoint:"
    "WorldModelTorchPackageRecipe"
)

TORCH_EXPORT_INTERN_MODULES = [
    "torchtitan.config.**",
    "torchtitan.distributed",
    "torchtitan.distributed.compile",
    "torchtitan.distributed.parallel_dims",
    "torchtitan.distributed.spmd_types",
    "torchtitan.distributed.utils",
    "torchtitan.experiments.worldmodel.model_for_inference",
    "torchtitan.experiments.worldmodel.model",
    "torchtitan.experiments.worldmodel.schedulers",
    "torchtitan.models.common.attention",
    "torchtitan.models.common.embedding",
    "torchtitan.models.common.nn_modules",
    "torchtitan.models.common.rope",
    "torchtitan.observability.**",
    "torchtitan.protocols.**",
    "torchtitan.tools.logging",
    "torchtitan.tools.utils",
]
TORCH_EXPORT_STRIP_FUTURE_ANNOTATIONS_MODULES = [
    "torchtitan.distributed.parallel_dims",
    "torchtitan.models.common.embedding",
    "torchtitan.protocols.module",
    "torchtitan.protocols.model_spec",
]
TORCH_EXPORT_EXTERN_MODULES = [
    "torch.**",
    "torchao.**",
    "numpy.**",
    "einops.**",
    "spmd_types.**",
    "typing_extensions.**",
    "tyro.**",
    "docstring_parser.**",
    "typeguard.**",
    "dataclasses.**",
    "collections.**",
    "argparse.**",
    "sys.**",
]
TORCH_EXPORT_DENY_MODULES = ["openpilot.**", "cereal", "cereal.**", "capnp", "capnp.**"]
TORCH_EXPORT_MOCK_MODULES = ["**"]
TORCH_EXPORT_CONFIG_INIT_SOURCE = """
import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

from .configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from .configurable import Configurable

__all__ = [
    "Configurable",
    "TORCH_DTYPE_MAP",
    "CompileConfig",
    "ParallelismConfig",
    "CommConfig",
    "TrainingConfig",
    "DebugConfig",
]
""".lstrip()


def build_meta_model(
    model_config: WorldModel.Config, *, dtype: torch.dtype = torch.bfloat16
) -> WorldModelForInference:
    with torch.device("meta"):
        return WorldModelForInference(model_config).to(dtype=dtype).eval()


def validate_model_config(state: Any) -> WorldModel.Config:
    if not isinstance(state, WorldModel.Config):
        raise TypeError(
            f"Worldmodel torch package state must be WorldModel.Config, "
            f"got {type(state).__name__}."
        )
    return state


def load_model_config(path: str) -> WorldModel.Config:
    return validate_model_config(load_recipe_state(path))


def convert_state_dict_to_fp8(
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8MMConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerTensor

    with torch.device("cpu"):
        model = WorldModelForInference(model_config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state_dict, strict=True, assign=True)
    state_dict.clear()
    del state_dict
    quantize_(
        model.blocks,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            mm_config=Float8MMConfig(),
        ),
    )
    converted = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    gc.collect()
    return converted


def build_package(
    *,
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
    step: int,
) -> bytes:

    with sl.log_trace_span("worldmodel_package_convert_fp8"):
        state_dict = convert_state_dict_to_fp8(model_config, state_dict)

    with sl.log_trace_span("worldmodel_package_model_io"):
        io_model_config = copy.deepcopy(model_config)
        if io_model_config.transformer.attention_impl == "FLEX":
            io_model_config.transformer.attention_impl = "SDPA"
        if io_model_config.plan_head.attention_impl == "FLEX":
            io_model_config.plan_head.attention_impl = "SDPA"
        io_model = build_meta_model(io_model_config)
        assert io_model.config.transformer.attention_mask != "NONE"
        num_prefill_frames = io_model.config.input_size[0] - 1
        model_io = io_model.get_model_io(
            dtype=torch.bfloat16,
            steps=1,
            num_prefill_frames=num_prefill_frames,
        )
        del io_model, io_model_config
        model = build_meta_model(model_config)

    with sl.log_trace_span("worldmodel_package_build"):
        package_buffer = io.BytesIO()
        with PackageExporter(package_buffer, debug=True) as exporter:
            exporter.intern(TORCH_EXPORT_INTERN_MODULES)
            exporter.extern(TORCH_EXPORT_EXTERN_MODULES)
            exporter.deny(TORCH_EXPORT_DENY_MODULES)
            exporter.save_source_string(
                "torchtitan.config",
                TORCH_EXPORT_CONFIG_INIT_SOURCE,
                is_package=True,
            )
            for module_name in TORCH_EXPORT_STRIP_FUTURE_ANNOTATIONS_MODULES:
                spec = importlib.util.find_spec(module_name)
                if spec is None or spec.origin is None:
                    raise ModuleNotFoundError(module_name)
                source = Path(spec.origin).read_text()
                source = source.replace("from __future__ import annotations\n\n", "")
                if module_name == "torchtitan.distributed.parallel_dims":
                    source = source.replace(
                        ") -> ParallelDims:\n", ') -> "ParallelDims":\n'
                    )
                exporter.save_source_string(module_name, source)
            exporter.mock(
                TORCH_EXPORT_MOCK_MODULES,
                exclude=TORCH_EXPORT_INTERN_MODULES
                + TORCH_EXPORT_EXTERN_MODULES
                + TORCH_EXPORT_DENY_MODULES,
            )
            exporter.save_pickle("model", "model.pkl", model)
            del model
            exporter.save_pickle(
                "meta", "meta.pkl", {"model_io": model_io, "step": step}
            )
            del model_io

            state_dict_buffer = io.BytesIO()
            torch.save(state_dict, state_dict_buffer)
            state_dict_bytes = state_dict_buffer.getvalue()
            del state_dict_buffer
            state_dict.clear()
            del state_dict
            gc.collect()
            exporter.save_binary("assets", "state_dict.pt", state_dict_bytes)
            del state_dict_bytes
        package = package_buffer.getvalue()
        del package_buffer
    sl.log_trace_scalar({"worldmodel_package.package_bytes": len(package)})
    return package


class WorldModelTorchPackageRecipe:
    def build_empty_state_dict(self, state: Any) -> dict[str, torch.Tensor]:
        model_config = validate_model_config(state)
        model = build_meta_model(model_config)
        try:
            return {
                name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                for name, tensor in model.state_dict().items()
            }
        finally:
            del model
            gc.collect()

    def build_package(
        self,
        *,
        state: Any,
        state_dict: dict[str, torch.Tensor],
        step: int,
    ) -> bytes:
        model_config = validate_model_config(state)
        return build_package(
            model_config=model_config,
            state_dict=state_dict,
            step=step,
        )


def export_torch_package(checkpoint_path: str) -> None:
    recipe_state_path = fs.join_path(checkpoint_path, MODEL_CONFIG_FILE)
    output_path = fs.join_path(checkpoint_path, PACKAGE_NAME)
    step = fs.basename(checkpoint_path)
    assert (
        step.isdigit()
    ), f"checkpoint path {checkpoint_path} does not end with a step number."
    model_config = load_model_config(recipe_state_path)
    try:
        export_recipe_torch_package(
            recipe=WorldModelTorchPackageRecipe(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            recipe_state=model_config,
            step=int(step),
            recipe_state_path=recipe_state_path,
        )
    finally:
        del model_config
        gc.collect()


class WorldModelTorchPackageCheckpointManager(TorchPackageCheckpointManager):
    """Worldmodel checkpoint manager configured with the worldmodel recipe."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchPackageCheckpointManager.Config):
        torch_package_recipe: str = WORLD_MODEL_TORCH_PACKAGE_RECIPE
        torch_package_file: str = PACKAGE_NAME
        torch_package_recipe_state_file: str = MODEL_CONFIG_FILE
        torch_package_structured_log_dir: str = STRUCTURED_LOG_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a worldmodel DCP checkpoint for inference."
    )
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()

    init_logger()
    sl.init_structured_logger(
        source="worldmodel_torchpackage_checkpoint",
        output_dir=STRUCTURED_LOG_DIR,
    )
    with sl.log_trace_span("worldmodel_package_total"):
        export_torch_package(args.checkpoint_path)


if __name__ == "__main__":
    main()
