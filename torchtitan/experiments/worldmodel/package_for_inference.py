from __future__ import annotations

import argparse
import copy
import gc
import io
import os
import posixpath
from typing import Any

import fsspec
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.package import PackageExporter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger

from torchtitan.experiments.worldmodel.model_config import model_registry
from torchtitan.experiments.worldmodel.model_for_inference import WorldModelForInference
from torchtitan.experiments.worldmodel.model import WorldModel


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

REPORTERV2_HOST = os.getenv("REPORTERV2_HOST", "mkv://data-gen.comma.life:3080/reporterv2")
STRUCTURED_LOG_DIR = os.getenv("TORCHTITAN_STRUCTURED_LOG_DIR", "./outputs/worldmodel_package_for_inference")
PACKAGE_NAME = "model.torchpackage"
INFERENCE_STEPS = 1

TORCH_EXPORT_INTERN_MODULES = [
    "torchtitan.config.**",
    "torchtitan.experiments.worldmodel.model_for_inference",
    "torchtitan.experiments.worldmodel.model",
    "torchtitan.experiments.worldmodel.schedulers",
    "torchtitan.models.common.attention",
    "torchtitan.models.common.nn_modules",
    "torchtitan.observability.**",
    "torchtitan.protocols.**",
    "torchtitan.tools.logging",
    "torchtitan.tools.utils",
]
TORCH_EXPORT_EXTERN_MODULES = [
    "torch.**",
    "torchao.**",
    "numpy.**",
    "einops.**",
    "typing_extensions.**",
    "dataclasses.**",
    "collections.**",
    "argparse.**",
    "sys.**",
]
TORCH_EXPORT_DENY_MODULES = ["openpilot.**", "cereal", "cereal.**", "capnp", "capnp.**"]
TORCH_EXPORT_MOCK_MODULES = ["**"]


def resolve_checkpoint_path(checkpoint_id: str) -> str:
    if "://" in checkpoint_id or os.path.exists(checkpoint_id):
        return checkpoint_id

    parts = checkpoint_id.strip("/").split("/")
    if len(parts) == 2:
        run_id, step = parts
        return posixpath.join(REPORTERV2_HOST.rstrip("/"), "checkpoint", run_id, step)
    return checkpoint_id


def default_output_path(checkpoint_path: str) -> str:
    return posixpath.join(checkpoint_path.rstrip("/"), PACKAGE_NAME)


def build_meta_model(model_config: WorldModel.Config, *, dtype: torch.dtype = torch.bfloat16) -> WorldModelForInference:
    with torch.device("meta"):
        return WorldModelForInference(model_config).to(dtype=dtype).eval()


def copy_model_config(model_config: WorldModel.Config) -> WorldModel.Config:
    model_config = copy.deepcopy(model_config)
    for _fqn, linear_config, parent, attr in list(model_config.traverse(Linear.Config)):
        if type(linear_config) is Linear.Config:
            continue
        new_config = Linear.Config(
            in_features=linear_config.in_features,
            out_features=linear_config.out_features,
            bias=linear_config.bias,
            param_init=linear_config.param_init,
            sharding_config=linear_config.sharding_config,
        )
        if isinstance(parent, list):
            parent[attr] = new_config
        else:
            setattr(parent, attr, new_config)
    return model_config


def model_io_config(model_config: WorldModel.Config) -> WorldModel.Config:
    model_config = copy.deepcopy(model_config)
    if model_config.transformer.attention_impl == "FLEX":
        model_config.transformer.attention_impl = "SDPA"
    if model_config.plan_head.attention_impl == "FLEX":
        model_config.plan_head.attention_impl = "SDPA"
    return model_config


def empty_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu") for name, tensor in model.state_dict().items()}


def state_dict_nbytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def load_dcp_model_state_dict(checkpoint_path: str, model_config: WorldModel.Config) -> dict[str, torch.Tensor]:
    model = build_meta_model(model_config)
    state_dict = empty_cpu_state_dict(model)
    dcp.load(
        state_dict,
        storage_reader=FsspecReader(checkpoint_path),
        checkpoint_id=checkpoint_path,
    )
    return state_dict


def convert_state_dict_to_fp8(
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8MMConfig, quantize_
    from torchao.quantization.granularity import PerTensor

    model = WorldModelForInference(model_config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state_dict, strict=True, assign=True)
    quantize_(
        model.blocks,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            mm_config=Float8MMConfig(),
        ),
    )
    converted = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    return converted


def torch_save_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def save_torch_package(
    *,
    model: WorldModelForInference,
    model_state_dict: dict[str, torch.Tensor],
    model_io: dict[str, Any],
    step: int,
) -> bytes:
    package_buffer = io.BytesIO()
    meta = {"model_io": model_io, "step": step}

    with PackageExporter(package_buffer, debug=True) as exporter:
        exporter.intern(TORCH_EXPORT_INTERN_MODULES)
        exporter.extern(TORCH_EXPORT_EXTERN_MODULES)
        exporter.deny(TORCH_EXPORT_DENY_MODULES)
        exporter.mock(
            TORCH_EXPORT_MOCK_MODULES,
            exclude=TORCH_EXPORT_INTERN_MODULES + TORCH_EXPORT_EXTERN_MODULES + TORCH_EXPORT_DENY_MODULES,
        )
        exporter.save_pickle("model", "model.pkl", model)
        exporter.save_pickle("meta", "meta.pkl", meta)
        exporter.save_binary("assets", "state_dict.pt", torch_save_bytes(model_state_dict))

    package_buffer.seek(0)
    return package_buffer.getvalue()


def write_bytes(path: str, data: bytes) -> None:
    with fsspec.open(path, "wb") as handle:
        handle.write(data)


def build_package(
    *,
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
    step: int,
) -> bytes:
    sl.log_trace_scalar(
        {
            "worldmodel_package.bf16_state_dict_bytes": state_dict_nbytes(state_dict),
            "worldmodel_package.bf16_state_dict_tensors": len(state_dict),
        }
    )

    with sl.log_trace_span("worldmodel_package_convert_fp8"):
        state_dict = convert_state_dict_to_fp8(model_config, state_dict)

    with sl.log_trace_span("worldmodel_package_model_io"):
        io_model = build_meta_model(model_io_config(model_config))
        num_conditioning_frames = 0 if io_model.config.transformer.attention_mask == "NONE" else 14
        model_io = io_model.get_model_io(
            dtype=torch.bfloat16,
            steps=INFERENCE_STEPS,
            num_conditioning_frames=num_conditioning_frames,
        )
        del io_model
        model = build_meta_model(model_config)

    with sl.log_trace_span("worldmodel_package_build"):
        package = save_torch_package(
            model=model,
            model_state_dict=state_dict,
            model_io=model_io,
            step=step,
        )
    sl.log_trace_scalar({"worldmodel_package.package_bytes": len(package)})
    return package


def export_package(
    *,
    checkpoint_path: str,
    output_path: str | None,
    flavor: str,
) -> None:
    step = int(posixpath.basename(checkpoint_path.rstrip("/")))
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    output_path = default_output_path(checkpoint_path) if output_path is None else output_path
    sl.set_step(step)
    logger.info("Packaging worldmodel checkpoint step=%s", step)
    logger.info("DCP checkpoint path: %s", checkpoint_path)
    logger.info("Torch package output path: %s", output_path)

    with sl.log_trace_span("worldmodel_package_load_config"):
        model_config = copy_model_config(model_registry(flavor).model)

    with sl.log_trace_span("worldmodel_package_load_dcp"):
        state_dict = load_dcp_model_state_dict(checkpoint_path, model_config)
    package = build_package(
        model_config=model_config,
        state_dict=state_dict,
        step=step,
    )
    package_bytes = len(package)

    with sl.log_trace_span("worldmodel_package_write"):
        write_bytes(output_path, package)
    logger.info("Saved %.2f GiB torch package to %s", package_bytes / (1024**3), output_path)
    del package, state_dict
    gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package worldmodel DCP checkpoints for inference.")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_path", nargs="?", default=None)
    parser.add_argument("--flavor", default="base")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    init_logger()
    sl.init_structured_logger(source="worldmodel_package_for_inference", output_dir=STRUCTURED_LOG_DIR)
    with sl.log_trace_span("worldmodel_package_total"):
        export_package(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            flavor=args.flavor,
        )


if __name__ == "__main__":
    main()
