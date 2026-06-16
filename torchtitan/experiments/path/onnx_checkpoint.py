from __future__ import annotations

from dataclasses import dataclass, field

import onnx
import torch
import torch.nn as nn

from xx.ml_tools.constants.model import ModelInputs
from xx.training.lib.onnx_helpers import add_onnx_metadata, patch_depthwise_convs

from torchtitan.components import fs
from torchtitan.components.onnx_checkpoint import (
    _ONNX_DTYPE_MAP,
    OnnxCheckpointManager,
    OnnxInputDType,
)

from .model import PathModel


class _VisionOnnxModel(nn.Module):
    def __init__(self, model: PathModel) -> None:
        super().__init__()
        self.vision = model.vision
        self.point_policy = model.point_policy

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.vision(inputs)
        return self.point_policy(features) | {"vision_features": features}


class _TemporalPolicyOnnxModel(nn.Module):
    def __init__(self, model: PathModel) -> None:
        super().__init__()
        self.temporal_policy = model.temporal_policy

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.temporal_policy(
            inputs[ModelInputs.FEATURES],
            inputs[ModelInputs.DESIRE],
            inputs[ModelInputs.TRAFFIC],
            inputs[ModelInputs.ACTION_T],
        )
        return {name: value[:, -1] if value.ndim == 3 else value for name, value in outputs.items()}


class PathOnnxCheckpointManager(OnnxCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(OnnxCheckpointManager.Config):
        checkpoint_base_folder: str = ""
        onnx_file: str = ""
        onnx_model_dtype: OnnxInputDType = "float32"
        vision_input_names: list[str] = field(default_factory=list)
        temporal_policy_input_names: list[str] = field(default_factory=list)

    def __init__(self, config: Config, **kwargs) -> None:
        if config.checkpoint_base_folder:
            kwargs["base_folder"] = config.checkpoint_base_folder
        super().__init__(config, **kwargs)
        self.onnx_model_dtype = _ONNX_DTYPE_MAP[config.onnx_model_dtype]
        self.vision_input_names = config.vision_input_names
        self.temporal_policy_input_names = config.temporal_policy_input_names

    def _export_onnx(self, model: nn.Module, path: str) -> None:
        assert isinstance(model, PathModel)
        model = model.to(dtype=self.onnx_model_dtype)
        self._export_one(
            _VisionOnnxModel(model).eval(),
            self._input_dict(self.vision_input_names),
            fs.join_path(path, "vision.onnx"),
            dynamo=True,
            optimize=True,
            external_data=True,
        )
        self._export_one(
            _TemporalPolicyOnnxModel(model).eval(),
            self._input_dict(self.temporal_policy_input_names),
            fs.join_path(path, "temporal_policy.onnx"),
            dynamo=True,
            optimize=True,
        )

    def _post_export_hook(self, onnx_data: bytes) -> bytes:
        onnx_data = patch_depthwise_convs(onnx_data)
        onnx_model = onnx.load_from_string(onnx_data)
        add_onnx_metadata(
            onnx_model,
            exporter_name="comma_torchtitan",
            exporter_version="0.1",
        )
        return onnx_model.SerializeToString()

    def _input_dict(self, names: list[str]) -> dict[str, torch.Tensor]:
        shapes = dict(zip(self.input_names, self.input_shapes, strict=True))
        dtypes = dict(zip(self.input_names, self.input_dtypes, strict=True))
        return {
            name: torch.zeros(shapes[name], dtype=_ONNX_DTYPE_MAP[dtypes[name]])
            for name in names
        }
