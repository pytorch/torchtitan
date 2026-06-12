from __future__ import annotations

import io
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.components import fs
from torchtitan.components.onnx_checkpoint import _ONNX_DTYPE_MAP, OnnxCheckpointManager
from xx.ml_tools.constants.model import ModelInputs

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
        onnx_file: str = ""
        vision_input_names: list[str] = field(default_factory=list)
        temporal_policy_input_names: list[str] = field(default_factory=list)

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.vision_input_names = config.vision_input_names
        self.temporal_policy_input_names = config.temporal_policy_input_names

    def _export_onnx(self, model: nn.Module, path: str) -> None:
        assert isinstance(model, PathModel)
        self._export_one(
            _VisionOnnxModel(model).eval(),
            self._input_dict(self.vision_input_names),
            fs.join_path(path, "vision.onnx"),
        )
        self._export_one(
            _TemporalPolicyOnnxModel(model).eval(),
            self._input_dict(self.temporal_policy_input_names),
            fs.join_path(path, "temporal_policy.onnx"),
        )

    def _export_one(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        path: str,
    ) -> None:
        with torch.no_grad():
            outputs = model(inputs)
        output_names = list(outputs.keys())
        input_names = list(inputs.keys())
        dynamic_shapes = ({name: {0: "b"} for name in input_names},)

        buffer = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                model,
                (inputs, {}),
                buffer,
                dynamo=True,
                external_data=False,
                optimize=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                export_params=True,
                keep_initializers_as_inputs=False,
            )
        with fs.open_file(path, "wb") as f:
            f.write(buffer.getvalue())

    def _input_dict(self, names: list[str]) -> dict[str, torch.Tensor]:
        shapes = dict(zip(self.input_names, self.input_shapes, strict=True))
        dtypes = dict(zip(self.input_names, self.input_dtypes, strict=True))
        return {
            name: torch.zeros(shapes[name], dtype=_ONNX_DTYPE_MAP[dtypes[name]])
            for name in names
        }
