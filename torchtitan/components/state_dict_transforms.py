# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict content transform pipeline.

Owns dtype conversion and HF format conversion (to_hf / from_hf),
cleanly separated from checkpoint orchestration in ``checkpoint.py``
and converter transforms in ``ModelWrapper``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


class StateDictTransforms:
    """Dtype conversion and HF format conversion for checkpoint saves/loads.

    Converter-specific transforms (key filtering, interval/export transforms)
    live on ``ModelWrapper`` via the ``ModelConverter`` protocol.  This class
    handles the format-level transforms that are independent of converters.

    Experiments can subclass to inject additional transforms.
    """

    def __init__(
        self,
        *,
        export_dtype: torch.dtype = torch.float32,
        sd_adapter: BaseStateDictAdapter | None = None,
    ) -> None:
        self._export_dtype = export_dtype
        self._sd_adapter = sd_adapter

    @classmethod
    def from_model_spec(
        cls,
        model_spec: ModelSpec,
        model_config: Any,
        export_dtype: str,
        hf_assets_path: str | None,
    ) -> StateDictTransforms:
        """Build from a ModelSpec, resolving the adapter if available."""
        return cls(
            export_dtype=TORCH_DTYPE_MAP[export_dtype],
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
        )

    # -- Properties --

    @property
    def sd_adapter(self) -> BaseStateDictAdapter | None:
        return self._sd_adapter

    @property
    def export_dtype(self) -> torch.dtype:
        return self._export_dtype

    @property
    def fqn_to_index_mapping(self) -> dict[Any, int] | None:
        if self._sd_adapter is None:
            return None
        return self._sd_adapter.fqn_to_index_mapping

    @property
    def hf_assets_path(self) -> str | None:
        if self._sd_adapter is None:
            return None
        return self._sd_adapter.hf_assets_path

    # -- Transforms --

    def apply_dtype_convert(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Cast tensors to the export dtype.

        No-op when export_dtype is float32 (the training default).
        """
        if self._export_dtype == torch.float32:
            return state_dict
        return {k: v.to(self._export_dtype) for k, v in state_dict.items()}

    def apply_to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HF format via sd_adapter."""
        if not self._sd_adapter:
            raise ValueError("apply_to_hf requires sd_adapter")
        return self._sd_adapter.to_hf(state_dict)

    def apply_from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to torchtitan format."""
        if not self._sd_adapter:
            raise ValueError("apply_from_hf requires sd_adapter")
        return self._sd_adapter.from_hf(hf_state_dict)
