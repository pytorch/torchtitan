# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict content transform pipeline.

Cleanly separates state dict content transforms (dtype conversion, FQN
renaming, value permutations, etc.) from checkpoint orchestration
(save/load/resume/purge/async staging) in ``checkpoint.py``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


class StateDictTransforms:
    """Ordered pipeline of state dict content transforms for save and load.

    Save-side (export / last-step):
        converter transform (via ModelWrapper.state_dict(mode="export")) ->
        dtype_convert -> to_hf (if requested)

    Load-side (import / initial load):
        from_hf (if loading HF) -> state dict ready for set_model_state_dict

    The converter transform (Float8, etc.) is *not* owned by this
    class — it lives on ``ModelWrapper.state_dict(mode="export")`` because it
    needs access to the model's internal state dict machinery.  This pipeline
    handles everything *after* the state dict has been extracted.

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

    # -- Properties for checkpoint.py to access adapter capabilities --

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

    # -- Save-side transforms --

    def apply_dtype_convert(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Cast all tensors to the export dtype.

        No-op when export_dtype is float32 (the training default) — float32
        is the native training dtype, so no conversion is needed.
        Assumes all values are tensors — this is guaranteed when called
        after ``ModelWrapper.state_dict()``, which only returns tensors.
        """
        if self._export_dtype != torch.float32:
            state_dict = {k: v.to(self._export_dtype) for k, v in state_dict.items()}
        return state_dict
