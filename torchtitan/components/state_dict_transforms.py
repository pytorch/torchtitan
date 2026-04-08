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

from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter


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

        No-op when export_dtype is float32 (the training default).
        """
        if self._export_dtype != torch.float32:
            state_dict = {k: v.to(self._export_dtype) for k, v in state_dict.items()}
        return state_dict

    # -- Load-side transforms --

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        """Return the HF storage reader for loading.

        Delegates to ``sd_adapter.get_hf_storage_reader()``.
        """
        assert (
            self._sd_adapter is not None
        ), "get_hf_storage_reader requested but no sd_adapter provided"
        return self._sd_adapter.get_hf_storage_reader(path, from_quantized)
