# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import functools
from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful


class StateDictMode(str, enum.Enum):
    """Mode for ``ModelWrapper.state_dict()``.

    RAW: Complete state dict with all keys in native format, no transforms.
    EXPORT: Apply converter transform for saving. The converter's own
        config determines the exact behavior for interval vs export saves.
    """

    RAW = "raw"
    EXPORT = "export"


class ModelWrapper(Stateful):
    """Wraps model parts into a :class:`Stateful` for checkpoint integration.

    Owns model state dict content transforms (converter key filtering and
    converter transforms). Format conversion (to_hf/from_hf) is provided
    as composable helpers that callers apply as needed.

    Two state dict producers:

    - ``state_dict()`` — RAW mode, all keys in native format
    - ``state_dict(EXPORT)`` — converter-transformed for saves

    Three composable helpers:

    - ``filter_base_keys(sd)`` — exclude converter-owned keys
    - ``to_hf(sd)`` — convert native state dict to HF format
    - ``from_hf(sd)`` — convert HF state dict to native format

    Args:
        model: A single model or list of model parts (e.g. pipeline stages).
        key_filter: Identifies converter-owned keys. Used by
            ``filter_base_keys()`` to exclude converter-owned keys.
        converter_transform: Converter transform taking ``(sd, last_step)``
            and returning the transformed state dict. Used in EXPORT mode.
        to_hf: Callable that converts native state dict to HF format.
        from_hf: Callable that converts HF state dict to native format.
    """

    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        *,
        key_filter: Callable[[str], bool] | None = None,
        converter_transform: (
            Callable[[dict[str, Any], bool], dict[str, Any]] | None
        ) = None,
        to_hf: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        from_hf: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self._key_filter = key_filter
        self._converter_transform = converter_transform
        self._to_hf = to_hf
        self._from_hf = from_hf

    def _get_state_dict(self) -> dict[str, Any]:
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def state_dict(
        self,
        mode: StateDictMode = StateDictMode.RAW,
        *,
        last_step: bool = False,
    ) -> dict[str, Any]:
        """Return the model state dict in the requested mode.

        Args:
            mode: RAW returns all keys unmodified. EXPORT applies
                converter_transform.
            last_step: Only used in EXPORT mode. Passed to
                converter_transform so it can distinguish interval
                saves from final exports. Ignored in RAW mode.
        """
        sd = self._get_state_dict()
        if mode == StateDictMode.EXPORT:
            if self._converter_transform is not None:
                sd = self._converter_transform(sd, last_step)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))

    def filter_base_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Exclude converter-owned keys from a state dict.

        Used today for HF load containers (HF checkpoints don't contain
        adapter keys). Will also be used for multi-source loading where
        base and adapter weights are loaded from separate sources.
        """
        if self._key_filter is None:
            return state_dict
        return {k: v for k, v in state_dict.items() if not self._key_filter(k)}

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a native state dict to HF format."""
        if self._to_hf is None:
            raise ValueError("to_hf requires a to_hf callable")
        return self._to_hf(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an HF state dict to native format."""
        if self._from_hf is None:
            raise ValueError("from_hf requires a from_hf callable")
        return self._from_hf(hf_state_dict)
