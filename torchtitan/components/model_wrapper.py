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
    """Mode for ``ModelWrapper.state_dict()`` and ``load_state_dict()``.

    RAW: Complete state dict with all keys in native format, no transforms.
    EXPORT: Apply converter transform for saving. The converter's own
        config determines the exact behavior for interval vs export saves.
    BASE_EXTERNAL: Base model keys in external format. For building DCP load
        containers that match an HF checkpoint on disk.
    CONVERTER_EXTERNAL: Converter-owned keys in external format. For building
        DCP load containers or loading converter checkpoints (e.g. PEFT).
    """

    RAW = "raw"
    EXPORT = "export"
    BASE_EXTERNAL = "base_external"
    CONVERTER_EXTERNAL = "converter_external"


class ModelWrapper(Stateful):
    """Wraps model parts into a :class:`Stateful` for checkpoint integration.

    Owns model state dict content transforms (converter key filtering and
    converter transforms). Format conversion (to_hf/from_hf) is provided
    as composable helpers.

    State dict modes:

    - ``state_dict(RAW)`` — all keys in native format
    - ``state_dict(EXPORT)`` — converter-transformed for saves
    - ``state_dict(BASE_EXTERNAL)`` — base keys in external format (load container)

    Format helpers (pure transforms, no filtering):

    - ``to_hf(sd)`` / ``from_hf(sd)`` — base model native ↔ HF format
    - ``converter_to_hf(sd)`` / ``converter_from_hf(sd)`` — converter native ↔ external format

    Key filtering is handled internally by ``state_dict(mode)`` and
    ``load_state_dict(sd, mode)`` via ``_partition()``.

    Args:
        model: A single model or list of model parts (e.g. pipeline stages).
        key_filter: Identifies converter-owned keys. Used internally by
            ``_partition()`` to separate base and converter keys.
        converter_transform: Converter transform taking ``(sd, last_step)``
            and returning the transformed state dict. Used in EXPORT mode.
        sd_adapter: Base model state dict adapter providing ``to_hf``/``from_hf``.
            Callables are extracted during construction; the adapter reference
            is not stored.
        converter_sd_adapter: Converter state dict adapter providing
            ``to_hf``/``from_hf`` for the converter's external format.
        to_hf: Explicit callable override (used by tests). ``sd_adapter``
            takes precedence if both are provided.
        from_hf: Explicit callable override (used by tests).
    """

    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        *,
        key_filter: Callable[[str], bool] | None = None,
        converter_transform: (
            Callable[[dict[str, Any], bool], dict[str, Any]] | None
        ) = None,
        sd_adapter: Any | None = None,
        converter_sd_adapter: Any | None = None,
        to_hf: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        from_hf: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self._key_filter = key_filter
        self._converter_transform = converter_transform
        self._to_hf = sd_adapter.to_hf if sd_adapter else to_hf
        self._from_hf = sd_adapter.from_hf if sd_adapter else from_hf
        self._converter_to_hf = (
            converter_sd_adapter.to_hf if converter_sd_adapter else None
        )
        self._converter_from_hf = (
            converter_sd_adapter.from_hf if converter_sd_adapter else None
        )

    def _get_state_dict(self) -> dict[str, Any]:
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def _partition(
        self, state_dict: dict[str, Any], keep_converter: bool
    ) -> dict[str, Any]:
        """Split state dict into base or converter keys.

        Args:
            state_dict: The state dict to partition.
            keep_converter: If True, keep only converter-owned keys.
                If False, keep only base (non-converter) keys.
        """
        if self._key_filter is None:
            return {} if keep_converter else state_dict
        return {
            k: v for k, v in state_dict.items() if self._key_filter(k) == keep_converter
        }

    def state_dict(
        self,
        mode: StateDictMode = StateDictMode.RAW,
        *,
        last_step: bool = False,
    ) -> dict[str, Any]:
        """Return the model state dict in the requested mode.

        Args:
            mode: Controls key filtering and format conversion.

                - RAW: all keys, native format, no transforms.
                - EXPORT: converter-transformed for saves.
                - BASE_EXTERNAL: base keys in external format (load container).
                - CONVERTER_EXTERNAL: converter keys in external format (load container).

            last_step: Only used in EXPORT mode. Passed to
                converter_transform so it can distinguish interval
                saves from final exports.
        """
        sd = self._get_state_dict()
        if mode == StateDictMode.EXPORT:
            if self._converter_transform is not None:
                sd = self._converter_transform(sd, last_step)
        elif mode == StateDictMode.BASE_EXTERNAL:
            sd = self.to_hf(self._partition(sd, keep_converter=False))
        elif mode == StateDictMode.CONVERTER_EXTERNAL:
            sd = self.converter_to_hf(self._partition(sd, keep_converter=True))
        return sd

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        mode: StateDictMode = StateDictMode.RAW,
    ) -> None:
        """Load a state dict into the model.

        Args:
            state_dict: The state dict to load.
            mode: Controls reverse format transform before loading.

                - RAW: load as-is (native format).
                - BASE_EXTERNAL: apply from_hf + partition before loading.
                - CONVERTER_EXTERNAL: apply converter_from_hf + partition before loading.
                - EXPORT: same as RAW (no reverse transform).
        """
        if mode == StateDictMode.BASE_EXTERNAL:
            state_dict = self._partition(self.from_hf(state_dict), keep_converter=False)
        elif mode == StateDictMode.CONVERTER_EXTERNAL:
            state_dict = self._partition(
                self.converter_from_hf(state_dict), keep_converter=True
            )
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))

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

    def converter_to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a native converter state dict to external format."""
        if self._converter_to_hf is None:
            raise ValueError("converter_to_hf requires a converter_to_hf callable")
        return self._converter_to_hf(state_dict)

    def converter_from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an external format converter state dict to native format."""
        if self._converter_from_hf is None:
            raise ValueError("converter_from_hf requires a converter_from_hf callable")
        return self._converter_from_hf(hf_state_dict)
