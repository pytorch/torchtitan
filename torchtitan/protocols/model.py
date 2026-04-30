# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)

from .module import Module


class StateDictMode(enum.Enum):
    """Mode for BaseModel.get_sd().

    FULL: all keys. For seed saves, DCP resume.
    TRAINABLE: only requires_grad=True keys. For interval saves, adapter export.
    BASE: only requires_grad=False keys (or all if nothing frozen).
        For HF load containers.
    """

    FULL = "full"
    TRAINABLE = "trainable"
    BASE = "base"


class BaseModel(Module):
    """Base class for all model classes.

    Models inherit from BaseModel (which is Module = nn.Module + Configurable).
    Each model defines a nested Config(BaseModel.Config) with model hyperparameters.
    The model is constructed via ``config.build()``.

    ``init_states`` (from Module) auto-recurses; override only for custom
    ordering (e.g., weight tying before init).
    """

    _pp_parts: list[nn.Module] | None = None
    """Set by the trainer when PP splits the model into multiple parts.
    When set, get_sd/load_sd operate on all parts.
    When None (default), they operate on ``self`` only."""

    def _get_parts(self) -> list[nn.Module]:
        return self._pp_parts if self._pp_parts is not None else [self]

    def _raw_sd(self) -> dict[str, Any]:
        """Merge state dicts from all PP parts."""
        sd: dict[str, Any] = {}
        for part in self._get_parts():
            sd.update(get_model_state_dict(part))
        return sd

    def get_sd(
        self,
        mode: StateDictMode = StateDictMode.FULL,
        *,
        external: bool = False,
        dtype: "torch.dtype | None" = None,
    ) -> dict[str, Any]:
        """Get state dict for checkpointing.

        Args:
            mode: Which params to include.
                FULL: all keys.
                TRAINABLE: only requires_grad=True keys (uses DCP's
                    ignore_frozen_params). When no params are frozen,
                    TRAINABLE == FULL.
                BASE: only requires_grad=False keys (or all if nothing frozen).
            external: If True, remap keys to external format via
                ``to_external`` (sd_adapter HF remapping, then converter
                transforms in order).
            dtype: Optional dtype cast for export (e.g. bfloat16).
        """
        if mode == StateDictMode.FULL:
            sd = self._raw_sd()
        elif mode == StateDictMode.TRAINABLE:
            sd = {}
            for part in self._get_parts():
                sd.update(
                    get_model_state_dict(
                        part,
                        options=StateDictOptions(ignore_frozen_params=True),
                    )
                )
        else:
            # BASE: full minus trainable
            full_sd = self._raw_sd()
            trainable_sd = self.get_sd(StateDictMode.TRAINABLE)
            if len(trainable_sd) == len(full_sd):
                sd = full_sd
            else:
                sd = {k: v for k, v in full_sd.items() if k not in trainable_sd}

        if external:
            sd = self.to_external(sd, mode)

        if dtype is not None:
            sd = {
                k: v.to(dtype) if isinstance(v, torch.Tensor) else v
                for k, v in sd.items()
            }

        return sd

    def load_sd(self, sd: dict[str, Any]) -> None:
        """Load state dict with strict=False (partial loads OK)."""
        for part in self._get_parts():
            set_model_state_dict(
                part,
                model_state_dict=sd,
                options=StateDictOptions(strict=False),
            )

    # State dict transforms, set once by set_sd_transforms().
    _to_hf = None
    _from_hf = None
    _converters_to_external: list | None = None
    _converters_from_external: list | None = None

    def set_sd_transforms(
        self, sd_adapter=None, converters: list | None = None
    ) -> None:
        """Wire state dict transforms onto this model.

        Extracts HF key remapping from sd_adapter and collects converter
        transforms. Each converter may implement
        ``build_external_transforms(sd_adapter)`` returning a dict with
        ``to_external``/``from_external`` callables.

        Called once by the trainer after model build.
        """
        if sd_adapter is not None:
            self._to_hf = sd_adapter.to_hf
            self._from_hf = sd_adapter.from_hf
        self._converters_to_external = None
        self._converters_from_external = None
        for converter in converters or []:
            build_fn = getattr(converter, "build_external_transforms", None)
            if build_fn is not None:
                ct = build_fn(sd_adapter)
                if ct is not None:
                    if self._converters_to_external is None:
                        self._converters_to_external = []
                        self._converters_from_external = []
                    if "to_external" in ct:
                        self._converters_to_external.append(ct["to_external"])
                    if "from_external" in ct:
                        self._converters_from_external.append(ct["from_external"])

    def to_external(self, sd: dict[str, Any], mode: StateDictMode) -> dict[str, Any]:
        """Convert native state dict keys to external format.

        The mode determines which transforms to apply:
        - BASE: HF remapping only (base model keys)
        - TRAINABLE: converter transforms only (adapter keys)
        - FULL: both (base + adapter keys)
        """
        if mode in (StateDictMode.FULL, StateDictMode.BASE):
            if self._to_hf is not None:
                sd = self._to_hf(sd)
        if mode in (StateDictMode.FULL, StateDictMode.TRAINABLE):
            if self._converters_to_external:
                for fn in self._converters_to_external:
                    sd = fn(sd)
        return sd

    def from_external(self, sd: dict[str, Any], mode: StateDictMode) -> dict[str, Any]:
        """Convert external format state dict keys to native format.

        Reverse of ``to_external``: converter transforms (reversed) then
        HF remapping, filtered by mode.
        """
        if mode in (StateDictMode.FULL, StateDictMode.TRAINABLE):
            if self._converters_from_external:
                for fn in reversed(self._converters_from_external):
                    sd = fn(sd)
        if mode in (StateDictMode.FULL, StateDictMode.BASE):
            if self._from_hf is not None:
                sd = self._from_hf(sd)
        return sd

    def init_weights(self, **kwargs) -> None:
        """Backward-compatible alias for ``init_states``.

        External tools (e.g., AutoParallel) wrap ``init_weights`` with
        DTensor-aware interception. This alias ensures they can find it.
        """
        # TODO: remove this once autoparallel has wrap_init_states
        buffer_device = kwargs.get("buffer_device")
        self.init_states(buffer_device=buffer_device)

    def verify_module_protocol(self) -> None:
        """Verify all submodules satisfy the ``Module`` protocol.

        Catches non-``Module`` submodules early with a clear error message,
        preventing obscure failures when the ``Module`` protocol is being
        used later.

        Override in models where some internal ``nn.Module`` submodules
        cannot conform to the ``Module`` protocol.
        """
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: "
                f"{details}"
            )

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Base config for all models.

        Subclasses define model-specific hyperparameters.
        """

        # TODO: This function violates encapsulation;
        # maybe replace it with config passes from outside.
        @abstractmethod
        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            pass

        @abstractmethod
        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            pass
