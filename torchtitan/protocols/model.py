# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .module import Module

if TYPE_CHECKING:
    from .state_dict_adapter import BaseStateDictAdapter


class StateDictMode(enum.Enum):
    """Which parameters to include in a state dict operation.

    FULL: all keys. For seed saves, DCP resume.
    TRAINABLE: only requires_grad=True keys. For adapter export.
    BASE: only requires_grad=False keys (or all if nothing frozen).
        For initial load in adapter training.
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

    _sd_adapter: BaseStateDictAdapter | None = None
    _adapter_to_hf_fns: list | None = None

    @property
    def sd_adapter(self) -> BaseStateDictAdapter | None:
        return self._sd_adapter

    def set_sd_adapter(
        self,
        sd_adapter: BaseStateDictAdapter | None = None,
        converters: list | None = None,
    ) -> None:
        """Store the state dict adapter and build converter transforms.

        Called once by the trainer after model build.
        """
        self._sd_adapter = sd_adapter
        self._adapter_to_hf_fns = None
        for converter in converters or []:
            build_fn = getattr(converter, "build_external_transforms", None)
            if build_fn is not None:
                ct = build_fn(sd_adapter)
                if ct is not None and "to_external" in ct:
                    if self._adapter_to_hf_fns is None:
                        self._adapter_to_hf_fns = []
                    self._adapter_to_hf_fns.append(ct["to_external"])

    def to_hf(self, sd: dict[str, Any]) -> dict[str, Any]:
        """Convert native base model keys to HF format."""
        if self._sd_adapter is not None:
            sd = self._sd_adapter.to_hf(sd)
        return sd

    def from_hf(self, sd: dict[str, Any]) -> dict[str, Any]:
        """Convert HF-format keys back to native format."""
        if self._sd_adapter is not None:
            sd = self._sd_adapter.from_hf(sd)
        return sd

    def adapter_to_hf(self, sd: dict[str, Any]) -> dict[str, Any]:
        """Convert native adapter keys to PEFT/HF format."""
        if self._adapter_to_hf_fns:
            for fn in self._adapter_to_hf_fns:
                sd = fn(sd)
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
