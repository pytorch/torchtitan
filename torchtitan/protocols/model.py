# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

    def get_sd(self, mode: StateDictMode = StateDictMode.FULL) -> dict[str, Any]:
        """Get state dict for checkpointing.

        Args:
            mode: FULL returns all keys. TRAINABLE returns only requires_grad=True
                keys (uses DCP's ignore_frozen_params for correct FQN handling
                with torch.compile / DDP wrappers). BASE returns only
                requires_grad=False keys (or all if nothing is frozen).
        """
        if mode == StateDictMode.FULL:
            return self._raw_sd()

        if mode == StateDictMode.TRAINABLE:
            sd: dict[str, Any] = {}
            for part in self._get_parts():
                sd.update(
                    get_model_state_dict(
                        part,
                        options=StateDictOptions(ignore_frozen_params=True),
                    )
                )
            return sd

        # BASE: full minus trainable
        full_sd = self._raw_sd()
        trainable_sd = self.get_sd(StateDictMode.TRAINABLE)
        if len(trainable_sd) == len(full_sd):
            return full_sd
        return {k: v for k, v in full_sd.items() if k not in trainable_sd}

    def load_sd(self, sd: dict[str, Any]) -> None:
        """Load state dict with strict=False (partial loads OK)."""
        for part in self._get_parts():
            set_model_state_dict(
                part,
                model_state_dict=sd,
                options=StateDictOptions(strict=False),
            )

    _to_external: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    _from_external: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def to_external(self, adapter_sd: dict[str, Any]) -> dict[str, Any]:
        """Convert native adapter SD to external format (e.g. PEFT).

        Set ``_to_external`` callable via trainer wiring (e.g. from
        ``LoRAStateDictAdapter.to_hf``).
        """
        if self._to_external is not None:
            return self._to_external(adapter_sd)
        return adapter_sd

    def from_external(self, external_sd: dict[str, Any]) -> dict[str, Any]:
        """Convert external format (e.g. PEFT) to native adapter SD.

        Set ``_from_external`` callable via trainer wiring.
        """
        if self._from_external is not None:
            return self._from_external(external_sd)
        return external_sd

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
