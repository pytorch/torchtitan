# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import Configurable

# Type alias for simple parameter initializers: (param) -> Any
# Uses Any return type because nn.init.* functions return Tensor,
# but the return value is always ignored by the dispatch layer.
ParamInitializer = Callable[[nn.Parameter], Any]


class _SkipParamInitType:
    """Sentinel type for modules whose parameters are managed externally.

    Set ``module._param_init = SKIP_PARAM_INIT`` to tell the protocol
    that this module's parameters are initialized by a parent or other
    mechanism.  ``_init_self_parameters`` will return immediately without
    raising.
    """

    _instance: "_SkipParamInitType | None" = None

    def __new__(cls) -> "_SkipParamInitType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIP_PARAM_INIT"


SKIP_PARAM_INIT = _SkipParamInitType()


def set_param_init(
    module: "Module",
    param_init: dict[str, ParamInitializer],
) -> None:
    """Set ``_param_init`` on *module*, raising if already set.

    Enforces single-ownership: each module's parameter initialization
    should be defined by exactly one source (either a parent recipe or
    the module's own Config).

    Args:
        module: The Module instance to configure.
        param_init: Mapping of local parameter names to init callables.
    """
    if module._param_init is not None:
        raise ValueError(
            f"{type(module).__name__} already has _param_init set. "
            f"Only one source (parent recipe or child Config) should define it."
        )
    object.__setattr__(module, "_param_init", param_init)


def validate_param_init(model: "Module") -> None:
    """Validate that every Module with parameters has ``_param_init`` set.

    Called from ``Module.Config.build()`` after the recipe function runs,
    to catch missing modules/params at build time rather than at
    ``init_states`` time.

    Args:
        model: The root Module to validate.

    Raises:
        ValueError: If a Module with parameters has no ``_param_init``,
            or if ``_param_init`` is missing entries for some parameters.
    """
    for name, module in model.named_modules():
        if not isinstance(module, Module):
            continue
        own_params = list(module.named_parameters(recurse=False))
        if not own_params:
            continue
        if module._param_init is SKIP_PARAM_INIT:
            continue
        if module._param_init is None:
            raise ValueError(
                f"param_init_fn missed module '{name}' " f"({type(module).__name__})"
            )
        param_init = module._param_init
        assert isinstance(param_init, dict)  # narrowed above
        for param_name, _ in own_params:
            if param_name not in param_init:
                raise ValueError(f"param_init_fn missed '{name}.{param_name}'")


# Cache: maps nn.Module subclass -> created Module wrapper class.
# Module classes are typically created at import time and live for
# the process lifetime.
_created_classes: dict[type, type] = {}


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    Initialization follows a two-phase pattern:

    1. ``init_states`` auto-recurses into children, then calls
       ``_init_self_parameters`` and ``_init_self_buffers`` on the current module.
    2. ``_init_self_parameters`` iterates own parameters and applies each
       entry from ``_param_init``.
    3. ``_init_self_buffers`` is a no-op by default.
       Override for device-aware buffer init (e.g., RoPE, MoE).

    Each module's ``_param_init`` is set by a **recipe function** stored on
    ``Config.param_init_fn``, called from ``Config.build()`` after
    construction.  The recipe walks the built model using attribute access
    and calls ``set_param_init()`` on each module.

    Subclasses should NOT override ``init_states`` unless they need custom
    ordering (e.g., weight tying before init). Override ``_init_self_buffers``
    for buffer initialization.
    """

    # Runtime type: dict[str, ParamInitializer] | _SkipParamInitType | None
    # Annotated as Any to avoid pyrefly union-widening in nn.Module.__getattr__.
    _param_init: Any = None

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_init_fn: Callable[..., None] | None = None

        def build(self, **kwargs: Any) -> "Module":
            # slots=True prevents super().build() from working; call explicitly.
            instance = Configurable.Config.build(self, **kwargs)
            if self.param_init_fn is not None:
                self.param_init_fn(instance)
                validate_param_init(instance)
            return instance

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize all states in the module tree.

        1. Recursively calls ``init_states`` on all direct Module children.
           Non-Module wrappers (e.g., CheckpointWrapper) are traversed
           to find Module descendants inside them.
        2. Calls ``self._init_self_parameters()``.
        3. Calls ``self._init_self_buffers(...)``.

        Args:
            buffer_device: Device for buffer initialization (e.g., RoPE, MoE).
        """

        # Use a stack (LIFO) to match the traversal order of the
        # previous implementation, ensuring identical random state
        # consumption and thus bit-wise identical initialization.
        stack: list[nn.Module] = list(self.children())
        while stack:
            child = stack.pop()
            if isinstance(child, Module):
                child.init_states(buffer_device=buffer_device)
            else:
                # Plain nn.Module (e.g., CheckpointWrapper, torch.compile
                # wrappers) — look inside for Module descendants.
                stack.extend(child.children())
        self._init_self_parameters()
        self._init_self_buffers(buffer_device=buffer_device)

    def _init_self_parameters(self) -> None:
        """Initialize this module's own parameters using ``_param_init``.

        Overridden internally by ``from_nn_module`` to delegate to
        ``reset_parameters``.  Not intended for subclass override —
        configure parameter initialization via recipe functions and
        ``set_param_init()`` instead.
        """
        if self._param_init is SKIP_PARAM_INIT:
            return
        own_params = list(self.named_parameters(recurse=False))
        if not own_params:
            return
        if self._param_init is None:
            raise ValueError(
                f"{type(self).__name__} has parameters but no _param_init. "
                f"Ensure the model's param_init_fn covers all modules."
            )
        param_init = self._param_init
        assert isinstance(param_init, dict)  # narrowed above
        for name, param in own_params:
            if name not in param_init:
                raise ValueError(
                    f"No initializer for '{name}' in {type(self).__name__}."
                )
            param_init[name](param)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Initialize this module's own buffers.

        The default is a no-op. Override for device-aware buffer
        initialization (e.g., RoPE cache, MoE counters).

        Args:
            buffer_device: Target device for buffer creation/initialization.
        """
        pass

    @classmethod
    def from_nn_module(cls, nn_module_cls: type[nn.Module]) -> type["Module"]:
        """Create a ``Module``-protocol-compatible version of *nn_module_cls*.

        The returned class inherits from ``(nn_module_cls, Module)`` and has the
        same constructor signature as *nn_module_cls*.

        * If *nn_module_cls* defines ``reset_parameters``, the injected
          ``_init_self_parameters`` delegates to it.
        * Otherwise ``_init_self_parameters`` is the inherited default from
          ``Module``.

        Results are cached so that repeated calls with the same class return
        the identical class object.

        Usage::

            Conv2d = Module.from_nn_module(nn.Conv2d)
            LayerNorm = Module.from_nn_module(nn.LayerNorm)
            # Then use Conv2d / LayerNorm exactly like nn.Conv2d / nn.LayerNorm
        """
        if nn_module_cls in _created_classes:
            return _created_classes[nn_module_cls]

        attrs: dict[str, Any] = {}
        if hasattr(nn_module_cls, "reset_parameters"):

            def _init_self_parameters(self: Any) -> None:
                self.reset_parameters()

            attrs["_init_self_parameters"] = _init_self_parameters

        name = f"Module({nn_module_cls.__name__})"
        new_cls = type(name, (nn_module_cls, Module), attrs)
        new_cls.__module__ = __name__
        new_cls.__qualname__ = name
        _created_classes[nn_module_cls] = new_cls
        return new_cls


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``."""

    pass


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    pass


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``."""

    pass
