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

# ParamInitializer and NamedParamInitializer are co-located here because
# Module.Config uses NamedParamInitializer in its type annotations, and
# common/param_init.py imports from this file — moving NamedParamInitializer
# there would create a circular import.  ParamInitializer stays with it
# for consistency (same family of types).

# Type alias for simple parameter initializers: (param) -> Any
# Uses Any return type because nn.init.* functions return Tensor,
# but the return value is always ignored by the dispatch layer.
ParamInitializer = Callable[[nn.Parameter], Any]


class NamedParamInitializer:
    """Base class for parameter initializers that receive the FQN.

    Most initializers are simple ``ParamInitializer`` callables that only
    need the parameter tensor.  Subclass ``NamedParamInitializer`` when
    the initialization logic depends on the parameter's fully-qualified
    name (e.g., depth-scaled init that parses the layer id from the FQN,
    or regex-based dispatch).

    Instances are set on ``Module._param_init`` or ``Module.Config.param_init``
    and called by ``_init_self_parameters`` with ``(fqn, param)``.
    """

    def __call__(self, name: str, param: nn.Parameter) -> None:
        raise NotImplementedError


# Cache: maps nn.Module subclass -> created Module wrapper class.
# Module classes are typically created at import time and live for
# the process lifetime.
_created_classes: dict[type, type] = {}


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    Initialization follows a three-phase pattern:

    1. ``init_states`` auto-recurses into children, then calls
       ``_init_self_parameters`` and ``_init_self_buffers`` on the current module.
    2. ``_init_self_parameters`` iterates own parameters and calls
       ``_init_param`` for each one.
    3. ``_init_param`` uses the module's own ``_param_init`` if available,
       otherwise walks up to the nearest Module ancestor (parent-walk).
    4. ``_init_self_buffers`` is a no-op by default.
       Override for device-aware buffer init (e.g., RoPE, MoE).

    Subclasses should NOT override ``init_states`` unless they need custom
    ordering (e.g., weight tying before init). Override ``_init_self_buffers``
    for buffer initialization.
    """

    _param_init: NamedParamInitializer | None = None
    # Set during init_states traversal; points to nearest Module ancestor.
    # Uses object.__setattr__ to avoid nn.Module submodule registration.
    _module_parent: "Module | None" = None
    _module_name: str = ""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_init: NamedParamInitializer | None = None

        def build(self, **kwargs):
            # slots=True prevents super().build() from working; call explicitly.
            # Assignment is done here rather than in Module.__init__ because
            # there is no common Module.__init__ that all subclasses call.
            instance = Configurable.Config.build(self, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            return instance

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize all states in the module tree.

        1. Recursively calls ``init_states`` on all direct Module children.
        2. Calls ``self._init_self_parameters()``.
        3. Calls ``self._init_self_buffers(...)``.

        During recursion, each child's ``_module_parent`` and ``_module_name``
        are temporarily set so that ``_init_param`` can walk up to find an
        ancestor with ``_param_init``.  They are cleaned up after init.

        Args:
            buffer_device: Device for buffer initialization (e.g., RoPE, MoE).
        """

        def prefixed_children(module, prefix):
            return [(f"{prefix}{n}", child) for n, child in module.named_children()]

        queue = prefixed_children(self, "")
        while queue:
            child_name, child = queue.pop()
            if isinstance(child, Module):
                # Temporarily set parent info for _init_param's parent-walk.
                # Must use object.__setattr__: nn.Module.__setattr__
                # registers Module values into _modules, which would make
                # the parent appear as a submodule of the child (creating
                # a circular reference that causes infinite recursion).
                object.__setattr__(child, "_module_parent", self)
                object.__setattr__(child, "_module_name", child_name)
                child.init_states(buffer_device=buffer_device)
                object.__setattr__(child, "_module_parent", None)
                object.__setattr__(child, "_module_name", "")
            else:
                # Plain nn.Module (e.g., CheckpointWrapper, torch.compile
                # wrappers) — look inside for Module descendants.
                queue.extend(prefixed_children(child, f"{child_name}."))

        self._init_self_parameters()
        self._init_self_buffers(buffer_device=buffer_device)

    def _init_self_parameters(self) -> None:
        """Initialize this module's own parameters via ``_init_param``.

        Overridden internally by ``from_nn_module`` to delegate to
        ``reset_parameters``. Not intended for subclass override — configure
        parameter initialization via ``param_init`` on the Config instead.
        """
        for name, param in self.named_parameters(recurse=False):
            self._init_param(name, param)

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        """Initialize a single parameter, walking up to parent if needed.

        If this module has ``_param_init``, uses it with *name* (local to this
        module).  Otherwise delegates to the nearest Module ancestor, prepending
        ``_module_name`` to build the fully-qualified name.  This mirrors
        the parent-walk pattern.
        """
        if self._param_init is not None:
            self._param_init(name, param)
            return
        if self._module_parent is None:
            raise ValueError(
                f"No param_init found for parameter '{name}' in "
                f"{type(self).__name__}. Set param_init on this "
                f"module's Config or on an ancestor's Config."
            )
        self._module_parent._init_param(f"{self._module_name}.{name}", param)

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
