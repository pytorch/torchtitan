# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from torchtitan.config import Configurable

# Type alias for named initializer functions: (fqn, param) -> None
NamedInitializer = Callable[[str, nn.Parameter], None]

# Cache: maps nn.Module subclass -> created Module wrapper class.
# Module classes are typically created at import time and live for
# the process lifetime.
_created_classes: dict[type, type] = {}


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    Initialization follows a three-phase pattern (inspired by sixlib):

    1. ``init_states`` auto-recurses into children, then calls
       ``init_self_parameters`` and ``init_self_buffers`` on the current module.
    2. ``init_self_parameters`` iterates own parameters and delegates to
       ``_init_single_param``, which walks up the parent chain until a module
       with ``param_init`` is found.
    3. ``init_self_buffers`` is a no-op by default.
       Override for device-aware buffer init (e.g., RoPE, MoE).

    Subclasses should NOT override ``init_states`` unless they need custom
    ordering (e.g., weight tying before init). Override ``init_self_parameters``
    or ``init_self_buffers`` instead.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_init: NamedInitializer | None = None

    def init_states(self, **kwargs) -> None:
        """Initialize all states in the module tree.

        1. Recursively calls ``init_states`` on all direct Module children.
        2. Calls ``self.init_self_parameters(**kwargs)``.
        3. Calls ``self.init_self_buffers(**kwargs)``.

        The ``**kwargs`` (e.g., ``buffer_device``) are forwarded to children
        and to the self-init methods.
        """
        for name, child in self.named_children():
            if isinstance(child, Module):
                # Use object.__setattr__ to avoid nn.Module registering
                # these as submodules (which would create circular references).
                object.__setattr__(child, "_init_parent", self)
                object.__setattr__(child, "_init_name", name)
                child.init_states(**kwargs)
            else:
                # Non-Module children (e.g., CheckpointWrapper from activation
                # checkpointing) may contain Module descendants. Recurse into
                # them so their Module subtrees get initialized.
                _init_non_module_subtree(self, name, child, **kwargs)
        self.init_self_parameters(**kwargs)
        self.init_self_buffers(**kwargs)
        # Clean up transient init-time references
        if hasattr(self, "_init_parent"):
            object.__delattr__(self, "_init_parent")
        if hasattr(self, "_init_name"):
            object.__delattr__(self, "_init_name")

    def init_self_parameters(self, **kwargs) -> None:
        """Initialize this module's own parameters using ``param_init``.

        The default iterates ``named_parameters(recurse=False)`` and calls
        ``_init_single_param`` for each, which delegates up the parent chain
        to find a ``param_init`` callable.

        Override only if this module needs custom parameter initialization
        that cannot be expressed via ``param_init`` regex patterns.
        """
        for name, param in self.named_parameters(recurse=False):
            self._init_single_param(name, param)

    def init_self_buffers(self, **kwargs) -> None:
        """Initialize this module's own buffers.

        The default is a no-op. Override for device-aware buffer
        initialization (e.g., RoPE cache, MoE counters).
        """
        pass

    def _init_single_param(self, name: str, param: nn.Parameter) -> None:
        """Initialize a single parameter.

        If this module has ``config.param_init``, calls it with ``(name, param)``.
        Otherwise, delegates to the parent module (set during ``init_states``
        recursion), prepending this module's name to build the full path.

        This produces fully-qualified names like ``layers.5.attention.wo.weight``
        when reaching the ancestor that has ``param_init`` set.
        """
        if (
            hasattr(self, "config")
            and hasattr(self.config, "param_init")
            and self.config.param_init is not None
        ):
            self.config.param_init(name, param)
            return
        if hasattr(self, "_init_parent") and self._init_parent is not None:
            # pyrefly: ignore [missing-attribute]
            self._init_parent._init_single_param(f"{self._init_name}.{name}", param)
            return
        raise ValueError(
            f"No param_init found for parameter '{name}' in {type(self).__name__}. "
            f"Set param_init on this module's Config or on an ancestor's Config."
        )

    @classmethod
    def from_nn_module(cls, nn_module_cls: type[nn.Module]) -> type["Module"]:
        """Create a ``Module``-protocol-compatible version of *nn_module_cls*.

        The returned class inherits from ``(nn_module_cls, Module)`` and has the
        same constructor signature as *nn_module_cls*.

        * If *nn_module_cls* defines ``reset_parameters``, the injected
          ``init_self_parameters`` delegates to it.
        * Otherwise ``init_self_parameters`` is the inherited default from
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

            def init_self_parameters(self: Any, **kwargs: Any) -> None:
                self.reset_parameters()

            attrs["init_self_parameters"] = init_self_parameters

        name = f"Module({nn_module_cls.__name__})"
        new_cls = type(name, (nn_module_cls, Module), attrs)
        new_cls.__module__ = __name__
        new_cls.__qualname__ = name
        _created_classes[nn_module_cls] = new_cls
        return new_cls


def _init_non_module_subtree(
    parent: Module,
    parent_name: str,
    node: nn.Module,
    **kwargs,
) -> None:
    """Recurse into a non-Module nn.Module (e.g., CheckpointWrapper) to
    find and initialize any Module descendants within it.

    This handles the case where activation checkpointing or other wrappers
    insert non-Module nodes in the module tree.
    """
    for child_name, child in node.named_children():
        fqn = f"{parent_name}.{child_name}"
        if isinstance(child, Module):
            object.__setattr__(child, "_init_parent", parent)
            object.__setattr__(child, "_init_name", fqn)
            child.init_states(**kwargs)
        else:
            _init_non_module_subtree(parent, fqn, child, **kwargs)


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``."""

    pass


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    pass


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``."""

    pass
