# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch.nn as nn

from torchtitan.config import Configurable


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    Subclasses with learnable parameters should override ``init_weights``.
    The default implementation is a no-op, which is appropriate for modules
    that have no learnable parameters or are loaded from external checkpoints.
    """

    def init_weights(self, **kwargs) -> None:
        """Initialize weights. Override in subclasses with learnable parameters."""
        pass


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``.

    ``init_weights`` recursively calls ``init_weights`` on each child.
    """

    def init_weights(self, **kwargs) -> None:
        for child in self:
            assert isinstance(child, Module)
            child.init_weights(**kwargs)


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``.

    ``init_weights`` recursively calls ``init_weights`` on each child.
    """

    def init_weights(self, **kwargs) -> None:
        for child in self.values():
            assert isinstance(child, Module)
            child.init_weights(**kwargs)


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``.

    ``init_weights`` recursively calls ``init_weights`` on each child.
    """

    def init_weights(self, **kwargs) -> None:
        for child in self:
            assert isinstance(child, Module)
            child.init_weights(**kwargs)


# Cache: maps nn.Module subclass -> created Module wrapper class.
# Module classes are typically created at import time and live for
# the process lifetime.
_created_classes: dict[type, type] = {}


def _init_weights_from_reset_parameters(self: Any, **kwargs: Any) -> None:
    """``init_weights`` implementation that delegates to ``reset_parameters``."""
    self.reset_parameters()


def create_module_class(nn_module_cls: type[nn.Module]) -> type[Module]:
    """Create a ``Module``-protocol-compatible version of *nn_module_cls*.

    The returned class inherits from ``(nn_module_cls, Module)`` and has the
    same constructor signature as *nn_module_cls*.

    * If *nn_module_cls* defines ``reset_parameters``, the injected
      ``init_weights`` delegates to it.
    * Otherwise ``init_weights`` is the inherited no-op from ``Module``.

    Results are cached so that repeated calls with the same class return
    the identical class object.

    Usage::

        Conv2d = create_module_class(nn.Conv2d)
        LayerNorm = create_module_class(nn.LayerNorm)
        # Then use Conv2d / LayerNorm exactly like nn.Conv2d / nn.LayerNorm
    """
    if nn_module_cls in _created_classes:
        return _created_classes[nn_module_cls]

    attrs: dict[str, Any] = {}
    if hasattr(nn_module_cls, "reset_parameters"):
        attrs["init_weights"] = _init_weights_from_reset_parameters

    name = f"Module({nn_module_cls.__name__})"
    cls = type(name, (nn_module_cls, Module), attrs)
    cls.__module__ = __name__
    cls.__qualname__ = name
    _created_classes[nn_module_cls] = cls
    return cls


def verify_all_module_protocol(model: nn.Module) -> None:
    """Assert every submodule in *model* satisfies the ``Module`` protocol.

    Walks the full module tree. Raises ``RuntimeError`` listing any
    submodule that is not an instance of ``Module``.  This works on
    meta-device models (only inspects the class hierarchy, not tensor data).
    """
    failures: list[tuple[str, str]] = []
    for fqn, mod in model.named_modules():
        if not isinstance(mod, Module):
            failures.append((fqn, type(mod).__name__))
    if failures:
        details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
        raise RuntimeError(
            f"The following modules do not satisfy the Module protocol: {details}"
        )
