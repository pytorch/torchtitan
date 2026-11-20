# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert plain HF nn.Module instances to TorchTitan Module protocol.

Dynamically creates classes that inherit from both the original HF class
and ``Module``, then swaps ``__class__`` on existing instances. This gives
every HF module the ``parallelize()``, ``_distribute_states()``, and
``_sharding_config`` capabilities without changing any module state,
forward behavior, or state_dict keys. After conversion,
``set_hf_sharding_configs`` sets ``_sharding_config`` on each module and
``model.parallelize()`` distributes parameters and wraps forwards.
"""

import functools
import logging

import torch.nn as nn

from torchtitan.models.base.module import Module

logger = logging.getLogger(__name__)


_module_class_cache: dict[type, type] = {}
_forward_override_class_cache: dict[tuple[type, type], type] = {}


def _make_module_class(nn_cls: type) -> type:
    """Create a class inheriting from both Module and nn_cls.

    Results are cached so repeated calls for the same class return
    the same type object.
    """
    if nn_cls in _module_class_cache:
        return _module_class_cache[nn_cls]
    new_cls = type(nn_cls.__name__, (Module, nn_cls), {})
    _module_class_cache[nn_cls] = new_cls
    return new_cls


def override_forward(module: nn.Module, override: type, **attrs: object) -> None:
    """Run ``override.forward`` in place of ``module``'s forward.

    ``override.forward`` reaches the original forward through ``super()``.
    ``module`` is swapped to a cached subclass of its class, like
    ``convert_hf_to_module`` does, so module state and state_dict keys are
    unchanged. The new ``forward`` reports the HF signature: ``Module`` maps
    positional inputs to its sharding config by parameter name. ``attrs`` are
    set on the instance.
    """
    cls = type(module)
    key = (override, cls)
    if key not in _forward_override_class_cache:
        override_fn = override.forward

        @functools.wraps(cls.forward)
        def forward(self, *args, **kwargs):
            return override_fn(self, *args, **kwargs)

        _forward_override_class_cache[key] = type(
            cls.__name__, (override, cls), {"forward": forward}
        )
    module.__class__ = _forward_override_class_cache[key]
    for name, value in attrs.items():
        setattr(module, name, value)


def convert_hf_to_module(model: nn.Module) -> None:
    """Convert all plain nn.Module children to Module protocol instances.

    Recursively walks the module tree via ``children()`` (matching the
    traversal that ``Module.parallelize()`` uses) and swaps each plain
    ``nn.Module``'s ``__class__`` to a ``Module``-wrapped version.

    Modules that are already ``Module`` instances (e.g., Titan MoE,
    ``HFTransformerModel`` itself) are skipped.
    """
    from torchtitan.experiments.transformers_modeling_backend.hf_sharding import (
        HFVocabParallelEmbedding,
    )

    for child in model.children():
        if type(child) is nn.Embedding:
            child.__class__ = HFVocabParallelEmbedding
        elif not isinstance(child, Module):
            cls = type(child)
            if any("__slots__" in k.__dict__ for k in cls.__mro__[:-1]):
                logger.warning(
                    f"Skipping Module conversion for {cls.__name__} "
                    f"(__slots__ incompatible with __class__ swap)"
                )
            else:
                child.__class__ = _make_module_class(cls)
        convert_hf_to_module(child)

    logger.info("Converted HF modules to Module protocol")
