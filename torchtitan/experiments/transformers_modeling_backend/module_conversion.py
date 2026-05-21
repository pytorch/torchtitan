# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert plain HF nn.Module instances to TorchTitan Module protocol.

Uses ``Module.from_nn_module`` to dynamically create classes that inherit
from both the original HF class and ``Module``, then swaps ``__class__``
on existing instances. This gives every HF module the ``parallelize()``,
``_shard_states()``, and ``_sharding_config`` capabilities without changing
any module state, forward behavior, or state_dict keys.
"""

import torch.nn as nn

from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


def convert_hf_to_module(model: nn.Module) -> None:
    """Convert all plain nn.Module children to Module protocol instances.

    Recursively walks the module tree via ``children()`` (matching the
    traversal that ``Module.parallelize()`` uses) and swaps each plain
    ``nn.Module``'s ``__class__`` to a ``Module``-wrapped version.

    Modules that are already ``Module`` instances (e.g., native MoE,
    ``HFTransformerModel`` itself) are skipped.
    """
    for child in model.children():
        if not isinstance(child, Module):
            cls = type(child)
            if any("__slots__" in k.__dict__ for k in cls.__mro__[:-1]):
                logger.warning(
                    f"Skipping Module conversion for {cls.__name__} "
                    f"(__slots__ incompatible with __class__ swap)"
                )
            else:
                child.__class__ = Module.from_nn_module(cls)
        convert_hf_to_module(child)

    logger.info("Converted HF modules to Module protocol")
