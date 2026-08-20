# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for relocated checkpoint and optimizer utilities."""

from .checkpointer.utils import canonical_fqn
from .optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

__all__ = [
    "canonical_fqn",
    "init_optim_state",
    "get_flat_optim_state_dict",
    "load_flat_optim_state_dict",
]
