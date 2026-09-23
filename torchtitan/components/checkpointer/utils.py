# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FQN helpers shared by checkpointing and the components it serializes."""

__all__ = [
    "canonical_fqn",
]

# Segment inserted by the activation checkpoint wrapper (checkpoint_wrapper) in
# named_parameters(). It can appear at any level of the FQN and is not part of the
# canonical model contract. torch.compile is applied in place and adds no segment.
_WRAPPER_PREFIXES: tuple[str, ...] = ("_checkpoint_wrapped_module",)


def canonical_fqn(name: str, prefixes: tuple[str, ...] = _WRAPPER_PREFIXES) -> str:
    """Strip wrapper segments from a dotted FQN.

    A segment may appear at any level, e.g.
    ``layers.0._checkpoint_wrapped_module.attention.wq.weight`` ->
    ``layers.0.attention.wq.weight``.
    """
    return ".".join(p for p in name.split(".") if p not in prefixes)
