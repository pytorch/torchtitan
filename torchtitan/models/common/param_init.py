# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter initialization utilities and factories for torchtitan models.

Provides ``NamedInitializer`` primitives (``init_by_regex``, ``init_zeros``,
etc.) and higher-level factories like ``make_decoder_param_init``.

Example usage in a model's ``__init__.py``::

    from torchtitan.models.common.param_init import (
        init_by_regex,
        make_decoder_param_init,
    )

    cfg = Llama3Model.Config(
        ...,
        param_init=init_by_regex(
            make_decoder_param_init(dim=256, n_layers=6)
        ),
    )
"""

import re
from collections.abc import Callable

import torch.nn as nn

# Type alias for named initializer functions: (fqn, param) -> None
NamedInitializer = Callable[[str, nn.Parameter], None]


def init_by_regex(
    regex_to_init: dict[str, NamedInitializer],
) -> NamedInitializer:
    """Return a ``NamedInitializer`` that regex-matches parameter FQNs.

    The first matching pattern wins. Raises ``ValueError`` if no pattern matches.

    Args:
        regex_to_init: Mapping from regex pattern to initializer function.
            Each initializer accepts ``(name: str, param: nn.Parameter) -> None``.
    """

    def init(name: str, param: nn.Parameter) -> None:
        for pattern, init_fn in regex_to_init.items():
            if re.fullmatch(pattern, name):
                init_fn(name, param)
                return
        raise ValueError(f"No initializers matched '{name}'.")

    return init


def init_trunc_normal(
    *, std: float = 0.02, mean: float = 0.0, a: float = -2.0, b: float = 2.0
) -> NamedInitializer:
    """Truncated normal init.

    Default bounds ``a=-2.0, b=2.0`` match ``torch.nn.init.trunc_normal_``
    defaults for bitwise reproducibility with the old ``init_weights`` code.
    """

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.trunc_normal_(param, mean=mean, std=std, a=a, b=b)

    return init


def init_normal(*, std: float = 0.02, mean: float = 0.0) -> NamedInitializer:
    """Normal init."""

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.normal_(param, mean=mean, std=std)

    return init


def init_ones() -> NamedInitializer:
    """Fill with ones."""

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.ones_(param)

    return init


def init_zeros() -> NamedInitializer:
    """Fill with zeros."""

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.zeros_(param)

    return init


def init_xavier_uniform() -> NamedInitializer:
    """Xavier uniform init."""

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.xavier_uniform_(param)

    return init


def init_constant(*, val: float) -> NamedInitializer:
    """Fill with a constant value."""

    def init(name: str, param: nn.Parameter) -> None:
        nn.init.constant_(param, val)

    return init


def _extract_layer_id(name: str) -> int | None:
    """Extract layer_id from FQN like ``layers.5.attention.wo.weight``."""
    m = re.match(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def init_depth_scaled_trunc_normal(
    *,
    base_std: float = 0.02,
    n_layers: int,
    depth_init: bool = True,
    a: float = -2.0,
    b: float = 2.0,
) -> NamedInitializer:
    """Truncated normal with depth-dependent std.

    Parses ``layer_id`` from the FQN (e.g., ``layers.5.attention.wo.weight``)
    and computes ``std = base_std / (2 * (layer_id + 1)) ** 0.5``.
    Falls back to ``base_std / (2 * n_layers) ** 0.5`` when ``depth_init``
    is False or the layer_id cannot be extracted.

    Default bounds ``a=-2.0, b=2.0`` match ``torch.nn.init.trunc_normal_``.
    """

    def init(name: str, param: nn.Parameter) -> None:
        layer_id = _extract_layer_id(name)
        if depth_init and layer_id is not None:
            std = base_std / (2 * (layer_id + 1)) ** 0.5
        else:
            std = base_std / (2 * n_layers) ** 0.5
        nn.init.trunc_normal_(param, mean=0.0, std=std, a=a, b=b)

    return init


def make_decoder_param_init(
    *,
    dim: int,
    n_layers: int,
    depth_init: bool = True,
    base_std: float = 0.02,
    tok_emb_std: float = 1.0,
) -> dict[str, NamedInitializer]:
    """Common param_init patterns for Decoder-based models."""

    depth_std = init_depth_scaled_trunc_normal(
        base_std=base_std, n_layers=n_layers, depth_init=depth_init
    )
    return {
        # Token embeddings
        r"tok_embeddings\.weight": init_normal(std=tok_emb_std),
        # Depth-scaled output projections in attention and FFN
        r"layers\..+\.attention\.wo\.weight": depth_std,
        r"layers\..+\.feed_forward\.w[23]\.weight": depth_std,
        # MoE expert weights (nn.Parameter, no .weight suffix)
        r"layers\..+\.moe\.experts\.w[23]": depth_std,
        # MoE router gate
        r"layers\..+\.moe\.router\.gate\.weight": depth_std,
        # MoE shared experts
        r"layers\..+\.moe\.shared_experts\.w[23]\.weight": depth_std,
        # Norm weights (RMSNorm, LayerNorm)
        r".*norm.*\.weight": init_ones(),
        # Output projection (cutoff bounds match old Decoder.init_weights)
        r"output\.weight": init_trunc_normal(
            std=dim**-0.5, a=-3 * dim**-0.5, b=3 * dim**-0.5
        ),
        # Default for remaining weights (wq, wk, wv, w1, etc.)
        r".*\.weight": init_trunc_normal(std=base_std),
        # Biases
        r".*\.bias": init_zeros(),
        # Catch-all for bare nn.Parameters (e.g., sinks)
        r".*": init_trunc_normal(std=base_std),
    }
