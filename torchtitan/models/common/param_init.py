# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter initialization utilities and factories for torchtitan models.

Provides ``RegexInitializer`` for regex-based FQN dispatch,
``DepthScaledTruncNormal`` for depth-scaled init, and
``make_decoder_param_init`` for common decoder patterns.

Example usage in a model's ``__init__.py``::

    from torchtitan.models.common.param_init import (
        RegexInitializer,
        make_decoder_param_init,
    )

    cfg = Llama3Model.Config(
        ...,
        param_init=RegexInitializer(
            make_decoder_param_init(dim=256, n_layers=6)
        ),
    )
"""

import re
from functools import partial

import torch.nn as nn

from torchtitan.protocols.module import NamedParamInitializer, ParamInitializer

# No-op initializer: explicitly skip initialization for a parameter.
# Useful when a parameter is tied to another (e.g., weight tying).
SKIP_PARAM_INIT: ParamInitializer = lambda param: None


class RegexInitializer(NamedParamInitializer):
    """Regex-match parameter FQNs to initializers.

    First matching pattern wins. Raises ``ValueError`` if no pattern matches.
    Note that Python dictionary is ordered.

    Dict values can be either:
    - ``ParamInitializer`` (``Callable[[nn.Parameter], None]``):
      e.g. ``nn.init.zeros_``, ``partial(nn.init.normal_, std=0.02)``
    - ``NamedParamInitializer``: for init functions that need the FQN
      (e.g. ``DepthScaledTruncNormal``)
    """

    def __init__(
        self,
        rules: dict[str, ParamInitializer | NamedParamInitializer],
    ) -> None:
        self._compiled = [
            (re.compile(pattern), init_fn) for pattern, init_fn in rules.items()
        ]

    def __call__(self, name: str, param: nn.Parameter) -> None:
        for pattern, init_fn in self._compiled:
            if pattern.fullmatch(name):
                if isinstance(init_fn, NamedParamInitializer):
                    init_fn(name, param)
                else:
                    init_fn(param)
                return
        raise ValueError(f"No initializer matched '{name}'.")


class DepthScaledTruncNormal(NamedParamInitializer):
    """Truncated normal with depth-dependent std.

    Parses ``layer_id`` from the FQN (e.g., ``layers.5.attention.wo.weight``)
    and scales std by ``1 / sqrt(2 * (layer_id + 1))``.
    """

    def __init__(
        self,
        *,
        base_std: float = 0.02,
        n_layers: int,
        a: float = -2.0,
        b: float = 2.0,
    ) -> None:
        self._base_std = base_std
        self._n_layers = n_layers
        self._a = a
        self._b = b
        self._layer_re = re.compile(r"layers\.(\d+)\.")

    def __call__(self, name: str, param: nn.Parameter) -> None:
        m = self._layer_re.match(name)
        assert m is not None, (
            f"Could not parse layer_id from FQN '{name}'. "
            f"Expected FQN to contain 'layers.<digit>.'."
        )
        layer_id = int(m.group(1))
        std = self._base_std / (2 * (layer_id + 1)) ** 0.5
        nn.init.trunc_normal_(param, mean=0.0, std=std, a=self._a, b=self._b)


def make_decoder_param_init(
    *,
    dim: int,
    n_layers: int,
    base_std: float = 0.02,
    tok_emb_std: float = 1.0,
) -> dict[str, ParamInitializer | NamedParamInitializer]:
    """Common param_init patterns for Decoder-based models.

    Covers Llama3, Llama4, Qwen3, and DeepSeek V3 (with model-specific
    extensions merged via dict update). See inline comments for pattern details.
    """
    depth_std = DepthScaledTruncNormal(base_std=base_std, n_layers=n_layers)
    final_out_std = dim**-0.5
    return {
        # Token embeddings
        r"tok_embeddings\.weight": partial(nn.init.normal_, std=tok_emb_std),
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
        r".*norm.*\.weight": nn.init.ones_,
        # Output projection
        r"output\.weight": partial(
            nn.init.trunc_normal_,
            std=final_out_std,
            a=-3 * final_out_std,
            b=3 * final_out_std,
        ),
        # Default for remaining weights (wq, wk, wv, w1, etc.)
        r".*\.weight": partial(nn.init.trunc_normal_, std=base_std),
        # Biases
        r".*\.bias": nn.init.zeros_,
        # Catch-all for bare nn.Parameters (e.g., sinks)
        r".*": partial(nn.init.trunc_normal_, std=base_std),
    }
