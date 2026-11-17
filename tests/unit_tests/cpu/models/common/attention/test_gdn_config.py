# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.models.common.activation import SiLU
from torchtitan.models.common.attention.gdn import GDN, GDNKernel, InnerGDN
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.norm import GatedRMSNorm

_KEY_HEAD_DIM = 2
_VALUE_HEAD_DIM = 2
_CONV_KERNEL_SIZE = 4
_DIM = 8


def _matching_in_proj_out_features(*, n_heads: int, n_key_heads: int) -> int:
    """Row count ``GDN.Config`` compares against ``in_proj.out_features``.

    Same arithmetic as ``in_proj_split_sizes``, including a negative
    ``n_key_heads`` (``4 // -2 == -2``). ``n_key_heads == 0`` cannot be
    mirrored because that property divides.
    """
    heads_per_key = n_heads // n_key_heads
    value_dim = heads_per_key * _VALUE_HEAD_DIM
    split_sizes = [
        _KEY_HEAD_DIM,
        _KEY_HEAD_DIM,
        value_dim,
        value_dim,
        heads_per_key,
        heads_per_key,
    ]
    return n_key_heads * sum(split_sizes)


def _gdn_config(*, n_heads: int, n_key_heads: int) -> GDN.Config:
    if n_key_heads == 0:
        # Unused: construction must fail before the row-count check.
        in_proj_out_features = 1
    else:
        in_proj_out_features = _matching_in_proj_out_features(
            n_heads=n_heads, n_key_heads=n_key_heads
        )
    # Conv widths are not validated by GDN.Config; keep them positive.
    key_channels = 4
    value_channels = n_heads * _VALUE_HEAD_DIM

    def linear(in_features: int, out_features: int) -> Linear.Config:
        return Linear.Config(in_features=in_features, out_features=out_features)

    def conv(channels: int) -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=_CONV_KERNEL_SIZE,
            groups=channels,
            bias=False,
        )

    return GDN.Config(
        n_heads=n_heads,
        key_head_dim=_KEY_HEAD_DIM,
        value_head_dim=_VALUE_HEAD_DIM,
        n_key_heads=n_key_heads,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        in_proj=linear(_DIM, in_proj_out_features),
        conv_q=conv(key_channels),
        conv_k=conv(key_channels),
        conv_v=conv(value_channels),
        inner_attention=InnerGDN.Config(
            conv_kernel_size=_CONV_KERNEL_SIZE,
            kernel=GDNKernel.Config(),
        ),
        norm=GatedRMSNorm.Config(
            dim=_VALUE_HEAD_DIM,
            eps=1e-6,
            activation_fn=SiLU.Config(),
        ),
        out_proj=linear(value_channels, _DIM),
    )


def test_zero_n_key_heads_raises_value_error_not_zero_division() -> None:
    with pytest.raises(ValueError, match=r"n_key_heads") as raised:
        _gdn_config(n_heads=4, n_key_heads=0)
    assert "0" in str(raised.value)


def test_negative_n_key_heads_raises_value_error_when_out_features_match() -> None:
    # 4 % -2 == 0, and out_features equals the negative split's row count,
    # so the divisibility and width checks would both accept this config.
    with pytest.raises(ValueError, match=r"n_key_heads") as raised:
        _gdn_config(n_heads=4, n_key_heads=-2)
    assert "-2" in str(raised.value)


def test_divisible_positive_n_key_heads_constructs() -> None:
    config = _gdn_config(n_heads=4, n_key_heads=2)
    assert config.n_heads == 4
    assert config.n_key_heads == 2
    assert config.in_proj.out_features == config.n_key_heads * sum(
        config.in_proj_split_sizes
    )
    assert all(size > 0 for size in config.in_proj_split_sizes)
