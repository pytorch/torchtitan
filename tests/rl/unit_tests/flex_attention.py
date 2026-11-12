# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.common.attention import FlexInnerAttention


class StridedFlexInnerAttention(FlexInnerAttention):
    """Exercise a custom mask that depends on document-local key positions."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        def get_mask_mod(self):
            # Three does not divide the packed-document alignment (128), so
            # using packed indices instead of document positions changes it.
            def strided_mask(b, h, q_idx, kv_idx):
                return (kv_idx % 3 == 0) | (q_idx == kv_idx)

            return strided_mask
