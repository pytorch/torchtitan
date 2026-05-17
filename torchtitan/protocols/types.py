# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared type primitives for torchtitan protocols.

Houses small types reused across the protocols package (and its callers)
so individual protocol modules don't grow their own copies.
"""

from enum import Enum


class StrEnum(str, Enum):
    """str + Enum for Python < 3.11 compatibility."""

    pass


class MeshAxisName(StrEnum):
    """Names for axes of a ``DeviceMesh``.

    Naming convention: throughout torchtitan code, comments, and docstrings
    we say ``axis`` for a ``DeviceMesh`` axis and ``dim`` for a tensor
    dimension. This avoids the ambiguity of ``dim`` referring to both.

    Note that PyTorch upstream's ``DeviceMesh`` API still uses the older
    ``mesh_dim_names`` attribute and ``mesh_dim`` parameter names; we keep
    those exact spellings when calling into PyTorch APIs (we cannot rename
    upstream surface), but use ``axis`` for any name we own.
    """

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    FSDP = "fsdp"
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    EFSDP = "efsdp"
