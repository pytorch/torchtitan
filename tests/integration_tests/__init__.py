# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    # Varlen/FA3 path activates FA3 on SM90+; skip there if flash-attn-3 is missing.
    requires_fa3: bool = False
    # --comm.mode torchcomms; skip if torchcomms is not installed.
    requires_torchcomms: bool = False
    # DeepEP / HybridEP MoE backends; skip if deep_ep is not installed.
    requires_deep_ep: bool = False
    # Float8 / MX / torchao converters; skip if torchao is not installed.
    requires_torchao: bool = False
    timeout: int | None = None

    def __repr__(self):
        return self.test_descr
