# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.protocols.model import BaseModel


def test_base_has_no_build_forward_inputs():
    assert not hasattr(BaseModel, "_build_forward_inputs")


def test_base_has_no_input_sharding():
    assert not hasattr(BaseModel, "input_sharding")
