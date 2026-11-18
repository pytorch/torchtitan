# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from tests.unit_tests.cpu.components import test_chunked_loss_fsdp


@pytest.mark.multi_gpu
class TestChunkedLossFSDP(test_chunked_loss_fsdp.TestChunkedLossFSDP):
    device_type = "cuda"
