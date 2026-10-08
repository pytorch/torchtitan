# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

from torch.numa.binding import AffinityMode

from torchtitan.distributed import maybe_apply_numa_binding


def test_numa_binding_skips_non_cuda_devices() -> None:
    with patch(
        "torch.numa.binding._maybe_apply_numa_binding_to_current_process"
    ) as bind:
        maybe_apply_numa_binding(0, "cpu")

    bind.assert_not_called()


def test_numa_binding_uses_cuda_device_node() -> None:
    with patch(
        "torch.numa.binding._maybe_apply_numa_binding_to_current_process"
    ) as bind:
        maybe_apply_numa_binding(3, "cuda")

    assert bind.call_args.kwargs["device_index"] == 3
    numa_options = bind.call_args.kwargs["numa_options"]
    assert numa_options.affinity_mode == AffinityMode.NODE
    assert numa_options.should_fall_back_if_binding_fails
