# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from torchtitan.distributed import utils


class TestDistributedUtils(unittest.TestCase):
    def test_set_pg_timeouts_skips_fake_backend_groups(self):
        fake_group = MagicMock()
        nccl_group = MagicMock()
        parallel_dims = MagicMock()
        parallel_dims.get_all_one_dimensional_meshes.return_value = {
            "fake": MagicMock(get_group=MagicMock(return_value=fake_group)),
            "nccl": MagicMock(get_group=MagicMock(return_value=nccl_group)),
        }

        def get_backend(group):
            if group is fake_group:
                return "fake"
            return "nccl"

        with (
            patch.object(utils.device_module, "current_device", return_value=0),
            patch.object(utils.device_module, "synchronize"),
            patch.object(utils.torch.distributed, "barrier"),
            patch.object(
                utils.torch.distributed, "get_backend", side_effect=get_backend
            ),
            patch.object(
                utils.torch.distributed.distributed_c10d, "_set_pg_timeout"
            ) as set_pg_timeout,
        ):
            utils.set_pg_timeouts(timedelta(seconds=3), parallel_dims)

        set_pg_timeout.assert_any_call(timedelta(seconds=3), nccl_group)
        set_pg_timeout.assert_any_call(timedelta(seconds=3), None)
        self.assertEqual(set_pg_timeout.call_count, 2)


if __name__ == "__main__":
    unittest.main()
