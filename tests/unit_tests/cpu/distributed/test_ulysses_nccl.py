# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed.utils import configure_nccl_for_ulysses
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import UlyssesCPFlexInnerAttention
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


class TestUlyssesNCCL(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.enterContext(
            mock.patch(
                "torchtitan.distributed.utils.dist.is_nccl_available",
                return_value=True,
            )
        )
        self.model = llama3_debugmodel(seq_len=512).model
        self.ulysses_model = ContextParallelTransform(
            inner_attention_map={FlexInnerAttention: UlyssesCPFlexInnerAttention}
        ).transform(llama3_debugmodel(seq_len=512).model)

    def configure(self, model, *, version=(2, 32, 2), **kwargs):
        kwargs = {"device_type": "cuda", "comm_backend": "default"} | kwargs
        with mock.patch(
            "torchtitan.distributed.utils.torch.cuda.nccl.version",
            return_value=version,
        ):
            configure_nccl_for_ulysses(model, **kwargs)
        return os.environ.get("NCCL_GRAPH_REGISTER")

    def test_disables_registration_before_fixed_version(self):
        self.assertEqual(self.configure(self.ulysses_model), "0")
        os.environ.clear()
        self.assertIsNone(self.configure(self.ulysses_model, version=(2, 32, 3)))

    def test_skips_non_ulysses_model(self):
        self.assertIsNone(self.configure(self.model))

    @parametrize("kwargs", ({"comm_backend": "fake"}, {"device_type": "cpu"}))
    def test_skips_unsupported_runtime(self, kwargs):
        self.assertIsNone(self.configure(self.ulysses_model, **kwargs))

    def test_skips_unavailable_nccl(self):
        with mock.patch(
            "torchtitan.distributed.utils.dist.is_nccl_available",
            return_value=False,
        ):
            self.assertIsNone(self.configure(self.ulysses_model))

    def test_overwrites_enabled_registration(self):
        os.environ["NCCL_GRAPH_REGISTER"] = "1"
        with self.assertLogs("torchtitan.distributed.utils", level="WARNING") as logs:
            self.assertEqual(self.configure(self.ulysses_model), "0")
        self.assertIn("NCCL_GRAPH_REGISTER=1 will be overridden to 0.", logs.output[0])


instantiate_parametrized_tests(TestUlyssesNCCL)

if __name__ == "__main__":
    run_tests()
