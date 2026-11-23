# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.observability.cuda_memory.stack import parse_stack


class TestStack(unittest.TestCase):
    def test_native_stack_fields_remain_aligned(self):
        text = (
            "frame #0: at::get_backtrace() + 0x30 (0x123 in /tmp/libtorch.so)\n"
            "<omitting python frames>\n"
            "frame #7: <unknown function> + 0x19 (0x456 in /tmp/libnccl.so.2)\n"
        )
        result = parse_stack(text, 8)
        self.assertEqual(result["native_pcs"], "291 1110")
        self.assertEqual(
            result["modules"].splitlines(), ["/tmp/libtorch.so", "/tmp/libnccl.so.2"]
        )
        self.assertEqual(result["stack_status"], "truncated")
