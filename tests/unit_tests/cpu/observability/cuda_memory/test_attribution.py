# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.observability.cuda_memory.attribution import (
    attribute_request,
    summarize_requests,
)


class TestAttribution(unittest.TestCase):
    def test_module_names_match_exactly(self):
        for path, expected in (
            ("/opt/libnccl.so.2", "NCCL"),
            ("/opt/libcublas.so.13", "cuBLAS"),
            ("/opt/libcublasLt.so.13", "cuBLASLt"),
            ("/opt/libcudnn_ops.so.9", "cuDNN"),
            ("/opt/libnccl.so.fake", "unknown"),
            ("/nccl/libextension.so", "unknown"),
            ("/opt/libnotnccl.so.2", "unknown"),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    attribute_request({"modules": path})["requester"], expected
                )

    def test_collector_torch_frames_are_not_allocation_evidence(self):
        report = attribute_request(
            {
                "modules": "/opt/libtorch_cpu.so\n/opt/libnccl.so.2\n/opt/libnccl.so.2",
                "symbols": "torch::get_cpp_backtrace()\n<unknown>\nncclAlloc()",
            }
        )
        self.assertEqual(report["requester"], "NCCL")
        self.assertEqual(report["requester_chain"], ["NCCL"])

    def test_torch_allocator_requires_module_and_symbol(self):
        symbol = "c10::cuda::CUDACachingAllocator::Native::alloc()"
        self.assertEqual(
            attribute_request({"modules": "/lib/libtorch_cuda.so", "symbols": symbol})[
                "requester"
            ],
            "PyTorch",
        )
        self.assertEqual(
            attribute_request({"modules": "/lib/libextension.so", "symbols": symbol})[
                "requester"
            ],
            "unknown",
        )
        self.assertEqual(
            attribute_request({"modules": "/lib/libtorch_cuda.so"})["requester"],
            "unknown",
        )

    def test_absent_or_unrecognized_stack_stays_unknown(self):
        for record in ({}, {"symbols": "ncclMalloc()"}, {"modules": "/lib/unknown.so"}):
            self.assertEqual(
                attribute_request(record),
                {
                    "requester": "unknown",
                    "requester_chain": [],
                    "evidence": "unavailable",
                },
            )

    def test_summary_keeps_api_device_and_memory_location_separate(self):
        def request(api, device=0, location=1):
            return {
                "api": api,
                "device": device,
                "location_type": location,
                "bytes": 4096,
                "attribution": {"requester": "NCCL"},
            }

        rows = summarize_requests(
            [
                request("cuMemCreate"),
                request("cuMemCreate"),
                request("cuMemAllocAsync"),
                request("cuMemCreate", device=1),
                request("cuMemCreate", device=-1, location=3),
            ]
        )
        self.assertEqual(len(rows), 4)
        backing = next(
            row for row in rows if row["api"] == "cuMemCreate" and row["device"] == 0
        )
        self.assertEqual(backing["request_bytes"], 8192)
        self.assertEqual(backing["count"], 2)
        self.assertTrue(all("live_bytes" not in row for row in rows))
