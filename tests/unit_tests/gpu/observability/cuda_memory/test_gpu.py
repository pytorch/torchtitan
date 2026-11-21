# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

if not torch.cuda.is_available() or torch.version.cuda != "13.5":
    pytest.skip("CUDA allocation capture requires CUDA 13.5", allow_module_level=True)

core = pytest.importorskip("torch.profiler._cuspy.core")
if not hasattr(core.Cuspy, "register_callback_handler"):
    pytest.skip("Cuspy callback registry is unavailable", allow_module_level=True)
driver = pytest.importorskip("cuda.bindings.driver")

from torchtitan.observability.cuda_memory import MemoryCapture


class TestAllocationCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.init()

    def call(self, function, *args):
        result, *values = function(*args)
        self.assertEqual(int(result), 0)
        return values[0] if len(values) == 1 else values

    def test_direct_and_managed_requests(self):
        with MemoryCapture(stacks=False) as capture:
            pointer = self.call(driver.cuMemAlloc, 1234567)
            self.call(driver.cuMemFree, pointer)
            pointer = self.call(
                driver.cuMemAllocManaged,
                2345678,
                driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL,
            )
            self.call(driver.cuMemFree, pointer)
        requests = capture.report()["requests"]
        self.assertTrue(
            any(r["api"] == "cuMemAlloc_v2" and r["bytes"] == 1234567 for r in requests)
        )
        self.assertTrue(
            any(
                r["api"] == "cuMemAllocManaged" and r["bytes"] == 2345678
                for r in requests
            )
        )
        self.assertTrue(all("Free" not in r["api"] for r in requests))

    def test_vmm_backing_request_without_mapping_hooks(self):
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = torch.cuda.current_device()
        size = self.call(
            driver.cuMemGetAllocationGranularity,
            prop,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        with MemoryCapture(stacks=False) as capture:
            handle = self.call(driver.cuMemCreate, size, prop, 0)
            self.call(driver.cuMemRelease, handle)
        matching = [
            r for r in capture.report()["requests"] if r["api"] == "cuMemCreate"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["bytes"], size)
        self.assertEqual(matching[0]["device"], torch.cuda.current_device())

    def test_async_and_pool_requests(self):
        stream = self.call(driver.cuStreamCreate, 0)
        pool = self.call(driver.cuDeviceGetDefaultMemPool, torch.cuda.current_device())
        try:
            with MemoryCapture(stacks=False) as capture:
                pointer = self.call(driver.cuMemAllocAsync, 3456789, stream)
                self.call(driver.cuMemFreeAsync, pointer, stream)
                pointer = self.call(
                    driver.cuMemAllocFromPoolAsync, 4567890, pool, stream
                )
                self.call(driver.cuMemFreeAsync, pointer, stream)
            # Completion is the workload's responsibility; capture only observed requests.
            self.call(driver.cuStreamSynchronize, stream)
        finally:
            self.call(driver.cuStreamDestroy, stream)
        sizes = {r["bytes"] for r in capture.report()["requests"]}
        self.assertTrue({3456789, 4567890}.issubset(sizes))

    def test_nccl_requester_from_native_stack(self):
        try:
            nccl = ctypes.CDLL("libnccl.so.2")
            allocate, free = nccl.ncclMemAlloc, nccl.ncclMemFree
        except (OSError, AttributeError):
            self.skipTest("shared NCCL with ncclMemAlloc required")
        allocate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        free.argtypes = [ctypes.c_void_p]
        pointer = ctypes.c_void_p()
        with MemoryCapture() as capture:
            self.assertEqual(allocate(ctypes.byref(pointer), 1048576), 0)
            self.assertEqual(free(pointer), 0)
        self.assertTrue(
            any(
                r["attribution"]["requester"] == "NCCL"
                for r in capture.report()["requests"]
            )
        )

    def test_startup_window_with_scheduled_cuspy(self):
        from torchtitan.observability.profiler import (
            CudaMemoryProfilerConfig,
            CuspyProfilerConfig,
            Profiler,
            record_cuda_memory,
        )

        config = Profiler.Config(
            enable_profiling=True,
            cuspy=CuspyProfilerConfig(),
            profile_freq=2,
            profiler_warmup=1,
            profiler_active=1,
            profiler_repeat=1,
            cuda_memory=CudaMemoryProfilerConfig(capture_steps=1),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            patch.object(torch.distributed, "get_rank", return_value=0),
        ):
            with record_cuda_memory(config, base_folder=directory):
                pointer = self.call(driver.cuMemAlloc, 456789)
                self.call(driver.cuMemFree, pointer)
                with config.build(base_folder=directory) as profiler:
                    for index in range(2):
                        pointer = self.call(driver.cuMemAlloc, 567890 + index)
                        self.call(driver.cuMemFree, pointer)
                        profiler.step()
            path = Path(directory) / "profiling/traces/cuda_memory/rank0/memory.json.gz"
            with gzip.open(path, "rt") as stream:
                report = json.load(stream)
            sizes = {r["bytes"] for r in report["requests"]}
            self.assertTrue({456789, 567890}.issubset(sizes))
            self.assertNotIn(567891, sizes)
            self.assertEqual(report["coverage"]["stop_reason"], "step_limit")
            self.assertTrue(
                list(
                    Path(directory).glob(
                        "profiling/traces/iteration_*/rank0_trace.json.gz"
                    )
                )
            )
