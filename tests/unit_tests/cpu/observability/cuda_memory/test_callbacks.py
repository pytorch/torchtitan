# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torchtitan.observability.cuda_memory import callbacks as module


class TestAllocationCallbacks(unittest.TestCase):
    def setUp(self):
        self.result = ctypes.c_int32(0)
        self.header = SimpleNamespace(
            callback_site=2,
            function_return_value=ctypes.addressof(self.result),
            function_params=123,
            context_uid=7,
            correlation_id=11,
        )
        self.cupti = SimpleNamespace(
            CallbackData=SimpleNamespace(from_ptr=MagicMock(return_value=self.header)),
            ApiCallbackSite=SimpleNamespace(API_EXIT=2),
            CallbackDomain=SimpleNamespace(DRIVER_API=1),
            DriverApiTraceCbid=SimpleNamespace(),
        )
        for ident, (name, parameter, _, _) in enumerate(module._APIS, start=100):
            setattr(self.cupti.DriverApiTraceCbid, name, ident)
            setattr(self.cupti, parameter, SimpleNamespace(from_ptr=MagicMock()))
        self.driver = SimpleNamespace(
            CUmemAllocationProp_st=MagicMock(),
            CUmemLocationType=SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=1),
        )
        self.versions = {"cupti-python": "13.5.0", "cuda-bindings": "13.5.0"}
        self.enterContext(
            patch.object(
                module, "_load_bindings",
                return_value=(self.cupti, self.driver, self.versions),
            )
        )
        self.decoder = module.AllocationCallbacks()

    def decode(self, name, parameter, **fields):
        wrapper = getattr(self.cupti, parameter).from_ptr
        wrapper.return_value = SimpleNamespace(**fields)
        record = self.decoder.decode(1, self.decoder.info["apis"][name], 456)
        wrapper.assert_called_with(123, readonly=True)
        return record

    def test_direct_and_managed_outputs_are_copied(self):
        pointer = ctypes.c_uint64(0xFEDCBA9876543210)
        for name, parameter, operation in (
            ("cuMemAlloc_v2", "CuMemAllocV2Params", "alloc"),
            ("cuMemAllocManaged", "CuMemAllocManagedParams", "managed_alloc"),
        ):
            with self.subTest(name=name):
                record = self.decode(
                    name, parameter, dptr=ctypes.addressof(pointer), bytesize=4096
                )
                self.assertEqual(record["address"], pointer.value)
                self.assertEqual(record["bytes"], 4096)
                self.assertEqual(record["operation"], operation)
                self.assertEqual((record["context"], record["correlation"]), (7, 11))
                self.assertTrue(
                    all(type(value) in (str, int) for value in record.values())
                )
        pointer.value = 0
        self.assertEqual(record["address"], 0xFEDCBA9876543210)

    def test_legacy_pointer_and_pitch_read_only_32_bits(self):
        pointer = (ctypes.c_uint32 * 2)(0xF1234567, 0xDEADBEEF)
        pitch = (ctypes.c_uint32 * 2)(512, 0xDEADBEEF)
        for name, parameter in (
            ("cuMemAlloc", "CuMemAllocParams"),
            ("cuMemAllocPitch", "CuMemAllocPitchParams"),
        ):
            with self.subTest(name=name):
                record = self.decode(
                    name,
                    parameter,
                    dptr=ctypes.addressof(pointer),
                    bytesize=1536,
                    p_pitch=ctypes.addressof(pitch),
                    height=3,
                )
                self.assertEqual(record["address"], 0xF1234567)
                self.assertEqual(record["bytes"], 1536)

    def test_modern_pitch_uses_returned_size_t_not_requested_width(self):
        pointer, pitch = ctypes.c_uint64(42), ctypes.c_size_t(1 << 33)
        record = self.decode(
            "cuMemAllocPitch_v2",
            "CuMemAllocPitchV2Params",
            dptr=ctypes.addressof(pointer),
            p_pitch=ctypes.addressof(pitch),
            height=3,
            width_in_bytes=100,
        )
        self.assertEqual(record["bytes"], 3 << 33)

    def test_async_stream_and_pool_are_values_not_pointers(self):
        pointer = ctypes.c_uint64(42)
        for name, parameter in (
            ("cuMemAllocAsync", "CuMemAllocAsyncParams"),
            ("cuMemAllocAsync_ptsz", "CuMemAllocAsyncPtszParams"),
            ("cuMemAllocFromPoolAsync", "CuMemAllocFromPoolAsyncParams"),
            ("cuMemAllocFromPoolAsync_ptsz", "CuMemAllocFromPoolAsyncPtszParams"),
        ):
            with self.subTest(name=name):
                record = self.decode(
                    name,
                    parameter,
                    dptr=ctypes.addressof(pointer),
                    bytesize=8192,
                    h_stream=0xFEED,
                    pool=0xCAFE,
                )
                self.assertEqual(record["stream"], 0xFEED)
                self.assertEqual(record["bytes"], 8192)
                if "FromPool" in name:
                    self.assertEqual(record["handle"], 0xCAFE)
                else:
                    self.assertNotIn("handle", record)

    def test_vmm_properties_use_vendor_wrapper_and_preserve_location(self):
        handle = ctypes.c_uint64(0xFEDCBA9876543210)
        for location, expected_device in ((1, 3), (2, -1)):
            self.driver.CUmemAllocationProp_st.return_value = SimpleNamespace(
                location=SimpleNamespace(type=location, id=3)
            )
            record = self.decode(
                "cuMemCreate",
                "CuMemCreateParams",
                prop=789,
                handle=ctypes.addressof(handle),
                size_=65536,
            )
            self.driver.CUmemAllocationProp_st.assert_called_with(_ptr=789)
            self.assertEqual(record["handle"], handle.value)
            self.assertEqual(record["bytes"], 65536)
            self.assertEqual(record["location_type"], location)
            self.assertEqual(record["device"], expected_device)

    def test_entry_failure_and_unselected_events_never_read_arguments(self):
        ident = self.decoder.info["apis"]["cuMemAlloc_v2"]
        self.header.function_params = 0
        self.header.function_return_value = 0
        self.header.callback_site = 1
        self.assertIsNone(self.decoder.decode(1, ident, 456))
        self.header.callback_site = 2
        self.result.value = 2
        self.header.function_return_value = ctypes.addressof(self.result)
        self.assertIsNone(self.decoder.decode(1, ident, 456))
        self.assertIsNone(self.decoder.decode(2, ident, 0))
        self.assertIsNone(self.decoder.decode(1, -1, 0))
        self.cupti.CuMemAllocV2Params.from_ptr.assert_not_called()

    def test_null_callback_result_arguments_and_outputs_are_errors(self):
        ident = self.decoder.info["apis"]["cuMemAlloc_v2"]
        with self.assertRaisesRegex(ValueError, "callback data"):
            self.decoder.decode(1, ident, 0)
        for field in ("function_return_value", "function_params"):
            with self.subTest(field=field), patch.object(self.header, field, 0):
                with self.assertRaises(ValueError):
                    self.decoder.decode(1, ident, 456)
        with self.assertRaisesRegex(ValueError, "scalar output"):
            self.decode("cuMemAlloc_v2", "CuMemAllocV2Params", dptr=0, bytesize=1)
        with self.assertRaisesRegex(ValueError, "properties"):
            self.decode("cuMemCreate", "CuMemCreateParams", prop=0)

    def test_unsupported_bindings_and_missing_types_fail_before_registration(self):
        with patch.dict(self.versions, {"cupti-python": "13.4.0"}):
            with self.assertRaisesRegex(RuntimeError, "13.5 only"):
                module.AllocationCallbacks()
        del self.cupti.CuMemCreateParams
        with self.assertRaisesRegex(RuntimeError, "required allocation callback type"):
            module.AllocationCallbacks()
