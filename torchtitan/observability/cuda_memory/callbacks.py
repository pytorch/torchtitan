# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Copy successful allocation exits using NVIDIA's parameter wrappers.

Only call ``decode`` from a live Cuspy callback. Vendor wrappers own the struct
layouts; ctypes reads only pointed-to scalar outputs. No wrapper escapes decode.
"""

import ctypes
import sys
from importlib.metadata import version

# API name, NVIDIA parameter class, request kind, pointed-to output scalar.
# Legacy cuMemAlloc/cuMemAllocPitch outputs remain 32-bit on 64-bit hosts.
_APIS = (
    ("cuMemAlloc_v2", "CuMemAllocV2Params", "alloc", ctypes.c_uint64),
    ("cuMemAllocPitch_v2", "CuMemAllocPitchV2Params", "alloc", ctypes.c_uint64),
    ("cuMemAllocManaged", "CuMemAllocManagedParams", "managed_alloc", ctypes.c_uint64),
    ("cuMemCreate", "CuMemCreateParams", "create", ctypes.c_uint64),
    ("cuMemAllocAsync", "CuMemAllocAsyncParams", "alloc_async", ctypes.c_uint64),
    ("cuMemAllocAsync_ptsz", "CuMemAllocAsyncPtszParams", "alloc_async", ctypes.c_uint64),
    (
        "cuMemAllocFromPoolAsync",
        "CuMemAllocFromPoolAsyncParams",
        "alloc_async",
        ctypes.c_uint64,
    ),
    (
        "cuMemAllocFromPoolAsync_ptsz",
        "CuMemAllocFromPoolAsyncPtszParams",
        "alloc_async",
        ctypes.c_uint64,
    ),
    ("cuMemAlloc", "CuMemAllocParams", "alloc", ctypes.c_uint32),
    ("cuMemAllocPitch", "CuMemAllocPitchParams", "alloc", ctypes.c_uint32),
)


def _load_bindings():
    try:
        from cuda.bindings import driver
        from cupti import cupti

        versions = {name: version(name) for name in ("cupti-python", "cuda-bindings")}
    except ImportError as error:
        raise RuntimeError(
            "CUDA allocation capture requires cupti-python~=13.5.0 and "
            "cuda-bindings~=13.5.0"
        ) from error
    return cupti, driver, versions


def _read(pointer, scalar):
    if not pointer:
        raise ValueError("missing CUDA callback scalar output")
    return scalar.from_address(pointer).value


class AllocationCallbacks:
    """Selected driver callbacks for the initial CUDA/CUPTI 13.5 contract."""

    def __init__(self):
        if (
            sys.platform != "linux"
            or sys.byteorder != "little"
            or ctypes.sizeof(ctypes.c_void_p) != 8
        ):
            raise RuntimeError("allocation callbacks require 64-bit little-endian Linux")
        cupti, driver, versions = _load_bindings()
        if any(value.split(".")[:2] != ["13", "5"] for value in versions.values()):
            raise RuntimeError(
                "allocation callbacks support NVIDIA Python bindings 13.5 only"
            )
        try:
            self._header = cupti.CallbackData.from_ptr
            self._exit = int(cupti.ApiCallbackSite.API_EXIT)
            self._property = driver.CUmemAllocationProp_st
            self._device = int(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE)
            self.domain = int(cupti.CallbackDomain.DRIVER_API)
            self._callbacks = {
                int(getattr(cupti.DriverApiTraceCbid, name)): (
                    name,
                    getattr(cupti, parameter).from_ptr,
                    operation,
                    scalar,
                )
                for name, parameter, operation, scalar in _APIS
            }
        except AttributeError as error:
            raise RuntimeError(
                "NVIDIA bindings lack a required allocation callback type"
            ) from error
        self.info = {
            "domain": self.domain,
            "binding_versions": versions,
            "apis": {entry[0]: ident for ident, entry in self._callbacks.items()},
        }

    def decode(self, domain, ident, pointer):
        """Return copied Python values, or None for entry/failure/unselected calls."""
        if domain != self.domain or ident not in self._callbacks:
            return None
        if not pointer:
            raise ValueError("missing CUPTI callback data")
        callback = self._header(pointer, readonly=True)
        if callback.callback_site != self._exit:
            return None
        # Outputs are undefined on failure; filter before reading any arguments.
        if _read(callback.function_return_value, ctypes.c_int32) != 0:
            return None
        if not callback.function_params:
            raise ValueError("missing CUDA callback arguments")
        name, wrap, operation, scalar = self._callbacks[ident]
        params = wrap(callback.function_params, readonly=True)
        record = {
            "api": name,
            "operation": operation,
            "cbid": ident,
            "result": 0,
            "context": int(callback.context_uid),
            "correlation": int(callback.correlation_id),
            "device": -1,
            "location_type": -1,
        }
        if operation == "create":
            if not params.prop:
                raise ValueError("missing CUDA allocation properties")
            prop = self._property(_ptr=params.prop)
            record["handle"] = _read(params.handle, ctypes.c_uint64)
            record["bytes"] = int(params.size_)
            record["location_type"] = int(prop.location.type)
            if record["location_type"] == self._device:
                record["device"] = int(prop.location.id)
        else:
            record["address"] = _read(params.dptr, scalar)
            if "Pitch" in name:
                record["bytes"] = _read(params.p_pitch, scalar) * int(params.height)
            else:
                record["bytes"] = int(params.bytesize)
            if operation == "alloc_async":
                record["stream"] = int(params.h_stream)
                if "FromPool" in name:
                    record["handle"] = int(params.pool)
        return record
