# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stack evidence for allocation requesters, without ownership or lifetime claims."""

import os
import re
from collections import defaultdict
from typing import Any

# Match full DSO basenames, not arbitrary symbols or paths containing a name.
_LIBRARIES = (
    (r"libnccl\.so(?:\.\d+)*", "NCCL"),
    (r"libcublasLt\.so(?:\.\d+)*", "cuBLASLt"),
    (r"libcublas\.so(?:\.\d+)*", "cuBLAS"),
    (r"libcudnn(?:_[a-z0-9_]+)?\.so(?:\.\d+)*", "cuDNN"),
)
_TORCH = re.compile(r"lib(?:torch(?:_cuda|_cpu|_python)?|c10(?:_cuda)?)\.so(?:\.\d+)*")
_ALLOCATORS = (
    "c10::cuda::CUDACachingAllocator::Native::",
    "c10::cuda::CUDACachingAllocator::CudaMallocAsync::",
)


def attribute_request(record: dict[str, Any]) -> dict[str, Any]:
    chain = []
    modules = record.get("modules", "").splitlines()
    symbols = record.get("symbols", "").splitlines()
    for index, module in enumerate(modules):
        name = os.path.basename(module)
        library = next(
            (label for pattern, label in _LIBRARIES if re.fullmatch(pattern, name)),
            "unknown",
        )
        # The unwinder itself adds Torch frames. Only allocator frames count.
        symbol = symbols[index] if index < len(symbols) else ""
        if _TORCH.fullmatch(name) and symbol.startswith(_ALLOCATORS):
            library = "PyTorch"
        if library != "unknown" and library not in chain:
            chain.append(library)
    return {
        "requester": chain[0] if chain else "unknown",
        "requester_chain": chain,
        "evidence": "native_stack" if chain else "unavailable",
    }


def summarize_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep API/device/location groups distinct: pool requests and VMM backing
    # are different views, and host/managed/device bytes must not become one total.
    groups = defaultdict(lambda: {"count": 0, "bytes": 0})
    for request in requests:
        key = (
            request["attribution"]["requester"],
            request["api"],
            request["device"],
            request["location_type"],
        )
        groups[key]["count"] += 1
        groups[key]["bytes"] += request["bytes"]
    return [
        {
            "requester": requester,
            "api": api,
            "device": device,
            "location_type": location,
            "count": values["count"],
            "request_bytes": values["bytes"],
        }
        for (requester, api, device, location), values in sorted(groups.items())
    ]
