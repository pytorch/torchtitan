# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse the existing PyTorch native backtrace text into frame evidence."""

import re


def parse_stack(text, maximum_frames):
    """Retain the original text and derive aligned PC/module/symbol evidence."""
    pcs, modules, symbols, indices = [], [], [], []
    for line in text.splitlines():
        index = re.match(r"frame #(\d+): ", line)
        if not index:
            continue
        indices.append(int(index[1]))
        frame = re.search(r"\((0x[0-9a-fA-F]+) in (.+)\)$", line)
        if frame:
            pcs.append(str(int(frame[1], 16)))
            modules.append(frame[2])
            symbol = line[index.end() : frame.start()].strip()
            symbols.append(re.sub(r"\s*\+ 0x[0-9a-fA-F]+\s*$", "", symbol))
        else:
            address = re.search(r"\[(0x[0-9a-fA-F]+)\]$", line)
            pcs.append(str(int(address[1], 16)) if address else "0")
            modules.append("")
            symbols.append("")
    truncated = bool(indices and max(indices) >= maximum_frames - 1)
    return {
        "native_pcs": " ".join(pcs),
        "modules": "\n".join(modules),
        "symbols": "\n".join(symbols),
        "stack_frames": len(pcs),
        "stack_status": "truncated"
        if truncated
        else "unwound"
        if pcs
        else "unavailable",
        "module_identity_stable": True,
    }
