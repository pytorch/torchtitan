# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DCP async_save writes uninitialized memory for TorchTitan's quantized-FSDP wrapper parameters.

TorchTitan's NVFP4 and MXFP8 Linear weights are `_ShardedFSDPTensor` wrapper subclasses
(`torch.Tensor._make_wrapper_subclass`): they have no storage of their own (data_ptr() == 0) and
keep the high-precision shard in `self._tensor`. `dcp.async_save` stages the state dict through
`StateDictStager._offload_tensor`, which for such a tensor:

1. calls `x.new_empty(x.size(), device="cpu")`. The wrapper does not preserve `aten.new_empty`
   (it is missing from `_FSDP_SHARDED_OPS`), so this returns a plain, uninitialized tensor;
2. skips the storage copy, because `x.data_ptr() == 0`;
3. copies `x.__dict__`, so the real values land in an unused `y._tensor` attribute.

The writer then saves the uninitialized tensor. `dcp.save` does not stage and saves correctly.
The fix adds `aten.new_empty` to `_FSDP_SHARDED_OPS`: the staged copy stays a wrapper whose
`_tensor` holds the staged values, and the async round trip is exact.

Usage: python scripts/repro_async_save_fsdp_wrapper.py
           [--without-fix | --with-fix]
By default it runs both cases, removing then adding the op in memory, and exits successfully
when only the unfixed async round trip loses values. Use either flag to run just that case;
a single case exits unsuccessfully if either round trip loses values. No source files change.
Needs torch and torchtitan; runs on CPU in one process.
"""

import argparse
import sys
import tempfile

import torch
import torch.distributed.checkpoint as dcp
from torchtitan.quantization import _fsdp_tensor
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor


class Wrapped(_ShardedFSDPTensor):
    """The wrapper NVFP4/MXFP8 Linear weights subclass; its operands are unused here."""

    def _build_operands(self, logical_tensor, out=None):
        raise NotImplementedError


def round_trip(save, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Distinct values per round trip: the staged copy is uninitialized memory, which the
    # allocator may refill from an earlier round trip's freed tensor of equal values.
    values = torch.randn(256, 128, generator=torch.Generator().manual_seed(seed))
    param = torch.nn.Parameter(Wrapped(values.clone()))
    with tempfile.TemporaryDirectory() as directory:
        result = save({"w": param}, checkpoint_id=directory, no_dist=True)
        if hasattr(result, "result"):  # async_save returns a future
            result.result()
        loaded = {"w": torch.zeros_like(values)}
        dcp.load(loaded, checkpoint_id=directory, no_dist=True)
    return values, loaded["w"]


parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--without-fix", dest="fixed", action="store_const", const=False)
mode.add_argument("--with-fix", dest="fixed", action="store_const", const=True)
args = parser.parse_args()
cases = (False, True) if args.fixed is None else (args.fixed,)
NEW_EMPTY = torch.ops.aten.new_empty.default
installed_fixed = NEW_EMPTY in _fsdp_tensor._FSDP_SHARDED_OPS
print(
    f"torch {torch.__version__}; installed wrapper preserves aten.new_empty: {installed_fixed}"
)
print(f"{'Case':<14} {'new_empty() type':<18} {'Sync save/load':<18} Async save/load")
success = True
try:
    for index, fixed in enumerate(cases):
        if fixed:
            _fsdp_tensor._FSDP_SHARDED_OPS.add(NEW_EMPTY)
        else:
            _fsdp_tensor._FSDP_SHARDED_OPS.discard(NEW_EMPTY)
        new_empty_type = type(Wrapped(torch.zeros(1)).new_empty((1,), device="cpu"))
        expected, loaded = round_trip(dcp.async_save, seed=2 * index)
        sync_expected, sync_loaded = round_trip(dcp.save, seed=2 * index + 1)
        ok_async = torch.equal(loaded, expected)
        ok_sync = torch.equal(sync_loaded, sync_expected)
        label = "with fix" if fixed else "without fix"
        print(
            f"{label:<14} {new_empty_type.__name__:<18} "
            f"{'EXACT' if ok_sync else 'CORRUPTED':<18} "
            f"{'EXACT' if ok_async else 'CORRUPTED'}",
            flush=True,
        )
        if not ok_async:
            print(f"  Expected first values: {expected.flatten()[:3].tolist()}")
            print(f"  Loaded first values:   {loaded.flatten()[:3].tolist()}")
        # In comparison mode the unfixed corruption is the expected reproduction.
        expected_async = fixed if args.fixed is None else True
        success &= ok_sync and ok_async == expected_async
        success &= issubclass(new_empty_type, Wrapped) == fixed
finally:
    if installed_fixed:
        _fsdp_tensor._FSDP_SHARDED_OPS.add(NEW_EMPTY)
    else:
        _fsdp_tensor._FSDP_SHARDED_OPS.discard(NEW_EMPTY)

if args.fixed is None:
    print(
        "Expected before/after behavior reproduced."
        if success
        else "Unexpected result."
    )
sys.exit(0 if success else 1)
