# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Backend selection for the selective gather.

Two backends implement the same op:
  * ``"lsa"`` -- the CuTeDSL LSA kernels. Intra-node CP over
    NVLink; needs nccl4py + CuTeDSL + NCCL windows + a Hopper+ domain.
  * ``"p2p"`` -- the portable ``batch_isend_irecv`` baseline. Runs anywhere
    NCCL/RCCL P2P works (incl. AMD); the fallback.

``select_backend`` returns ``"lsa"`` only when ALL of these hold, else ``"p2p"``:
  1. nccl4py >= 0.3.1
  2. NCCL   >= 2.30.7
  3. CuTeDSL (cutlass) importable
  4. device is Hopper+ (compute capability >= 9.0)
  5. the LSA kernels can run this dtype and block size
  6. every rank in the CP group agrees on 1-5 and they share one host

The first five are local; the sixth makes the choice group-wide. Both matter:
LSA reads peer shards over NVLink, so it is wrong across nodes, and a split
choice deadlocks the group with some ranks in the LSA kernel and the rest in
``batch_isend_irecv``.

Thresholds are the versions validated on the dev box (H100, nccl4py 0.3.1,
NCCL 2.30.7). Set env ``SELECTIVE_GATHER_BACKEND`` to force a choice, which is
useful for testing the fallback on a fully-capable box.
"""

import os
import warnings

import torch

SUPPORTED_BACKENDS = ("lsa", "p2p")

# (major, minor, patch) minimums -- validated working on the dev box.
_MIN_NCCL4PY = (0, 3, 1)
_MIN_NCCL = (2, 30, 7)
_MIN_CAPABILITY = (9, 0)  # Hopper+ (also admits later archs, e.g. Blackwell)

# What the LSA kernels can actually run. The backward accumulates in FP32
# registers, so it needs a float cute type (lsa_kernel._CUTE_DT mirrors this).
# The vectorized copy tiles a block over 256 threads of 16-byte words with no
# tail mask, so a block must be a whole number of those tiles.
LSA_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_LSA_BLOCK_ALIGN = 4096  # 16-byte words x 256 threads


def _parse_version(s: str) -> tuple[int, ...]:
    parts = []
    for tok in str(s).split("."):
        num = "".join(c for c in tok if c.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _nccl4py_ok() -> bool:
    try:
        import importlib.metadata as md

        if _parse_version(md.version("nccl4py")) < _MIN_NCCL4PY:
            return False
        # Real import, not just metadata: an installed-but-broken nccl4py (bad
        # native lib, or a cutlass shim gap) must fall back to "p2p", not
        # hard-crash on the first lazy import in the kernel path.
        import nccl.core.device.cute  # noqa: F401

        return True
    except Exception:
        return False


def _nccl_ok() -> bool:
    try:
        return tuple(torch.cuda.nccl.version()) >= _MIN_NCCL
    except Exception:
        return False


def _cutlass_ok() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        return True
    except Exception:
        return False


def _hopper_ok(device) -> bool:
    try:
        return torch.cuda.get_device_capability(device) >= _MIN_CAPABILITY
    except Exception:
        return False


def lsa_supports_op(dtype, block_numel) -> bool:
    """Whether the LSA kernels can run a gather of this dtype and block size.

    They raise on anything else, so selection has to ask first: P2P handles the
    same op and is the right choice there.
    """
    if dtype not in LSA_DTYPES or block_numel is None:
        return False
    return (block_numel * dtype.itemsize) % _LSA_BLOCK_ALIGN == 0


def _lsa_domain_ok(device, group, dtype, block_numel) -> bool:
    """Whether every rank can run this op with LSA and they share one host.

    One ``all_gather_object`` settles capability, operation support, and the
    host set together. Every rank takes part unconditionally, carrying its own
    verdict: gating the collective on local capability would hang the capable
    ranks against the ones that skipped it.

    Uses ``socket.gethostname()``, so it assumes per-host-unique hostnames (the
    norm for SLURM / k8s / MAST). A launcher handing out non-unique hostnames
    would make an inter-node group look intra-node; force
    ``SELECTIVE_GATHER_BACKEND=p2p`` there.

    Without a group the domain cannot be established, so the answer is no.
    """
    if group is None:
        return False
    import socket

    import torch.distributed as dist

    mine = (
        _hopper_ok(device)
        and _cutlass_ok()
        and _nccl4py_ok()
        and _nccl_ok()
        and lsa_supports_op(dtype, block_numel)
    )
    verdicts = [None] * group.size()
    dist.all_gather_object(verdicts, (bool(mine), socket.gethostname()), group=group)
    return all(ok for ok, _ in verdicts) and len({host for _, host in verdicts}) == 1


def select_backend(device=None, group=None, *, dtype=None, block_numel=None) -> str:
    """Return ``"lsa"`` when the whole CP group can run this op, else ``"p2p"``.

    ``dtype`` and ``block_numel`` describe the gather the caller will run; the
    LSA kernels reject some of them, so leaving them out means ``"lsa"`` is
    never chosen. Runs one small collective, so call it once at context setup.

    ``SELECTIVE_GATHER_BACKEND`` overrides the check. An unsupported value warns
    instead of silently taking no effect.
    """
    forced = os.environ.get("SELECTIVE_GATHER_BACKEND")
    if forced in SUPPORTED_BACKENDS:
        return forced
    if forced is not None:
        warnings.warn(
            f"SELECTIVE_GATHER_BACKEND={forced!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_BACKENDS)}); auto-selecting.",
            stacklevel=2,
        )
    return "lsa" if _lsa_domain_ok(device, group, dtype, block_numel) else "p2p"
