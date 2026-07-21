# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Backend selection for the selective gather.

Three backends implement the same op:
  * ``"lsa"`` -- the CuTeDSL LSA kernels. Intra-node CP over
    NVLink; needs nccl4py + CuTeDSL + NCCL windows + a Hopper+ domain.
  * ``"gin"`` -- the CuTeDSL GIN kernel (device-initiated network put). Inter-node
    CP; same capability requirements as ``"lsa"`` plus a real RDMA fabric.
  * ``"p2p"`` -- the portable ``batch_isend_irecv`` baseline. Runs anywhere
    NCCL/RCCL P2P works (incl. AMD); the fallback.

``select_backend`` first honors ``SELECTIVE_GATHER_BACKEND``. Otherwise a
CuTeDSL backend needs all of:
  1. nccl4py >= 0.3.1
  2. NCCL   >= 2.30.7
  3. CuTeDSL (cutlass) importable
  4. device is Hopper+ (compute capability >= 9.0)
  5. that backend's kernels can run this dtype and block size
  6. every rank in the CP group agrees on 1-5

The first five are local; the sixth makes the choice group-wide, and a split
choice would deadlock the group with some ranks in a CuTeDSL kernel and the
rest in ``batch_isend_irecv``. The host set then picks the transport: ``"lsa"``
reads peers over NVLink so it only applies inside one host, ``"gin"`` pushes
over the network between them. A CP group is never a mix of the two.

Thresholds are the versions validated on the dev box (H100, nccl4py 0.3.1,
NCCL 2.30.7); bump them here if a newer minimum is required. Set env
``SELECTIVE_GATHER_BACKEND`` to force a choice, which is useful for testing a
fallback on a fully-capable box.
"""

import os
import warnings

import torch

SUPPORTED_BACKENDS = ("gin", "lsa", "p2p")

# (major, minor, patch) minimums -- validated working on the dev box.
_MIN_NCCL4PY = (0, 3, 1)
_MIN_NCCL = (2, 30, 7)
_MIN_CAPABILITY = (9, 0)  # Hopper+ (also admits later archs, e.g. Blackwell)

# What the CuTeDSL kernels can actually run. Both backwards accumulate in FP32
# registers, so they need a float cute type (lsa_kernel._CUTE_DT and
# gin_kernel._REDUCE_DT mirror this). Each tiles a block across threads with no
# tail mask, so a block must be a whole number of tiles: LSA copies 16-byte
# words over 256 threads, and GIN reuses the LSA reduce over 256 elements.
CUTEDSL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_LSA_BLOCK_ALIGN = 4096  # bytes: 16-byte words x 256 threads
_REDUCE_TILE = 256  # elements


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
    if dtype not in CUTEDSL_DTYPES or block_numel is None:
        return False
    return (block_numel * dtype.itemsize) % _LSA_BLOCK_ALIGN == 0


def gin_supports_op(dtype, block_numel) -> bool:
    """Whether the GIN kernels can run a gather of this dtype and block size.

    GIN reuses the LSA reduce in its backward, which tiles a block across
    threads with no tail mask, so the block has to be a whole number of tiles.
    """
    if dtype not in CUTEDSL_DTYPES or block_numel is None:
        return False
    return block_numel % _REDUCE_TILE == 0


def _agreed_backend(device, group, dtype, block_numel) -> str:
    """Settle the backend across the whole CP group with one collective.

    Capability, per-backend operation support, and the host set are gathered
    together, so every rank reaches the same answer. Every rank takes part
    unconditionally, carrying its own verdict: gating the collective on local
    capability would hang the capable ranks against the ones that skipped it.

    The host set picks the transport -- LSA reads peers over NVLink, so it only
    applies inside one host; GIN pushes over the network between them.
    Hostnames are assumed per-host-unique (the norm for SLURM / k8s / MAST); a
    launcher handing out non-unique ones would make an inter-node group look
    intra-node, so force ``SELECTIVE_GATHER_BACKEND`` there.

    Without a group there is nothing to agree with, so the answer is ``"p2p"``.
    """
    if group is None:
        return "p2p"
    import socket

    import torch.distributed as dist

    capable = _hopper_ok(device) and _cutlass_ok() and _nccl4py_ok() and _nccl_ok()
    verdicts = [None] * group.size()
    dist.all_gather_object(
        verdicts,
        (
            capable and lsa_supports_op(dtype, block_numel),
            capable and gin_supports_op(dtype, block_numel),
            socket.gethostname(),
        ),
        group=group,
    )
    if len({host for _, _, host in verdicts}) == 1:
        return "lsa" if all(lsa for lsa, _, _ in verdicts) else "p2p"
    return "gin" if all(gin for _, gin, _ in verdicts) else "p2p"


def select_backend(device=None, group=None, *, dtype=None, block_numel=None) -> str:
    """Return the backend for this CP group: ``"gin"`` / ``"lsa"`` / ``"p2p"``.

    ``dtype`` and ``block_numel`` describe the gather the caller will run; the
    CuTeDSL kernels reject some of them, so leaving them out means only
    ``"p2p"`` is chosen. Runs one small collective, so call it once at context
    setup.

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
    return _agreed_backend(device, group, dtype, block_numel)
