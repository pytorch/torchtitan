# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static D2H/H2D transfer-time estimator for a joint fwd+loss+bwd FX graph.

The transfer-time analogue of ``memory_estimator.py``. Where that tracks storage
liveness, this one walks the
joint graph and, for every node, estimates the time to move its tensors across
the host<->device link: a node's *inputs* must be RELOADED (H2D) before it runs,
and its *outputs* would be OFFLOADED (D2H) after it produces them. It is the cost
model an activation-offload solver uses to decide what fits in the offload
bandwidth budget.

Each ``TensorObject`` carries the underlying ``storage_id`` and the producer's
``consumer_nodes`` so the downstream solver can dedup: tensors that are
views/aliases share a storage and need to move only once, and a tensor with
multiple consumers is reloaded only once. Sizes use the underlying storage's
bytes (a view offloads its whole storage), not the logical tensor size.

Bandwidth: ``estimate_transfertime`` accepts explicit ``bw_h2d`` / ``bw_d2h``
(GB/s); when omitted it calls ``get_transfer_bw``, which benchmarks the
host<->device link once per device and caches the result for the process (the
link bandwidth -- GPU + PCIe/NVLink topology + NUMA binding -- does not change
during a run). Pass the bandwidths in (e.g. in tests) to skip the live
benchmark; call ``measure_transfer_bw`` directly to force a fresh measurement.
"""

import functools
import time
from dataclasses import dataclass, field

import torch
import torch.utils._pytree as pytree

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node


def measure_transfer_bw(
    nbytes: int = 512 * 1024 * 1024,
    iters: int = 20,
    pinned: bool = True,
    dev: str = "cuda:0",
) -> dict:
    """Measure H2D and D2H bandwidth (GB/s) with pinned host memory.

    Pageable host memory is much slower, so pinned is the default and the regime
    an offload path would actually use.

    The default 512 MB saturates the link. A single tensor smaller than that may
    not reach this bandwidth, so the roofline over-predicts throughput for tiny
    offloads; in practice an offload solver should bundle small tensors to
    saturate the link (or treat sub-link-saturating sizes as launch-bound).
    """
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    def time_copy(dst, src):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    # warmup
    gpu.copy_(host, non_blocking=True)
    host.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize()

    h2d_s = time_copy(gpu, host)  # host -> device
    d2h_s = time_copy(host, gpu)  # device -> host
    return {"h2d": nbytes / h2d_s / 1e9, "d2h": nbytes / d2h_s / 1e9}


# Cache the immutable (h2d, d2h) pair keyed by the args that actually change the
# result: the device (its host link + NUMA binding) and the measurement regime
# (size / iters / pinned). All are fixed within a training process, so the
# benchmark runs once per distinct key. lru_cache caches a tuple (hashable +
# immutable) so a caller cannot mutate the shared result.
@functools.lru_cache(maxsize=None)
def _cached_transfer_bw(
    dev: str, nbytes: int, iters: int, pinned: bool
) -> tuple[float, float]:
    bw = measure_transfer_bw(nbytes=nbytes, iters=iters, pinned=pinned, dev=dev)
    return bw["h2d"], bw["d2h"]


def get_transfer_bw(
    dev: str = "cuda:0",
    nbytes: int = 512 * 1024 * 1024,
    iters: int = 20,
    pinned: bool = True,
) -> dict:
    """Measured ``{h2d, d2h}`` GB/s for ``dev``, benchmarked once per device and
    cached for the process lifetime (see ``_cached_transfer_bw``).

    Returns a fresh dict each call so callers may mutate it safely. Use
    ``measure_transfer_bw`` directly to bypass the cache and force a fresh run.
    """
    h2d, d2h = _cached_transfer_bw(dev, nbytes, iters, pinned)
    return {"h2d": h2d, "d2h": d2h}


def _transfer_ms(nbytes: int, bw_gbps: float) -> float:
    """Transfer time in ms for ``nbytes`` at ``bw_gbps`` GB/s.
    GB/s -> bytes/ms is a factor of 1e6 (1e9 bytes/s / 1e3 ms/s)."""
    assert nbytes >= 0, f"negative bytes: {nbytes}"
    return nbytes / (bw_gbps * 1e6)


def _storage_bytes_and_id(t: torch.Tensor) -> tuple[int, int]:
    """Underlying storage size (bytes) and its id. The id (``_cdata``) identifies
    views/aliases (they share a storage) but is only unique among *live* tensors:
    the allocator reuses it after a storage frees, so dedup within a liveness
    window, not globally."""
    st = t.untyped_storage()
    return st.nbytes(), st._cdata


class TensorObject:
    """One tensor moved across the host<->device link, with its transfer time.

    ``size`` is the underlying storage's bytes (a view offloads its whole
    storage). ``storage_id`` lets a downstream solver merge views/aliases so a
    storage moves once; ``consumer_nodes`` lists every reader of the producer so a
    multi-consumer tensor is reloaded once.
    """

    def __init__(self, tensor_type, shape, dtype, device, size, storage_id, time_ms):
        self.tensor_type = tensor_type  # "forward" | "backward" (producer region)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.size = size  # underlying storage bytes
        self.storage_id = storage_id
        self.offload_time_ms = time_ms
        self.consumer_nodes = []

    def __repr__(self):
        return (
            f"TensorObject(type={self.tensor_type}, shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device}, size={self.size}, "
            f"storage_id={self.storage_id})"
        )


def categorize_fwd_bwd(node: torch.fx.Node) -> str:
    return "backward" if _is_backward_node(node) else "forward"


def _max_offload(d: dict) -> float:
    return max((t.offload_time_ms for lst in d.values() for t in lst), default=0.0)


@dataclass
class TransferEstimatorResult:
    # node name -> list[TensorObject]:
    #   inputs  = tensors this node would RELOAD  (H2D)
    #   outputs = tensors this node would OFFLOAD (D2H)
    input_tensor_transfer_times_ms: dict = field(default_factory=dict)
    output_tensor_transfer_times_ms: dict = field(default_factory=dict)

    def max_input_offload_time(self) -> float:
        return _max_offload(self.input_tensor_transfer_times_ms)

    def max_output_offload_time(self) -> float:
        return _max_offload(self.output_tensor_transfer_times_ms)

    def summary(self, top_k: int = 10) -> str:
        if (
            not self.output_tensor_transfer_times_ms
            and not self.input_tensor_transfer_times_ms
        ):
            return "No tensor data found"
        lines = [
            f"max input  reload  (H2D) time: {self.max_input_offload_time():.3f} ms",
            f"max output offload (D2H) time: {self.max_output_offload_time():.3f} ms",
        ]
        # rank nodes by their heaviest produced tensor
        ranked = sorted(
            self.output_tensor_transfer_times_ms.items(),
            key=lambda kv: max((t.offload_time_ms for t in kv[1]), default=0.0),
            reverse=True,
        )
        for name, lst in ranked[:top_k]:
            t = max(lst, key=lambda t: t.offload_time_ms, default=None)
            if t is not None:
                lines.append(f"    {t.offload_time_ms:8.4f} ms  {name}  ({t.size} B)")
        return "\n".join(lines)


def _tensor_offload_ms(
    node: torch.fx.Node, bw_h2d: float, bw_d2h: float
) -> tuple[list[TensorObject], list[TensorObject]]:
    """Build the input (reload, H2D) and output (offload, D2H) ``TensorObject``
    lists for ``node``.

    Inputs are the values of this node's predecessors -- if offloaded, they must
    be reloaded (H2D) before this node runs -- tagged with the producer's real
    consumer set (``prod.users``). Outputs are this node's own produced tensors,
    which an offload path would move to host (D2H), tagged with ``node.users``.
    Sizes/ids come from the underlying storage so views/aliases are dedupable
    downstream by ``storage_id``.
    """
    input_tensor_objects: list[TensorObject] = []
    output_tensor_objects: list[TensorObject] = []

    for prod in node.all_input_nodes:
        prod_consumers = [u.name for u in prod.users]
        for t in pytree.tree_leaves(prod.meta.get("val", None)):
            if isinstance(t, torch.Tensor):
                nbytes, sid = _storage_bytes_and_id(t)
                obj = TensorObject(
                    tensor_type=categorize_fwd_bwd(prod),
                    shape=t.shape,
                    dtype=t.dtype,
                    device=t.device,
                    size=nbytes,
                    storage_id=sid,
                    time_ms=_transfer_ms(nbytes, bw_h2d),  # reload: H2D
                )
                obj.consumer_nodes = prod_consumers
                input_tensor_objects.append(obj)

    # Consumers of this node's outputs are node.users. For a multi-output op these
    # are the getitem accessors; mapping each output index to its true downstream
    output_consumers = [u.name for u in node.users]
    for t in pytree.tree_leaves(node.meta.get("val", None)):
        if isinstance(t, torch.Tensor):
            nbytes, sid = _storage_bytes_and_id(t)
            obj = TensorObject(
                tensor_type=categorize_fwd_bwd(node),
                shape=t.shape,
                dtype=t.dtype,
                device=t.device,
                size=nbytes,
                storage_id=sid,
                time_ms=_transfer_ms(nbytes, bw_d2h),  # offload: D2H
            )
            obj.consumer_nodes = output_consumers
            output_tensor_objects.append(obj)

    return input_tensor_objects, output_tensor_objects


def estimate_transfertime(
    gm: torch.fx.GraphModule,
    bw_h2d: float | None = None,
    bw_d2h: float | None = None,
    dev: str = "cuda:0",
) -> TransferEstimatorResult:
    """Per-node D2H/H2D transfer times for the joint fwd+bwd graph.

    ``bw_h2d`` / ``bw_d2h`` (GB/s) default to the cached ``get_transfer_bw(dev)``
    measurement, so repeated calls in one process reuse a single benchmark.
    """
    if bw_h2d is None or bw_d2h is None:
        measured = get_transfer_bw(dev=dev)
        bw_h2d = bw_h2d if bw_h2d is not None else measured["h2d"]
        bw_d2h = bw_d2h if bw_d2h is not None else measured["d2h"]

    input_times: dict = {}
    output_times: dict = {}
    for node in gm.graph.nodes:
        inp, out = _tensor_offload_ms(node, bw_h2d, bw_d2h)
        input_times[node.name] = inp
        output_times[node.name] = out

    result = TransferEstimatorResult(
        input_tensor_transfer_times_ms=input_times,
        output_tensor_transfer_times_ms=output_times,
    )
    return result
