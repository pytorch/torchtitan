# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static D2H/H2D transfer-time estimator for a joint fwd+loss+bwd FX graph.

The transfer-time analogue of ``memory_estimator.py`` / ``runtime_estimator.py``.
Where those track storage liveness and per-op compute time, this one walks the
joint graph and, for every node, estimates the time to move each input tensor
(D2H) and each output tensor (H2D) across the host<->device link. It is the cost
model an activation-offload solver uses to decide what fits in the offload
bandwidth budget.

Bandwidth: ``estimate_transfertime`` accepts explicit ``bw_h2d`` / ``bw_d2h``
(GB/s); when omitted it calls ``measure_transfer_bw`` once to measure them on the
current device. Pass them in (e.g. in tests) to avoid the live benchmark.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.utils._pytree as pytree

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node

# Default placeholder D2H/H2D bandwidth (GB/s) when no measurement is supplied.
# GB200 NVLink-C2C NUMA-local copies run ~350 GB/s; PCIe5 x16 is ~50 GB/s.
_OFFLOAD_BANDWIDTH_GBPS = 50.0


def measure_transfer_bw(
    nbytes: int = 512 * 1024 * 1024,
    iters: int = 20,
    pinned: bool = True,
    dev: str = "cuda:0",
) -> dict:
    """Measure H2D and D2H bandwidth (GB/s) with pinned host memory.

    Pageable host memory is much slower, so pinned is the default and the regime
    an offload path would actually use.
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


def _transfer_ms(nbytes: int, bw_gbps: float) -> float:
    """Transfer time in ms for ``nbytes`` at ``bw_gbps`` GB/s.
    GB/s -> bytes/ms is a factor of 1e6 (1e9 bytes/s / 1e3 ms/s)."""
    assert nbytes >= 0, f"negative bytes: {nbytes}"
    return nbytes / (bw_gbps * 1e6)


class TensorObject:
    """One tensor moved across the host<->device link, with its transfer time."""

    def __init__(self, tensor_type, shape, dtype, device, offload_time_ms):
        self.tensor_type = tensor_type  # "forward" | "backward" (producer region)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.size = math.prod(shape) * dtype.itemsize  # shape is torch.Size
        self.offload_time_ms = offload_time_ms
        self.consumer_nodes = []

    def __repr__(self):
        return (
            f"TensorObject(type={self.tensor_type}, shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device}, size={self.size})"
        )


def categorize_fwd_bwd(node: torch.fx.Node) -> str:
    return "backward" if _is_backward_node(node) else "forward"


def _max_offload(d: dict) -> float:
    return max((t.offload_time_ms for lst in d.values() for t in lst), default=0.0)


@dataclass
class TransferEstimatorResult:
    # node name -> list[TensorObject] consumed (inputs) / produced (outputs)
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
            f"max input  transfer time: {self.max_input_offload_time():.3f} ms",
            f"max output transfer time: {self.max_output_offload_time():.3f} ms",
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
    """Build the input (D2H) and output (H2D) ``TensorObject`` lists for ``node``.

    Inputs are the values of this node's predecessors (the tensors that would be
    offloaded), tagged with their real consumer set (``prod.users``). Outputs are
    this node's own produced tensors, tagged with ``node.users``.
    """
    input_tensor_objects: list[TensorObject] = []
    output_tensor_objects: list[TensorObject] = []

    for prod in node.all_input_nodes:
        prod_consumers = [u.name for u in prod.users]
        for t in pytree.tree_leaves(prod.meta.get("val", None)):
            if isinstance(t, torch.Tensor):
                obj = TensorObject(
                    tensor_type=categorize_fwd_bwd(prod),
                    shape=t.shape,
                    dtype=t.dtype,
                    device=t.device,
                    offload_time_ms=_transfer_ms(t.numel() * t.element_size(), bw_d2h),
                )
                obj.consumer_nodes = prod_consumers
                input_tensor_objects.append(obj)

    # Consumers of this node's outputs are node.users. For a multi-output op these
    # are the getitem accessors; mapping each output index to its true downstream
    # consumers is a TODO -- for single-output nodes node.users is already exact.
    output_consumers = [u.name for u in node.users]
    for t in pytree.tree_leaves(node.meta.get("val", None)):
        if isinstance(t, torch.Tensor):
            obj = TensorObject(
                tensor_type=categorize_fwd_bwd(node),
                shape=t.shape,
                dtype=t.dtype,
                device=t.device,
                offload_time_ms=_transfer_ms(t.numel() * t.element_size(), bw_h2d),
            )
            obj.consumer_nodes = output_consumers
            output_tensor_objects.append(obj)

    return input_tensor_objects, output_tensor_objects


def estimate_transfertime(
    gm: torch.fx.GraphModule,
    *,
    bw_h2d: Optional[float] = None,
    bw_d2h: Optional[float] = None,
    verbose: bool = False,
) -> TransferEstimatorResult:
    """Per-node D2H/H2D transfer times for the joint fwd+bwd graph.

    ``bw_h2d`` / ``bw_d2h`` (GB/s) are measured once on the device if omitted.
    """
    if bw_h2d is None or bw_d2h is None:
        measured = measure_transfer_bw()
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
    if verbose:
        print(result.summary())
    return result
