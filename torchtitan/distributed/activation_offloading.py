# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Async activation offloading to CPU with prefetch for torchtitan.

Activations are copied D2H during forward on a dedicated CUDA stream, then
prefetched H2D during backward on a second stream, so transfers overlap with
compute at near-zero overhead.

Key design decisions vs. torchtune's OffloadActivations:
- Pre-allocated pinned CPU memory pool avoids per-tensor
  ``torch.empty(..., pin_memory=True)`` calls, which trigger implicit CUDA
  synchronisation and negate the overlap benefit.
- Explicit event-based synchronisation keeps the compute stream from consuming
  data before H2D transfers finish.
"""

import contextlib
import threading
from typing import Any

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


class ActivationOffloadingManager(torch.autograd.graph.saved_tensors_hooks):
    """Context manager that offloads activation tensors to CPU during forward
    and prefetches them back before backward.

    Uses ``saved_tensors_hooks`` so it is transparent to the rest of the
    training loop.  Only tensors that live on CUDA, are not ``nn.Parameter``
    instances, and exceed *min_offload_size* bytes are moved.

    Args:
        use_streams: Use separate CUDA streams for D2H/H2D copies.  Disable
            only for debugging on CPU-only machines.
        min_offload_size: Minimum tensor size in bytes to offload.
        max_fwd_stash_size: Maximum number of forward-stash entries to keep
            alive before draining completed D2H copies.  A larger value
            increases overlap at the cost of temporarily holding more GPU
            memory references.
        cpu_pool_size: Number of bytes to pre-allocate in the pinned CPU pool.
            When the pool is exhausted a fallback allocation is made with a
            warning.
    """

    def __init__(
        self,
        *,
        use_streams: bool = True,
        min_offload_size: int = 1024,
        max_fwd_stash_size: int = 5,
        cpu_pool_size: int = 2 * 1024**3,  # 2 GiB default
    ) -> None:
        self._use_streams = use_streams and torch.cuda.is_available()
        self._min_offload_size = min_offload_size
        self._max_fwd_stash_size = max_fwd_stash_size

        # CUDA streams for overlapping transfers with computation.
        if self._use_streams:
            self._d2h_stream: torch.cuda.Stream | None = torch.cuda.Stream()
            self._h2d_stream: torch.cuda.Stream | None = torch.cuda.Stream()
        else:
            self._d2h_stream = None
            self._h2d_stream = None

        # --- Pinned CPU memory pool (slab allocator) -----------------------
        # We keep a flat pinned tensor and carve slices from it.  This avoids
        # the per-tensor cudaMallocHost calls that torch.empty(pin_memory=True)
        # issues, each of which flushes the CUDA work queue.
        self._pool_lock = threading.Lock()
        if cpu_pool_size > 0 and torch.cuda.is_available():
            try:
                self._cpu_pool: torch.Tensor | None = torch.empty(
                    cpu_pool_size, dtype=torch.uint8, pin_memory=True
                )
            except Exception:
                logger.warning(
                    "activation_offloading: failed to allocate %d-byte pinned "
                    "pool; falling back to per-tensor allocations.",
                    cpu_pool_size,
                )
                self._cpu_pool = None
        else:
            self._cpu_pool = None

        # Byte offset of the next free region inside _cpu_pool.
        self._pool_offset: int = 0

        # _fwd_stash[tensor_id] = (cpu_buf, gpu_tensor, d2h_event)
        # Keeps the GPU tensor alive until the D2H copy has finished.
        self._fwd_stash: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}

        # _bwd_stash[tensor_id] = (gpu_buf, h2d_event, cpu_buf)
        # Keeps both the prefetched GPU tensor AND the source CPU buffer alive
        # until the next __enter__.  The cpu_buf ref is critical for the fallback
        # (non-pool) path: cpu_buf.to("cuda", non_blocking=True) submits an async
        # H2D copy, and if cpu_buf is freed by Python GC before the kernel
        # completes, the copy reads garbage data.  Pool slices are safe because
        # the pool tensor (self._cpu_pool) is always alive, but fresh pin_memory
        # allocations must be kept alive explicitly.
        self._bwd_stash: dict[int, tuple[torch.Tensor, Any, torch.Tensor]] = {}

        # Monotonically-increasing ID assigned to each offloaded tensor.
        self._next_id: int = 0

        super().__init__(self._pack_hook, self._unpack_hook)

    # ------------------------------------------------------------------
    # Pool allocation helpers
    # ------------------------------------------------------------------

    def _alloc_cpu_buf(self, tensor: torch.Tensor) -> torch.Tensor:
        """Allocate a pinned CPU buffer with the same shape/dtype as *tensor*.

        Tries the pre-allocated pool first; falls back to a fresh allocation.
        """
        nbytes = tensor.nbytes
        elem_size = tensor.element_size()
        with self._pool_lock:
            if self._cpu_pool is not None:
                # Align pool offset to the target dtype's element size so that
                # raw.view(tensor.dtype) never hits a misaligned-storage error.
                align = elem_size
                aligned_offset = (self._pool_offset + align - 1) & ~(align - 1)
                if aligned_offset + nbytes <= self._cpu_pool.numel():
                    raw = self._cpu_pool[aligned_offset : aligned_offset + nbytes]
                    self._pool_offset = aligned_offset + nbytes
                    buf = raw.view(tensor.dtype).reshape(tensor.shape)
                    return buf

        # Pool exhausted – fall back to a fresh pinned CPU allocation.
        logger.warning(
            "activation_offloading: pinned pool exhausted; allocating %d bytes "
            "with pin_memory=True (may cause CUDA sync).",
            nbytes,
        )
        return torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)

    def _reset_pool(self) -> None:
        """Reset the pool offset so all slices can be reused next forward."""
        with self._pool_lock:
            self._pool_offset = 0

    # ------------------------------------------------------------------
    # saved_tensors_hooks callbacks
    # ------------------------------------------------------------------

    def _pack_hook(self, tensor: torch.Tensor) -> int | torch.Tensor:
        """Called by autograd for every tensor saved during forward.

        Returns a unique integer ID that autograd stores; the original tensor
        is kept alive in ``_fwd_stash`` only until the D2H copy completes.
        """
        # Only offload eligible CUDA tensors.
        if (
            tensor.device.type != "cuda"
            or tensor.nbytes < self._min_offload_size
            or isinstance(tensor, nn.Parameter)
        ):
            return tensor  # type: ignore[return-value]

        tensor_id = self._next_id
        self._next_id += 1

        cpu_buf = self._alloc_cpu_buf(tensor)

        if self._use_streams:
            assert self._d2h_stream is not None  # guaranteed when _use_streams=True
            # Capture the compute stream *before* switching context — inside
            # ``torch.cuda.stream(s)``, ``current_stream()`` returns ``s``,
            # so the wait_stream call must use the captured reference.
            compute_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._d2h_stream):
                # Make sure the compute stream has finished writing the tensor
                # before we start copying it.
                self._d2h_stream.wait_stream(compute_stream)
                cpu_buf.copy_(tensor, non_blocking=True)
            d2h_event = torch.cuda.Event()
            d2h_event.record(self._d2h_stream)
        else:
            cpu_buf.copy_(tensor)
            d2h_event = None

        self._fwd_stash[tensor_id] = (cpu_buf, tensor, d2h_event)

        # Drain stash entries whose D2H copy has finished to release GPU refs.
        if len(self._fwd_stash) > self._max_fwd_stash_size:
            self._drain_fwd_stash(blocking=False)

        return tensor_id

    def _drain_fwd_stash(self, *, blocking: bool) -> None:
        """Release GPU tensor references for completed D2H copies."""
        done = []
        for tid, (cpu_buf, _gpu_tensor, event) in self._fwd_stash.items():
            if event is None or blocking or event.query():
                done.append((tid, cpu_buf))
        for tid, cpu_buf in done:
            self._fwd_stash[tid] = (cpu_buf, None, None)  # type: ignore[assignment]

    def _unpack_hook(self, tensor_id: int | torch.Tensor) -> torch.Tensor:
        """Called by autograd for every saved tensor consumed during backward.

        Prefetches the CPU buffer back to GPU on *h2d_stream*, then lets the
        compute stream wait for it.
        """
        # If pack_hook returned the tensor directly (not offloaded), just return it.
        if isinstance(tensor_id, torch.Tensor):
            return tensor_id

        # Already prefetched during a previous unpack (shouldn't happen, but be safe).
        if tensor_id in self._bwd_stash:
            gpu_buf, _, _cpu_ref = self._bwd_stash[tensor_id]
            return gpu_buf

        # Retrieve the CPU buffer from whichever stash still has it.
        if tensor_id in self._fwd_stash:
            cpu_buf, _gpu_ref, event = self._fwd_stash.pop(tensor_id)
            if event is not None:
                # Block until the D2H copy is fully visible on the CPU side.
                event.synchronize()
        else:
            raise RuntimeError(
                f"activation_offloading: tensor_id {tensor_id} not found in "
                "fwd_stash.  This is a bug — please file an issue."
            )

        # Prefetch back to GPU.
        if self._use_streams:
            assert self._h2d_stream is not None  # guaranteed when _use_streams=True
            with torch.cuda.stream(self._h2d_stream):
                gpu_buf = cpu_buf.to("cuda", non_blocking=True)
            h2d_event = torch.cuda.Event()
            h2d_event.record(self._h2d_stream)
            # Make the compute stream wait for the H2D transfer to finish.
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
        else:
            gpu_buf = cpu_buf.to("cuda")
            h2d_event = None

        # Keep cpu_buf alive until __enter__ so the async H2D copy (on h2d_stream)
        # can finish reading it even after this function returns.
        self._bwd_stash[tensor_id] = (gpu_buf, h2d_event, cpu_buf)
        return gpu_buf

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(  # pyrefly: ignore[bad-override]
        self,
    ) -> "ActivationOffloadingManager":
        self._reset_pool()
        self._fwd_stash.clear()
        self._bwd_stash.clear()
        self._next_id = 0
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        super().__exit__(*args)
        # In debug builds, warn if stashes are not empty.
        if __debug__:
            if self._fwd_stash:
                logger.debug(
                    "activation_offloading: %d fwd_stash entries remain after "
                    "context exit — GPU tensor refs may linger.",
                    len(self._fwd_stash),
                )
            if self._bwd_stash:
                logger.debug(
                    "activation_offloading: %d bwd_stash entries remain after "
                    "context exit — prefetched GPU tensors may linger.",
                    len(self._bwd_stash),
                )


def get_activation_offloading_ctx(
    model: nn.Module,
    enable: bool,
    *,
    use_streams: bool = True,
    min_offload_size: int = 1024,
    max_fwd_stash_size: int = 5,
) -> contextlib.AbstractContextManager:
    """Return an activation-offloading context manager, or a no-op if disabled.

    Also registers forward hooks on the model's ``output`` sub-module (the final
    LM-head linear) so that its inputs/outputs are wrapped with a no-op
    ``saved_tensors_hooks`` — identical to the torchtune pattern.  The output
    projection is large and consumed immediately after forward, so offloading it
    wastes bandwidth.

    Args:
        model: The model whose forward pass will be wrapped.
        enable: If ``False``, returns ``contextlib.nullcontext()``.
        use_streams: Enable separate CUDA streams for D2H/H2D copies.
        min_offload_size: Minimum tensor size (bytes) to offload.
        max_fwd_stash_size: Window of forward-stash entries kept alive for overlap.

    Returns:
        An ``ActivationOffloadingManager`` or ``contextlib.nullcontext()``.
    """
    if not enable:
        return contextlib.nullcontext()

    manager = ActivationOffloadingManager(
        use_streams=use_streams,
        min_offload_size=min_offload_size,
        max_fwd_stash_size=max_fwd_stash_size,
    )

    # Wrap the output projection with a passthrough hook so its activations
    # are never offloaded (they are immediately used by the loss and would
    # require an extra round-trip for no memory benefit).
    output_mod = None
    try:
        output_mod = model.get_submodule("output")
    except AttributeError:
        pass

    if output_mod is not None:
        _noop_ctx = torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)

        def _pre_hook(mod: nn.Module, args: Any) -> None:
            _noop_ctx.__enter__()

        def _post_hook(mod: nn.Module, args: Any, output: Any) -> None:
            _noop_ctx.__exit__(None, None, None)

        output_mod.register_forward_pre_hook(_pre_hook)
        output_mod.register_forward_hook(_post_hook)

    return manager
