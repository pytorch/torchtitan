# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Async activation offloading to CPU with prefetch for torchtitan.

Activations are copied D2H during forward on a dedicated CUDA stream.  During
backward, H2D transfers are prefetched one layer ahead using
``register_full_backward_pre_hook``, so transfers overlap with backward compute.

Key design decisions:
- GPU-side event ordering (``stream.wait_event``) instead of host-blocking
  ``event.synchronize()`` keeps the CPU thread free to issue work.
- Backward pre-hooks on transformer layers trigger H2D copies one layer
  ahead of where backward currently is, overlapping transfers with compute.
- Pinned CPU buffers are allocated with ``torch.empty(..., pin_memory=True)``
  and rely on PyTorch's ``CachingHostAllocator`` to cache and reuse them
  across steps.
"""

import contextlib
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

    Call :meth:`register_prefetch_hooks` after construction to enable
    layer-level prefetching.  Without it, H2D copies happen reactively in
    ``_unpack_hook`` with no compute overlap.

    Args:
        use_streams: Use separate CUDA streams for D2H/H2D copies.  Disable
            only for debugging on CPU-only machines.
        min_offload_size: Minimum tensor size in bytes to offload.
        max_fwd_stash_size: Maximum number of forward-stash entries to keep
            alive before draining completed D2H copies.  A larger value
            increases overlap at the cost of temporarily holding more GPU
            memory references.
    """

    def __init__(
        self,
        *,
        use_streams: bool = True,
        min_offload_size: int = 1024,
        max_fwd_stash_size: int = 5,
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

        # _fwd_stash[tensor_id] = (cpu_buf, gpu_tensor, d2h_event)
        # Keeps the GPU tensor alive until the D2H copy has finished.
        self._fwd_stash: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}

        # _bwd_stash[tensor_id] = (gpu_buf, h2d_event, cpu_buf)
        # Keeps both the prefetched GPU tensor AND the source CPU buffer alive
        # until the next __enter__.  The cpu_buf ref is critical because
        # cpu_buf.to("cuda", non_blocking=True) submits an async H2D copy, and
        # if cpu_buf is freed by Python GC before the kernel completes, the
        # copy reads garbage data.
        self._bwd_stash: dict[int, tuple[torch.Tensor, Any, torch.Tensor]] = {}

        # Monotonically-increasing ID assigned to each offloaded tensor.
        self._next_id: int = 0

        # --- Prefetch tracking ---------------------------------------------
        # Module ID currently executing in forward (None outside tracked modules).
        self._current_module_id: int | None = None
        # Map from module_id → list of tensor_ids packed during that module's fwd.
        self._module_tensor_ids: dict[int, list[int]] = {}
        # Number of tracked modules (set by register_prefetch_hooks).
        self._num_tracked_modules: int = 0
        # Registered hooks (for cleanup).
        self._hooks: list[Any] = []

        super().__init__(self._pack_hook, self._unpack_hook)

    # ------------------------------------------------------------------
    # Prefetch hooks
    # ------------------------------------------------------------------

    def register_prefetch_hooks(self, modules: list[nn.Module]) -> None:
        """Register forward/backward hooks on *modules* for layer-level prefetching.

        Modules should be provided in forward execution order (e.g., the list
        of transformer blocks).  During backward (reverse order), H2D copies
        for the next layer's activations are started one layer ahead, so
        transfers overlap with backward compute.

        Args:
            modules: Ordered list of modules (typically transformer layers).
        """
        for i, mod in enumerate(modules):
            module_id = i

            def _fwd_pre(mod: nn.Module, args: Any, *, mid: int = module_id) -> None:
                self._current_module_id = mid
                self._module_tensor_ids.setdefault(mid, [])

            def _fwd_post(
                mod: nn.Module, args: Any, output: Any, *, mid: int = module_id
            ) -> None:
                if self._current_module_id == mid:
                    self._current_module_id = None

            def _bwd_pre(
                mod: nn.Module, grad_output: Any, *, mid: int = module_id
            ) -> None:
                # Prefetch this module's tensors (may already be done by the
                # previous module's backward_pre_hook look-ahead).
                for tid in self._module_tensor_ids.get(mid, []):
                    self._prefetch_tensor(tid)
                # Look-ahead: start H2D for the next module in backward order.
                if mid > 0:
                    for tid in self._module_tensor_ids.get(mid - 1, []):
                        self._prefetch_tensor(tid)

            self._hooks.append(mod.register_forward_pre_hook(_fwd_pre))
            self._hooks.append(mod.register_forward_hook(_fwd_post))
            self._hooks.append(mod.register_full_backward_pre_hook(_bwd_pre))

        self._num_tracked_modules = len(modules)

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

        cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)

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

        # Track which module this tensor belongs to for prefetching.
        if self._current_module_id is not None:
            self._module_tensor_ids[self._current_module_id].append(tensor_id)

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

    def _prefetch_tensor(self, tensor_id: int) -> None:
        """Start an async H2D copy for a single tensor on ``_h2d_stream``.

        If the tensor is already in ``_bwd_stash`` (already prefetched) or
        not in ``_fwd_stash`` (already consumed), this is a no-op.
        """
        if tensor_id in self._bwd_stash:
            return
        if tensor_id not in self._fwd_stash:
            return

        cpu_buf, _gpu_ref, d2h_event = self._fwd_stash.pop(tensor_id)

        if self._use_streams:
            assert self._h2d_stream is not None
            # GPU-side ordering: wait for D2H to finish before reading cpu_buf.
            if d2h_event is not None:
                self._h2d_stream.wait_event(d2h_event)
            with torch.cuda.stream(self._h2d_stream):
                gpu_buf = cpu_buf.to("cuda", non_blocking=True)
            h2d_event = torch.cuda.Event()
            h2d_event.record(self._h2d_stream)
        else:
            if d2h_event is not None:
                d2h_event.synchronize()
            gpu_buf = cpu_buf.to("cuda")
            h2d_event = None

        self._bwd_stash[tensor_id] = (gpu_buf, h2d_event, cpu_buf)

    def _unpack_hook(self, tensor_id: int | torch.Tensor) -> torch.Tensor:
        """Called by autograd for every saved tensor consumed during backward.

        If the tensor was prefetched by a backward pre-hook, the H2D copy is
        already in flight (or complete) on ``_h2d_stream``.  Otherwise, a
        synchronous fallback path handles it.
        """
        # If pack_hook returned the tensor directly (not offloaded), just return it.
        if isinstance(tensor_id, torch.Tensor):
            return tensor_id

        # If not yet prefetched, do it now (fallback for untracked tensors).
        if tensor_id not in self._bwd_stash:
            self._prefetch_tensor(tensor_id)

        if tensor_id not in self._bwd_stash:
            raise RuntimeError(
                f"activation_offloading: tensor_id {tensor_id} not found in "
                "fwd_stash or bwd_stash.  This is a bug — please file an issue."
            )

        gpu_buf, h2d_event, _cpu_ref = self._bwd_stash[tensor_id]

        # Make the compute stream wait for the H2D transfer to finish.
        if h2d_event is not None and self._use_streams:
            torch.cuda.current_stream().wait_event(h2d_event)

        return gpu_buf

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(  # pyrefly: ignore[bad-override]
        self,
    ) -> "ActivationOffloadingManager":
        # Ensure all H2D copies from the previous step have completed before
        # we drop CPU buffer references.
        if self._use_streams and self._h2d_stream is not None:
            self._h2d_stream.synchronize()
        self._fwd_stash.clear()
        self._bwd_stash.clear()
        self._module_tensor_ids.clear()
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

    Registers forward/backward hooks on the model's transformer layers (if
    found under ``model.layers``) for layer-level H2D prefetching.  Also
    registers hooks on the ``output`` sub-module to exclude it from
    offloading.

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

    # Register prefetch hooks on transformer layers for H2D overlap.
    layers = None
    try:
        layers_mod = model.get_submodule("layers")
        if isinstance(layers_mod, nn.ModuleList):
            layers = list(layers_mod)
    except AttributeError:
        pass

    if layers:
        manager.register_prefetch_hooks(layers)

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
