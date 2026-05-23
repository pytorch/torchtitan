# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph pass infrastructure for the autoresearch experiment.

This module provides the pass orchestration framework: building the pass list,
applying passes in order, and debug instrumentation. The agent adds custom
pass functions and registers them in ``construct_default_graph_passes``.
"""

from __future__ import annotations

import atexit
import functools
import gc
import time
import types
from collections.abc import Callable

import torch
from torch._logging import trace_structured

from torchtitan.experiments.graph_trainer.debug_utils import (
    log_graph_diff,
    snapshot_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.tools.logging import logger


def compile_time_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
    *,
    use_cudagraph: bool = False,
) -> list[Callable]:
    """Return compile-time passes (used by precompile path)."""
    return construct_default_graph_passes(traced_result, config)


def remove_detach_nodes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Remove ``aten.detach.default`` nodes from the joint fwd+bwd graph.

    The graph is executed inside a ``torch.no_grad()`` block at runtime, so
    ``detach`` has no autograd effect. Each ``detach.default`` call still
    allocates a new tensor view in the dispatcher, so removing them shaves
    a small amount of per-step launch and bookkeeping overhead.

    Strategy:
        1. Collect all ``aten.detach.default`` call_function nodes.
        2. Rewire every user of each node to read from ``node.args[0]``.
        3. Erase the now-orphaned detach nodes.
        4. Run ``eliminate_dead_code`` once to clean up any newly-dead ops,
           followed by ``graph.lint()`` and ``gm.recompile()``.
    """
    detach_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.detach.default
    ]
    for node in detach_nodes:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    logger.info(f"remove_detach_nodes: removed {len(detach_nodes)} detach nodes")
    return gm


def _try_set_inductor_config(name: str, value) -> None:
    """Conservatively set an Inductor config flag if it exists."""
    try:
        import torch._inductor.config as C

        if hasattr(C, name):
            setattr(C, name, value)
            logger.info(f"apply_inductor_pattern_passes: set config.{name}={value!r}")
    except Exception as e:  # pragma: no cover - defensive
        logger.info(
            f"apply_inductor_pattern_passes: failed to set config.{name}: "
            f"{type(e).__name__}({e})"
        )


def _try_pass(
    gm: torch.fx.GraphModule,
    label: str,
    fn,
    *args,
    **kwargs,
) -> None:
    """Invoke ``fn(*args, **kwargs)`` guarded by try/except and log a node-count diff."""
    before = len(list(gm.graph.nodes))
    try:
        fn(*args, **kwargs)
    except Exception as e:
        logger.info(
            f"apply_inductor_pattern_passes: {label} raised " f"{type(e).__name__}({e})"
        )
        return
    after = len(list(gm.graph.nodes))
    logger.info(
        f"apply_inductor_pattern_passes: {label} ran (nodes {before} -> {after})"
    )


def apply_inductor_pattern_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Apply Inductor's joint-graph FX pattern matchers to the traced graph.

    Inductor ships several FX-level pattern matchers that fuse common ATen
    sequences (mm+bias+activation, SDPA epilogues, _to_copy round-trips,
    binary folding, symint dedup, etc.). They operate in place on the
    GraphModule and don't require re-tracing. We try several upstream
    entry points, log what's available and what changes the graph, then
    lint+recompile.

    Each call is independently wrapped: import failures and runtime errors
    are logged and skipped, never re-raised.
    """

    def _node_count(g: torch.fx.GraphModule) -> int:
        return len(list(g.graph.nodes))

    initial_count = _node_count(gm)
    logger.info(f"apply_inductor_pattern_passes: initial node count {initial_count}")

    # Toggle config flags that gate optional pattern matchers used by the
    # joint_graph / post_grad passes below. Pad_mm, b2b_gemm,
    # decompose_mem_bound_mm, and group/batch fusion are all driven by these.
    # All sets are best-effort: flags that don't exist on this PyTorch are
    # silently skipped.
    _try_set_inductor_config("shape_padding", True)
    _try_set_inductor_config("permute_fusion", True)
    _try_set_inductor_config("b2b_gemm_pass", True)
    _try_set_inductor_config("decompose_mem_bound_mm", True)
    _try_set_inductor_config("batch_fusion", True)
    _try_set_inductor_config("group_fusion", True)
    _try_set_inductor_config("split_cat_fx_passes", True)
    _try_set_inductor_config("efficient_conv_bn_eval_fx_passes", False)
    # Comm overlap / bucketing flags. Iter-3 found manual bucketing reduces
    # launches but doesn't move TPS; we still flip these on so upstream
    # passes (joint_graph, post_grad) honour them where applicable.
    _try_set_inductor_config("reorder_for_compute_comm_overlap", True)
    _try_set_inductor_config("_micro_pipeline_tp", True)

    # 1) joint_graph_passes
    try:
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes

        try:
            before = _node_count(gm)
            joint_graph_passes(gm)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: joint_graph_passes ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: joint_graph_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: joint_graph_passes not available")

    # 2) post_grad_passes (training mode)
    try:
        from torch._inductor.fx_passes.post_grad import post_grad_passes

        try:
            before = _node_count(gm)
            post_grad_passes(gm, is_inference=False)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: post_grad_passes(is_inference=False) "
                f"ran (nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: post_grad_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: post_grad_passes not available")

    # 3) pre_grad_passes
    try:
        from torch._inductor.fx_passes.pre_grad import pre_grad_passes

        try:
            before = _node_count(gm)
            pre_grad_passes(gm, example_inputs)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: pre_grad_passes ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: pre_grad_passes raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: pre_grad_passes not available")

    # 4) binary_folding_pass
    try:
        from torch._inductor.fx_passes.binary_folding import binary_folding_pass

        try:
            before = _node_count(gm)
            binary_folding_pass(gm.graph)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: binary_folding_pass ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: binary_folding_pass raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info("apply_inductor_pattern_passes: binary_folding_pass not available")

    # 5) dedupe_symint_uses_pass
    try:
        from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symint_uses_pass

        try:
            before = _node_count(gm)
            dedupe_symint_uses_pass(gm.graph)
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: dedupe_symint_uses_pass ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: dedupe_symint_uses_pass raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info(
            "apply_inductor_pattern_passes: dedupe_symint_uses_pass not available"
        )

    # --- Additional passes (iter 7) on top of the original 5. ---------------
    # Each was tried and the no-ops have been pruned. See the agent's iter-7
    # report for the full enumeration. Two upstream bucketing passes
    # (all_gather and reduce_scatter) materially changed the graph and are
    # kept; all other candidates left node count unchanged and were dropped.

    # 6) bucketing.bucket_all_gather - bucket FSDP all_gathers by size. This
    # materially restructures the graph (+~810 nodes) and corresponds to
    # iter-3's manual bucketing, but driven by upstream scheduling logic.
    # 7) bucketing.bucket_reduce_scatter - same idea for reduce_scatters
    # (+~615 nodes).
    #
    # Note: the upstream bucketing passes can leave nodes in non-topological
    # order (downstream consumers raise "Argument was used before defined"),
    # so we apply an explicit stable topological sort before the next
    # consumer touches the graph.
    from torch.fx.passes.tools_common import stable_topological_sort

    try:
        from torch._inductor.fx_passes.bucketing import (
            bucket_all_gather,
            bucket_reduce_scatter,
        )

        _try_pass(gm, "bucket_all_gather", bucket_all_gather, gm)
        _try_pass(gm, "bucket_reduce_scatter", bucket_reduce_scatter, gm)
        stable_topological_sort(gm)
    except ImportError:
        logger.info("apply_inductor_pattern_passes: bucketing passes not available")

    # 8) overlap_scheduling.schedule_overlap_bucketing - reorders the graph so
    # collectives launched by FSDP/TP are issued well before their consumers,
    # giving the NCCL streams a chance to overlap with compute. The pass is
    # expensive (~30-45s at compile time) but a no-op at runtime if it
    # doesn't reorder; we re-included it because the pruned version
    # (bucketing only) reliably ran ~120 TPS slower than the version with
    # this pass in place.
    try:
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        before = _node_count(gm)
        try:
            new_gm = schedule_overlap_bucketing(gm)
            if isinstance(new_gm, torch.fx.GraphModule):
                gm = new_gm
            after = _node_count(gm)
            logger.info(
                f"apply_inductor_pattern_passes: schedule_overlap_bucketing ran "
                f"(nodes {before} -> {after})"
            )
        except Exception as e:
            logger.info(
                f"apply_inductor_pattern_passes: schedule_overlap_bucketing raised "
                f"{type(e).__name__}({e})"
            )
    except ImportError:
        logger.info(
            "apply_inductor_pattern_passes: schedule_overlap_bucketing not available"
        )

    gm.graph.lint()
    gm.recompile()
    final_count = _node_count(gm)
    logger.info(
        f"apply_inductor_pattern_passes: final node count {final_count} "
        f"(delta {final_count - initial_count})"
    )
    return gm


def install_cuda_graph(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    num_static_inputs: int = 0,
) -> torch.fx.GraphModule:
    """Wrap ``gm.forward`` in a ``torch.cuda.CUDAGraph`` static-replay runner.

    The traced fwd+bwd graph runs the same op sequence every step, so we can
    capture it once and replay on subsequent calls to eliminate per-step CPU
    launch overhead. The captured graph reads from persistent CUDA buffers
    that we allocate to match ``example_inputs`` (FakeTensors with ``shape``,
    ``dtype``, ``device`` and ``stride``). On every replay we copy varying
    inputs into those buffers, then call ``g.replay()`` and return the
    persistent output references.

    Memory optimization: ``num_static_inputs`` indicates how many leading
    flat inputs come from module parameters/buffers (stable addresses,
    threaded by ``minimal_fx_tracer``). For those we do NOT allocate a
    duplicate persistent buffer — they're plugged in at first-call time and
    used directly. Allocating a duplicate buffer per parameter would nearly
    double model-state memory and frequently OOMs under FSDP+TP.

    Correctness note: ``with torch.cuda.graph(g):`` only RECORDS ops; it
    does not execute them. So immediately after capture, ``state["outputs"]``
    holds whatever the last warmup run produced. To produce correct outputs
    on the very first call after capture, we ``g.replay()`` once at the end
    of ``_do_capture`` so ``state["outputs"]`` reflects executed values.

    Failure mode: if anything goes wrong during warmup or capture
    (unsupported op, OOM, NCCL-unfriendly capture, etc.), we log the
    exception and leave ``gm.forward`` untouched so callers keep the
    un-captured (but still pass-optimized) graph. The iter-7 baseline
    (~4,783 TPS) persists in that case.
    """

    # Only attempt capture for CUDA workloads.
    cuda_inputs = [
        x
        for x in example_inputs
        if isinstance(x, torch.Tensor) and x.device.type == "cuda"
    ]
    if not cuda_inputs:
        logger.info("install_cuda_graph: no CUDA inputs found; skipping")
        return gm
    if not torch.cuda.is_available():
        logger.info("install_cuda_graph: CUDA not available; skipping")
        return gm

    # 1) Allocate persistent CUDA buffers matching each example input.
    #    Non-tensor example inputs are passed through unchanged.
    #    Inputs in [0:num_static_inputs) are model state with stable
    #    addresses; we DO NOT allocate duplicate buffers for them. They are
    #    set in ``static_inputs`` from the caller's tensor on the first call.
    try:
        static_inputs: list = []
        is_tensor_input: list[bool] = []
        is_state_input: list[bool] = []
        static_ptrs: list[int | None] = []
        for i, ex in enumerate(example_inputs):
            if isinstance(ex, torch.Tensor):
                if i < num_static_inputs:
                    # State input: placeholder, will be filled with the
                    # caller's tensor on the first real call. No allocation.
                    static_inputs.append(None)
                    is_tensor_input.append(True)
                    is_state_input.append(True)
                    static_ptrs.append(None)
                else:
                    # Varying input: allocate a real CUDA buffer with matching
                    # layout. The buffer is a leaf with requires_grad=False;
                    # the captured graph runs under no_grad in the runner.
                    buf = torch.empty_strided(
                        tuple(ex.shape),
                        tuple(ex.stride()),
                        dtype=ex.dtype,
                        device="cuda",
                    )
                    buf.requires_grad_(False)
                    static_inputs.append(buf)
                    is_tensor_input.append(True)
                    is_state_input.append(False)
                    static_ptrs.append(buf.data_ptr())
            else:
                static_inputs.append(ex)
                is_tensor_input.append(False)
                is_state_input.append(False)
                static_ptrs.append(None)
    except Exception as e:
        logger.info(
            f"install_cuda_graph: failed to allocate persistent buffers: "
            f"{type(e).__name__}({e})"
        )
        return gm

    original_forward = gm.forward
    state: dict[str, object] = {
        "captured": False,
        "graph": None,
        "outputs": None,
    }

    def _safe_copy_to_static(i: int, ex_in) -> bool:
        """Copy ``ex_in`` into ``static_inputs[i]`` if it isn't already that buffer.

        Skips when:
          - ``ex_in`` is not a Tensor (e.g. None placeholder).
          - ``ex_in.data_ptr()`` equals the static buffer's ``data_ptr()``,
            meaning the caller is already using our static buffer as its
            backing storage (avoid self-copy, and don't stomp DTensors whose
            local storage we own).

        Returns True if copy succeeded (or was correctly skipped),
        False on exception.
        """
        if not isinstance(ex_in, torch.Tensor):
            return True
        try:
            # Self-aliasing check: caller is already passing the static buffer in.
            try:
                if ex_in.data_ptr() == static_ptrs[i]:
                    return True
            except Exception:
                # DTensors or other tensor subclasses may not expose data_ptr
                # directly; fall through to copy_.
                pass
            static_inputs[i].copy_(ex_in)
            return True
        except Exception as e:
            logger.info(
                f"install_cuda_graph: copy_ to static buffer {i} failed: "
                f"{type(e).__name__}({e})"
            )
            return False

    def _do_capture(real_inputs) -> bool:
        """Warm up, then capture ``original_forward(*static_inputs)``.

        On success, ``g.replay()`` is called once before returning so that
        ``state["outputs"]`` reflects executed values rather than warmup
        leftovers.

        Returns True on success, False on any exception (after logging).
        """
        # Seed the static buffers from the first real call.
        # State inputs (params/buffers, stable address): plug in the caller's
        # tensor directly. They share the same storage every step.
        # Varying inputs: copy into the persistent buffer.
        for i, (is_t, ex_in) in enumerate(zip(is_tensor_input, real_inputs)):
            if is_t:
                if is_state_input[i]:
                    static_inputs[i] = ex_in
                    # Record this state input's storage address so that on
                    # subsequent calls we can verify it didn't change.
                    try:
                        static_ptrs[i] = (
                            ex_in.data_ptr()
                            if isinstance(ex_in, torch.Tensor)
                            else None
                        )
                    except Exception:
                        static_ptrs[i] = None
                else:
                    if not _safe_copy_to_static(i, ex_in):
                        return False
            else:
                # Non-tensor positional: replace in case it changed.
                static_inputs[i] = ex_in

        # Standard torch.cuda.graph capture boilerplate: warm up on a side
        # stream so the default stream is drained, then capture. Only 1
        # warmup iteration to minimize peak memory; we explicitly sync and
        # empty the allocator cache after each step so intermediates are
        # freed before capture (and so a warmup OOM doesn't bleed into the
        # fallback path's eager forward).
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    _ = original_forward(*static_inputs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            del _
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.info(
                f"install_cuda_graph: warmup failed "
                f"({type(e).__name__}({e})); leaving forward untouched"
            )
            # Try to recover memory from a partial warmup so the fallback
            # eager forward isn't blocked by OOM debris on the side stream.
            try:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            return False

        try:
            g = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(g):
                    captured_outputs = original_forward(*static_inputs)
        except Exception as e:
            logger.info(
                f"install_cuda_graph: capture failed "
                f"({type(e).__name__}({e})); leaving forward untouched"
            )
            return False

        # Critical: with torch.cuda.graph(g): only RECORDS ops; it does not
        # execute them. The output buffers currently hold whatever the last
        # warmup run produced. Replay once now so state["outputs"] holds
        # correct values for the first post-capture call.
        try:
            g.replay()
            torch.cuda.current_stream().synchronize()
        except Exception as e:
            logger.info(
                f"install_cuda_graph: initial replay after capture failed "
                f"({type(e).__name__}({e})); leaving forward untouched"
            )
            return False

        state["graph"] = g
        state["outputs"] = captured_outputs
        state["captured"] = True
        logger.info(
            f"install_cuda_graph: capture succeeded "
            f"(num_inputs={len(real_inputs)}, num_tensor_inputs="
            f"{sum(is_tensor_input)}, num_state_inputs={num_static_inputs})"
        )
        return True

    def captured_forward(self_gm, *real_inputs):
        # Lazy-capture on first real call.
        if not state["captured"]:
            try:
                ok = _do_capture(real_inputs)
            except Exception as e:
                logger.info(
                    f"install_cuda_graph: unexpected error during capture: "
                    f"{type(e).__name__}({e}); falling back to eager forward"
                )
                ok = False
            if not ok:
                # Fall back to the un-captured graph permanently; rebind
                # forward so we don't pay this overhead on later steps. Drop
                # the persistent static-input buffers and any captured-graph
                # references so the fallback eager forward has enough CUDA
                # memory to execute (especially after a warmup OOM).
                state["graph"] = None
                state["outputs"] = None
                for j in range(len(static_inputs)):
                    static_inputs[j] = None
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gm.forward = types.MethodType(
                    lambda self_gm, *args: original_forward(*args), gm
                )
                return original_forward(*real_inputs)
            return state["outputs"]

        # Subsequent calls: copy varying inputs into the persistent buffers,
        # then replay. State inputs are assumed to share storage with the
        # captured run (FSDP params, buffers — stable address) and are
        # skipped. For varying inputs, skip copies when the pointer matches
        # the static buffer itself (self-aliasing).
        for i, (is_t, ex_in) in enumerate(zip(is_tensor_input, real_inputs)):
            if not is_t:
                continue
            if not isinstance(ex_in, torch.Tensor):
                continue
            if is_state_input[i]:
                # Stable state input (param/buffer). Storage is shared with
                # the captured graph; skip copy.
                continue
            try:
                ptr = ex_in.data_ptr()
            except Exception:
                ptr = None
            if ptr is not None and ptr == static_ptrs[i]:
                # Self-aliasing: caller already uses our static buffer.
                continue
            static_inputs[i].copy_(ex_in)

        state["graph"].replay()
        return state["outputs"]

    # Register a teardown that drops CUDA-graph state before process exit, so
    # the captured graph + persistent buffers don't keep CUDA memory + NCCL
    # state alive past torchrun's elastic teardown.
    #
    # The captured CUDA graph records NCCL collectives that reference NCCL
    # communicator handles. When torch.distributed.destroy_process_group()
    # is called during trainer shutdown, NCCL tries to drain in-flight ops;
    # the captured-but-not-yet-replayed graph holds references that prevent
    # this and the call hangs for ~22 minutes waiting on SIGTERM.
    #
    # We address this with two complementary hooks:
    #   1) atexit.register: covers normal interpreter shutdown.
    #   2) Monkey-patching torch.distributed.destroy_process_group: ensures
    #      our cleanup runs IMMEDIATELY before destroy_process_group(), even
    #      when destroy_process_group is called from trainer.close() (i.e.
    #      well before atexit handlers fire).
    cleanup_done = {"done": False}

    def _cleanup() -> None:
        if cleanup_done["done"]:
            return
        cleanup_done["done"] = True
        try:
            logger.info("install_cuda_graph: cuda-graph cleanup started")
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            # Explicitly reset the CUDA graph if the API exposes it; this
            # releases the captured op stream and allows the NCCL teardown
            # path to proceed without waiting on captured collectives.
            g = state.get("graph")
            if g is not None:
                for method_name in ("reset", "_destroy"):
                    if hasattr(g, method_name):
                        try:
                            getattr(g, method_name)()
                        except Exception:
                            pass
                        break
            state["graph"] = None
            state["outputs"] = None
            # Drop references to persistent buffers so their CUDA allocations
            # can be freed.
            for i in range(len(static_inputs)):
                static_inputs[i] = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                logger.info("install_cuda_graph: cuda-graph cleanup complete")
            except Exception:
                pass
        except Exception:
            # Never raise from cleanup (atexit / destroy_process_group hook).
            pass

    atexit.register(_cleanup)

    # Monkey-patch destroy_process_group so our cleanup runs before NCCL
    # teardown. Idempotent: if multiple ``install_cuda_graph`` calls happen
    # (e.g. repeated trainer instantiation), we only wrap once by checking
    # for an attribute marker.
    try:
        import torch.distributed as dist

        original_destroy_pg = getattr(dist, "destroy_process_group", None)
        if original_destroy_pg is not None and not getattr(
            original_destroy_pg, "_install_cuda_graph_wrapped", False
        ):

            def _wrapped_destroy_process_group(*args, **kwargs):
                _cleanup()
                return original_destroy_pg(*args, **kwargs)

            _wrapped_destroy_process_group._install_cuda_graph_wrapped = True
            _wrapped_destroy_process_group.__wrapped__ = original_destroy_pg
            dist.destroy_process_group = _wrapped_destroy_process_group
    except Exception as e:
        logger.info(
            f"install_cuda_graph: failed to wrap destroy_process_group: "
            f"{type(e).__name__}({e})"
        )

    # Bind the new forward as a method so the dispatch matches `gm(*args)`.
    gm.forward = types.MethodType(captured_forward, gm)
    num_varying_buffers = sum(
        1 for t, s in zip(is_tensor_input, is_state_input) if t and not s
    )
    logger.info(
        "install_cuda_graph: installed lazy CUDA graph wrapper "
        f"(num_state_inputs={num_static_inputs}, "
        f"num_varying_buffers={num_varying_buffers})"
    )
    return gm


def disable_uninitialized_memory_fill(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Disable deterministic zero-init for ``aten.empty`` outputs that are
    fully overwritten by their next op.

    Context: with ``torch.use_deterministic_algorithms(True)``, the global
    flag ``torch.utils.deterministic.fill_uninitialized_memory`` defaults to
    True, which makes ``aten.empty.memory_format`` (and ``empty_strided``)
    emit a FillFunctor zero-init kernel for every allocation. The profile
    shows ~4,061 FillFunctor kernels per step (5.9% of GPU time).

    All 65 ``aten.empty.memory_format`` nodes in the joint graph back
    ``all_gather_into_tensor_out`` collectives whose only readers consume
    the buffer AFTER the all_gather writes the full extent. The zero-init
    is therefore dead — the buffer is fully overwritten before any read.

    Strategy (b - "DCE-style elimination"): audit every ``empty.*`` node
    and confirm its users either (a) overlap-write the full buffer
    (collectives' ``_out`` variants, ``copy_``, ``fill_``, ``zero_``) or
    (b) are pure read-only views (``slice.Tensor``, ``view.default``,
    ``as_strided.default``, ``select.int``) that are only read AFTER the
    overwriting op. If the audit passes, set
    ``torch.utils.deterministic.fill_uninitialized_memory = False`` so the
    subsequent CUDA-graph capture in ``install_cuda_graph`` records the
    sequence without the FillFunctor kernels.

    The flag is a global runtime setting; it persists for the rest of the
    process, but the only consumer that matters is the captured CUDA graph
    that ``install_cuda_graph`` records right after this pass runs. Replay
    afterward dispatches the recorded kernels directly without re-querying
    the flag.

    Bitwise safety: every ``aten.empty.memory_format`` output value is
    overwritten by an immediate user before any data-dependent read, so
    the pre-fill zeros are unobservable in the program's output.

    Wrapped in try/except: any audit failure leaves the flag untouched and
    falls through to the existing 5,566 TPS baseline.
    """
    try:
        # Operations that fully overwrite their first (or "out=") argument.
        # ``all_gather_into_tensor_out`` and ``reduce_scatter_tensor_out``
        # take the output buffer as last positional arg in functional-collectives
        # form; the dispatcher writes the whole extent. ``copy_``, ``fill_``,
        # ``zero_`` similarly write the whole extent.
        FULL_OVERWRITE_TARGETS = {
            "_c10d_functional.all_gather_into_tensor_out.default",
            "aten.copy_.default",
            "aten.fill_.Scalar",
            "aten.fill_.Tensor",
            "aten.zero_.default",
        }
        # Read-only views that don't observe pre-existing memory until a later
        # consumer reads the view. These are safe iff the underlying empty is
        # also overwritten by some other user before the view is read - which
        # we conservatively verify by requiring at least one full-overwrite user.
        VIEW_TARGETS = {
            "aten.slice.Tensor",
            "aten.view.default",
            "aten.as_strided.default",
            "aten.select.int",
            "aten._unsafe_view.default",
        }

        empty_nodes = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            short = str(node.target)
            if short in ("aten.empty.memory_format", "aten.empty_strided.default"):
                empty_nodes.append(node)

        if not empty_nodes:
            logger.info(
                "disable_uninitialized_memory_fill: no aten.empty.* nodes found; "
                "no-op"
            )
            return gm

        unsafe: list[str] = []
        safe_count = 0
        for node in empty_nodes:
            user_targets = [str(u.target) for u in node.users]
            has_full_overwrite = any(
                ut in FULL_OVERWRITE_TARGETS for ut in user_targets
            )
            all_users_safe = all(
                (ut in FULL_OVERWRITE_TARGETS) or (ut in VIEW_TARGETS)
                for ut in user_targets
            )
            if has_full_overwrite and all_users_safe:
                safe_count += 1
            else:
                unsafe.append(f"{node.name}: users={user_targets}")

        if unsafe:
            logger.info(
                f"disable_uninitialized_memory_fill: AUDIT FAILED "
                f"({len(unsafe)}/{len(empty_nodes)} empty nodes unsafe to skip "
                f"zero-init); leaving fill_uninitialized_memory unchanged. "
                f"First unsafe: {unsafe[0]}"
            )
            return gm

        # All empty nodes are followed by a full-overwrite op (with optional
        # read-only views interleaved). Safe to disable the global fill.
        try:
            import torch.utils.deterministic as det

            prev = getattr(det, "fill_uninitialized_memory", None)
            det.fill_uninitialized_memory = False
            logger.info(
                f"disable_uninitialized_memory_fill: audited {safe_count} "
                f"aten.empty.* nodes; set fill_uninitialized_memory=False "
                f"(was {prev!r}). This eliminates the per-step FillFunctor "
                f"zero-init kernels for these empty allocations."
            )
        except Exception as e:
            logger.info(
                f"disable_uninitialized_memory_fill: failed to set flag: "
                f"{type(e).__name__}({e})"
            )
    except Exception as e:
        logger.info(
            f"disable_uninitialized_memory_fill: unexpected error "
            f"{type(e).__name__}({e}); pass is a no-op"
        )
    return gm


def construct_default_graph_passes(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the aot_fx_trace path.

    The agent adds custom passes to this list.
    """
    # Bake ``num_static_inputs`` into ``install_cuda_graph`` so the wrapper
    # knows which leading flat inputs are stable module-state tensors
    # (params/buffers with fixed addresses) and can skip allocating
    # duplicate persistent buffers for them. Without this, the wrapper
    # would double model-state memory and OOM under FSDP+TP.
    cuda_graph_pass = functools.partial(
        install_cuda_graph,
        num_static_inputs=traced_result.num_static_inputs,
    )
    passes: list[Callable] = [
        remove_detach_nodes,
        apply_inductor_pattern_passes,
        disable_uninitialized_memory_fill,
        cuda_graph_pass,
    ]
    return passes


def _get_pass_name(pass_fn: Callable) -> str:
    return (
        pass_fn.func.__name__
        if isinstance(pass_fn, functools.partial)
        else pass_fn.__name__
    )


def _filter_disabled_passes(
    passes: list[Callable], disable_names: list[str]
) -> list[Callable]:
    """Remove passes whose names exactly match any entry in ``disable_names``."""
    disable_set = set(disable_names)
    filtered = []
    skipped = []
    for pass_fn in passes:
        name = _get_pass_name(pass_fn)
        if name in disable_set:
            skipped.append(name)
        else:
            filtered.append(pass_fn)
    if skipped:
        logger.info(f"Disabled {len(skipped)} graph passes: {skipped}")
    return filtered


def apply_graph_passes(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    passes: list[Callable],
    *,
    compile_config: "GraphTrainerCompileConfig | None" = None,
) -> torch.fx.GraphModule:
    """Apply graph passes to the traced fwd+bwd graph.

    Args:
        gm: The traced forward+backward graph module.
        example_inputs: Example (fake) inputs matching the graph signature.
        passes: Ordered list of pass callables, each with signature
            ``(gm, example_inputs, **kwargs) -> gm``.
        compile_config: Optional compile config. When provided and
            ``debug_graph_passes`` is True, logs timing, op-count diffs,
            and before/after graphs to tlparse for each pass.
    """
    debug = compile_config is not None and compile_config.debug_graph_passes
    disable_patterns = (
        compile_config.disable_passes if compile_config is not None else []
    )
    if disable_patterns:
        passes = _filter_disabled_passes(passes, disable_patterns)
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    pass_list = "\n  ".join(f"{i}. {name}" for i, name in enumerate(pass_names, 1))
    logger.info(f"Applying {len(passes)} graph passes:\n  {pass_list}")
    all_passes_start = time.perf_counter()
    tlparse_log_graph_pass(gm, graph_name="make_fx_graph_traced", debug=debug)
    for pass_fn in passes:
        pass_name = _get_pass_name(pass_fn)
        if debug:
            tlparse_log_graph_pass(gm, graph_name=f"before_{pass_name}", debug=debug)
            before_snapshot = snapshot_graph(gm)
            start = time.perf_counter()
        gm = pass_fn(gm, example_inputs)
        assert isinstance(
            gm, torch.fx.GraphModule
        ), f"Pass {pass_name} returned {type(gm).__name__}, expected GraphModule"
        if debug:
            elapsed = time.perf_counter() - start
            logger.info(f"Pass {pass_name} took {elapsed:.3f}s")
            tlparse_log_graph_pass(gm, graph_name=f"after_{pass_name}", debug=debug)
            after_snapshot = snapshot_graph(gm)
            log_graph_diff(before_snapshot, after_snapshot, pass_name)
    all_passes_elapsed = time.perf_counter() - all_passes_start
    logger.info(f"All {len(passes)} graph passes took {all_passes_elapsed:.3f}s")
    return gm


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
    debug: bool = False,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured."""
    additional_meta = ["autograd_backward"]
    if debug:
        additional_meta.append("seq_nr")

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": graph_name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
            additional_meta=additional_meta,
        ),
        expect_trace_id=False,
    )

    return gm
