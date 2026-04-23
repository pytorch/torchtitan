#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Debug test: Trace DTensor hook behavior in MoE layers during vLLM inference.

Instruments PrepareModuleInputOutput pre/post hooks, MoE.forward, and the router
gate to trace tensor types (DTensor vs plain vs FakeTensor), shapes, placements,
and data checksums at each stage. Compares profiling pass (FakeTensors) vs actual
inference (real tensors) to find where DTensor hooks corrupt data.

Key tracing points for the MoE data flow:
  1. PRE_HOOK input: the raw input to the MoE module (before all-gather)
  2. PRE_HOOK output: input after Shard(1)->Replicate conversion (after all-gather)
  3. MoE.forward entry: x after PrepareModuleInput (should be DTensor Replicate)
  4. MoE.forward after to_local(): plain tensor (should be identical on all ranks)
  5. Router gate: input/output and cross-rank consistency
  6. GroupedExperts: input, weight shapes, output (partial sum per rank)
  7. MoE.forward output: raw output before POST_HOOK (partial sum, plain tensor)
  8. POST_HOOK: labels output as Partial, redistributes to Shard(1)

Usage:
    PYTHONPATH=/data/users/jianiw/torchtitan \
    PATH="/data/users/jianiw/miniconda3/envs/pytorch-3.12/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda-13.0 \
    VLLM_MOE_BACKEND=triton \
    /data/users/jianiw/miniconda3/envs/pytorch-3.12/bin/python -m torch.distributed.run \
        --nproc_per_node=4 \
        torchtitan/experiments/rl/scripts/test_moe_dtensor_hooks.py 2>&1
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import functools
import types

import torch
import torch.distributed as dist
import torch.distributed.tensor
from torch.distributed._tensor import DTensor
from torch.distributed.tensor import Partial

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Only trace first layer and first N calls to avoid log flood.
# Layer 0 is enough to diagnose the hook behavior.
_TRACE_LAYER = "0"
_MAX_CALLS = 4

# Track which phase we are in (INIT / PROFILE / INFERENCE)
_PHASE = "INIT"

# Store the TP mesh for manual DTensor checks in the output hook
_TP_MESH = None


def _set_phase(phase: str):
    global _PHASE
    _PHASE = phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_fake(t):
    """Check if tensor is a FakeTensor (used during vLLM profiling pass)."""
    if t is None:
        return False
    if hasattr(torch, "_subclasses") and hasattr(torch._subclasses, "FakeTensor"):
        if isinstance(t, torch._subclasses.FakeTensor):
            return True
    if isinstance(t, DTensor):
        return _is_fake(t._local_tensor)
    return False


def _stats(t):
    """Compute quick stats on a tensor for data fingerprinting."""
    if t is None or not t.is_floating_point() or t.numel() == 0:
        return ""
    if _is_fake(t):
        return "FAKE_TENSOR"
    if t.device.type != "cuda":
        return f"device={t.device}"
    try:
        with torch.no_grad():
            f = t.float()
            return (
                f"abssum={f.abs().sum().item():.4f} "
                f"mean={f.mean().item():.6f} "
                f"max={f.abs().max().item():.6f} "
                f"norm={f.norm().item():.4f}"
            )
    except Exception as e:
        return f"stats_err={e}"


def _tinfo(t, name="tensor"):
    """One-line tensor descriptor: type, shape, placement, stats."""
    if t is None:
        return f"{name}=None"
    fake = " FAKE" if _is_fake(t) else ""
    if isinstance(t, DTensor):
        local = t._local_tensor
        return (
            f"{name}: DTensor{fake} placement={t.placements} "
            f"mesh={t.device_mesh.mesh_dim_names} "
            f"full_shape={list(t.shape)} local_shape={list(local.shape)} "
            f"{_stats(local)}"
        )
    return (
        f"{name}: plain{fake} shape={list(t.shape)} dtype={t.dtype} "
        f"device={t.device} {_stats(t)}"
    )


def _cross_rank_check(t, label="tensor"):
    """All-gather a scalar summary to check cross-rank consistency.

    Skipped during profiling pass (FakeTensors can't do real collectives).
    """
    if _PHASE == "PROFILE":
        return
    if t is None or not t.is_floating_point() or t.numel() == 0:
        return
    if _is_fake(t):
        return
    if t.device.type != "cuda":
        return
    with torch.no_grad():
        s = t.float().sum()
        gathered = [torch.zeros_like(s) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, s)
        vals = [g.item() for g in gathered]
        ref = vals[0]
        tol = max(abs(ref) * 1e-3, 1e-6)
        match = all(abs(v - ref) < tol for v in vals)
        rank = dist.get_rank()
        if rank == 0:
            tag = "MATCH" if match else "MISMATCH"
            print(
                f"  [{_PHASE}] {label} cross-rank sums: "
                f"{[f'{v:.4f}' for v in vals]} {tag}",
                flush=True,
            )


def _log(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank{rank}][{_PHASE}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Hook wrappers
# ---------------------------------------------------------------------------


def _patch_hooks_and_forward(model):
    """Wrap PrepareModuleInputOutput hooks and MoE.forward on layer _TRACE_LAYER."""
    from torchtitan.models.common.moe import MoE, GroupedExperts

    for layer_name, layer in model.layers.items():
        if str(layer_name) != _TRACE_LAYER:
            continue
        if not hasattr(layer, "moe") or not isinstance(layer.moe, MoE):
            continue

        moe = layer.moe
        lidx = layer_name

        # ---- Wrap pre_hooks (PrepareModuleInput) ----
        pre_count = {"n": 0}
        for hid, hook_fn in list(moe._forward_pre_hooks.items()):
            orig = hook_fn

            def _make_pre(orig_fn, counts):
                def _traced(module, inputs):
                    counts["n"] += 1
                    n = counts["n"]
                    if n <= _MAX_CALLS:
                        _log(f"[layer{lidx}] PRE_HOOK #{n} ---")
                        if isinstance(inputs, tuple):
                            for i, inp in enumerate(inputs):
                                _log(f"  IN[{i}]: {_tinfo(inp, f'inp{i}')}")
                        else:
                            _log(f"  IN: {_tinfo(inputs, 'inp')}")

                    result = orig_fn(module, inputs)

                    if n <= _MAX_CALLS:
                        if isinstance(result, tuple):
                            for i, r in enumerate(result):
                                _log(f"  OUT[{i}]: {_tinfo(r, f'out{i}')}")
                                if isinstance(r, DTensor):
                                    _cross_rank_check(
                                        r._local_tensor,
                                        f"pre_hook out[{i}] local",
                                    )
                                elif isinstance(r, torch.Tensor):
                                    _cross_rank_check(r, f"pre_hook out[{i}]")
                        elif result is not None:
                            _log(f"  OUT: {_tinfo(result, 'out')}")
                        else:
                            _log(f"  OUT: None (pass-through)")
                    return result
                return _traced

            moe._forward_pre_hooks[hid] = _make_pre(orig, pre_count)

        # ---- Wrap post_hooks (PrepareModuleOutput) ----
        post_count = {"n": 0}
        for hid, hook_fn in list(moe._forward_hooks.items()):
            orig = hook_fn

            def _make_post(orig_fn, counts):
                def _traced(module, inputs, outputs):
                    counts["n"] += 1
                    n = counts["n"]
                    if n <= _MAX_CALLS:
                        _log(f"[layer{lidx}] POST_HOOK #{n} ---")
                        _log(f"  fwd_output (raw): {_tinfo(outputs, 'fwd_out')}")
                        # Raw output is a plain tensor = partial sum. Check across ranks.
                        raw = outputs._local_tensor if isinstance(outputs, DTensor) else outputs
                        _cross_rank_check(raw, "fwd_out (partial?)")

                        # Manually simulate what the hook does to check correctness:
                        # 1) from_local(out, mesh, (Partial(),)) -> wraps as Partial DTensor
                        # 2) redistribute(Partial -> Shard(1)) -> reduce-scatter
                        # Check the intermediate DTensor shape
                        if (
                            isinstance(outputs, torch.Tensor)
                            and not isinstance(outputs, DTensor)
                            and _TP_MESH is not None
                            and not _is_fake(outputs)
                        ):
                            _log(f"  [SIM] Manual from_local check:")
                            _log(f"  [SIM] raw output shape={list(outputs.shape)}")
                            sim_dt = DTensor.from_local(
                                outputs, _TP_MESH, (Partial(),),
                                run_check=False,
                            )
                            _log(
                                f"  [SIM] DTensor(Partial) shape={list(sim_dt.shape)} "
                                f"local_shape={list(sim_dt._local_tensor.shape)}"
                            )
                            # For Partial, full_shape == local_shape
                            if list(sim_dt.shape) != list(outputs.shape):
                                _log(f"  [SIM] WARNING: shape mismatch! "
                                     f"DTensor shape {list(sim_dt.shape)} != "
                                     f"local shape {list(outputs.shape)}")

                    result = orig_fn(module, inputs, outputs)

                    if n <= _MAX_CALLS:
                        _log(f"  hook_result: {_tinfo(result, 'hook_out')}")
                        if isinstance(result, DTensor):
                            _log(
                                f"  hook_result local: {_tinfo(result._local_tensor, 'local')}"
                            )
                            _cross_rank_check(
                                result._local_tensor, "post_hook local"
                            )

                            # Verify: full_tensor should give the correctly reduced result
                            try:
                                full = result.full_tensor()
                                _log(f"  hook_result full_tensor: {_tinfo(full, 'full')}")
                                _cross_rank_check(full, "post_hook full_tensor")
                            except Exception as e:
                                _log(f"  hook_result full_tensor FAILED: {e}")
                        elif isinstance(result, torch.Tensor):
                            _cross_rank_check(result, "post_hook result")
                    return result
                return _traced

            moe._forward_hooks[hid] = _make_post(orig, post_count)

        # ---- Wrap MoE.forward ----
        # Hooks fire via __call__. forward() receives the already-hooked input.
        orig_moe_fwd = type(moe).forward  # unbound
        fwd_count = {"n": 0}

        def _make_moe_fwd(orig, counts, layer_idx):
            def _traced(self_moe, x):
                counts["n"] += 1
                n = counts["n"]
                if n > _MAX_CALLS:
                    return orig(self_moe, x)

                _log(f"[layer{layer_idx}] MoE.forward #{n} ---")
                _log(f"  x (after pre_hook): {_tinfo(x, 'x')}")

                # The DTensor-to-local conversion happens inside forward:
                if isinstance(x, DTensor):
                    x_local = x.to_local()
                    _log(f"  x.to_local(): {_tinfo(x_local, 'x_local')}")
                    _cross_rank_check(x_local, "x after to_local (should match for Replicate)")

                result = orig(self_moe, x)

                _log(f"  MoE raw output: {_tinfo(result, 'out')}")
                # This is BEFORE post_hook. For TP-only, this is partial sum.
                raw = result._local_tensor if isinstance(result, DTensor) else result
                _cross_rank_check(raw, "MoE raw output (should differ for Partial)")

                # Also check: if we manually reduce (sum across ranks), does it look reasonable?
                if raw.is_floating_point() and raw.numel() > 0 and raw.device.type == "cuda":
                    with torch.no_grad():
                        reduced = raw.clone()
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        _log(f"  Manual all-reduce of raw output: {_tinfo(reduced, 'reduced')}")

                return result
            return _traced

        moe.forward = types.MethodType(
            _make_moe_fwd(orig_moe_fwd, fwd_count, lidx), moe
        )

        # ---- Wrap GroupedExperts.forward ----
        experts = moe.experts
        orig_exp_fwd = type(experts).forward
        exp_count = {"n": 0}

        def _make_exp_fwd(orig, counts, layer_idx):
            def _traced(self_exp, x, top_scores, selected_experts_indices, shared_experts=None):
                counts["n"] += 1
                n = counts["n"]
                if n > _MAX_CALLS:
                    return orig(self_exp, x, top_scores, selected_experts_indices, shared_experts=shared_experts)

                _log(f"[layer{layer_idx}] Experts.forward #{n} ---")
                _log(f"  x: {_tinfo(x, 'x')}")
                _log(f"  top_scores: {_tinfo(top_scores, 'scores')}")
                _log(f"  selected_experts: {_tinfo(selected_experts_indices, 'eidx')}")

                # Show expert weight info
                w1 = self_exp.w1
                _log(f"  w1: {_tinfo(w1, 'w1')}")
                if isinstance(w1, DTensor):
                    _log(f"  w1.to_local() shape={list(w1.to_local().shape)}")

                # Check cross-rank consistency of routing
                _cross_rank_check(top_scores, "top_scores")
                if selected_experts_indices.is_floating_point():
                    _cross_rank_check(selected_experts_indices, "selected_experts")
                else:
                    # For int tensors, convert to float for the check
                    _cross_rank_check(
                        selected_experts_indices.float(), "selected_experts (as float)"
                    )

                result = orig(self_exp, x, top_scores, selected_experts_indices, shared_experts=shared_experts)

                _log(f"  Experts output: {_tinfo(result, 'out')}")
                _cross_rank_check(result, "experts output (should differ for TP-sharded)")
                return result
            return _traced

        experts.forward = types.MethodType(
            _make_exp_fwd(orig_exp_fwd, exp_count, lidx), experts
        )

        # ---- Wrap router gate ----
        gate = moe.router.gate
        orig_gate_fwd = gate.forward
        gate_count = {"n": 0}

        def _make_gate_fwd(orig, gate_mod, counts, layer_idx):
            def _traced(x, *args, **kwargs):
                counts["n"] += 1
                n = counts["n"]
                if n > _MAX_CALLS:
                    return orig(x, *args, **kwargs)

                _log(f"[layer{layer_idx}] gate.forward #{n} ---")
                _log(f"  gate input: {_tinfo(x, 'x')}")

                # Show gate weight
                for pn, pp in gate_mod.named_parameters():
                    if "weight" in pn:
                        _log(f"  gate {pn}: {_tinfo(pp, pn)}")
                        break

                _cross_rank_check(
                    x._local_tensor if isinstance(x, DTensor) else x,
                    "gate input",
                )

                result = orig(x, *args, **kwargs)

                _log(f"  gate output: {_tinfo(result, 'out')}")
                out_raw = result._local_tensor if isinstance(result, DTensor) else result
                _cross_rank_check(out_raw, "gate output")
                return result
            return _traced

        gate.forward = _make_gate_fwd(orig_gate_fwd, gate, gate_count, lidx)

        _log(f"Patched layer {lidx} MoE for tracing")
        break  # Only patch the target layer


# ---------------------------------------------------------------------------
# Wrapper forward patch (detects profiling vs inference phase)
# ---------------------------------------------------------------------------


def _patch_wrapper_forward(wrapper):
    """Detect profiling (FakeTensor) vs real inference passes."""
    original_forward = wrapper.forward
    fwd_count = {"n": 0}

    @functools.wraps(original_forward)
    def traced_forward(input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        fwd_count["n"] += 1
        n = fwd_count["n"]
        rank = dist.get_rank()

        is_fake = input_ids is not None and _is_fake(input_ids)
        phase = "PROFILE" if is_fake else "INFERENCE"
        _set_phase(phase)

        if rank == 0 and n <= 10:
            ids_info = (
                f"type={type(input_ids).__name__} shape={list(input_ids.shape)} "
                f"device={input_ids.device}"
                if input_ids is not None
                else "None"
            )
            print(
                f"\n{'='*80}\n"
                f"[{phase}] WRAPPER.forward #{n}\n"
                f"  input_ids: {ids_info} fake={is_fake}\n"
                f"{'='*80}",
                flush=True,
            )

        result = original_forward(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if rank == 0 and n <= 10:
            print(f"[{phase}] WRAPPER.forward #{n} DONE: {_tinfo(result, 'out')}", flush=True)

        return result

    wrapper.forward = traced_forward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from torchtitan.models.qwen3 import model_registry
    from torchtitan.experiments.rl.models.vllm_registry import (
        register_model_to_vllm_model_registry,
        VLLM_MODEL_NAME,
    )
    from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
    from vllm import EngineArgs, LLMEngine, SamplingParams

    model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
    model_spec = model_registry("30B-A3B", attn_backend="varlen")

    # Intercept model __init__ to apply patches after construction + weight load
    original_init = TorchTitanVLLMModelWrapper.__init__

    def patched_init(self, *, model_spec, vllm_config, prefix=""):
        global _TP_MESH
        original_init(self, model_spec=model_spec, vllm_config=vllm_config, prefix=prefix)
        rank = dist.get_rank()

        # Store the TP mesh for later use in hook tracing
        if self.parallel_dims.tp_enabled:
            _TP_MESH = self.parallel_dims.get_mesh("tp")
            if rank == 0:
                print(f"[INIT] TP mesh: {_TP_MESH}", flush=True)

        # ---------- report hooks and param types on layer 0 ----------
        for layer_name, layer in self.model.layers.items():
            if str(layer_name) != _TRACE_LAYER:
                continue
            if not hasattr(layer, "moe"):
                continue
            moe = layer.moe
            if rank == 0:
                print(f"\n{'='*80}", flush=True)
                print(f"[INIT] Layer {layer_name} MoE inspection:", flush=True)
                print(f"  pre_hooks ({len(moe._forward_pre_hooks)}):", flush=True)
                for hid, h in moe._forward_pre_hooks.items():
                    print(f"    [{hid}] {h}", flush=True)
                print(f"  post_hooks ({len(moe._forward_hooks)}):", flush=True)
                for hid, h in moe._forward_hooks.items():
                    print(f"    [{hid}] {h}", flush=True)

                # Gate
                gate = moe.router.gate
                print(f"  gate pre_hooks: {len(gate._forward_pre_hooks)}", flush=True)
                print(f"  gate post_hooks: {len(gate._forward_hooks)}", flush=True)
                for n, p in gate.named_parameters():
                    print(f"  gate.{n}: {_tinfo(p, n)}", flush=True)

                # Expert weights (first 2)
                for n, p in list(moe.experts.named_parameters())[:2]:
                    print(f"  experts.{n}: {_tinfo(p, n)}", flush=True)

                print(f"{'='*80}\n", flush=True)
            break

        # Apply tracing patches
        _patch_hooks_and_forward(self.model)
        _patch_wrapper_forward(self)

        if rank == 0:
            print("[INIT] Debug tracing applied to MoE layer 0", flush=True)

    TorchTitanVLLMModelWrapper.__init__ = patched_init

    register_model_to_vllm_model_registry(model_spec)

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=4,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=0.95,
            max_model_len=256,
            enforce_eager=True,
            hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            attention_backend="CUSTOM",
        )
    )

    rank = dist.get_rank()
    _set_phase("INFERENCE")

    prompts = ["The capital of France is"]
    for i, p in enumerate(prompts):
        engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=10))

    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished and rank == 0:
                print(f"\n{'='*80}", flush=True)
                print(f"RESULT:", flush=True)
                print(f"  Prompt: {o.prompt!r}", flush=True)
                print(f"  Output: {o.outputs[0].text!r}", flush=True)
                print(f"  Token IDs: {list(o.outputs[0].token_ids)}", flush=True)
                print(f"{'='*80}\n", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
