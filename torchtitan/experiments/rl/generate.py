#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference + benchmark script using TorchTitan models with vLLM LLMEngine.

Two modes:

1. Example mode (default): render a single chat prompt and generate from it.

    torchrun --nproc_per_node=4 \
        torchtitan/experiments/rl/generate.py --config rl_grpo_qwen3_30b_a3b_varlen

2. Benchmark mode (--benchmark): measure inference generation throughput.
   - vLLM engine startup is EXCLUDED from timing.
   - The tokenizer is SKIPPED: synthetic token-id prompts are fed to vLLM
     directly (--input-len controls prompt length in tokens).
   - --warmup-runs full generation passes run before timing (default 2) so
     CUDA graph capture / compile do not pollute the measured window.

    # Priority workload from the [3/3] ablation plan (Qwen3-32B, TP=8):
    NCCL_ALGO=Tree torchrun --nproc_per_node=8 \
        torchtitan/experiments/rl/generate.py --benchmark \
        --config rl_grpo_qwen3_32b \
        --model-path /data/users/jianiw/model/Qwen3-32B \
        --batch-size 8 --input-len 1024 --max-tokens 128 \
        --temperature 0 --ignore-eos --compile aot_eager --cudagraph on

The benchmark rungs from the plan map to flags:
  Baseline (eager DTensor, SP off)   --compile off --cudagraph off
  + compile(aot_eager)               --compile aot_eager --cudagraph off
  + cudagraph(FULL)                  --compile aot_eager --cudagraph on
  + tree-based TP all-reduce         (above) plus NCCL_ALGO=Tree (or --nccl-algo Tree)
  + RoPE (Helion kernel)             --rope-kernel helion
  + SiluAndMul (vLLM kernel)         --silu-vllm
  + vLLM RMSNorm                     --rmsnorm-vllm
  + vLLM RoPE kernel                 --rope-kernel vllm
  + remove double-transpose          --no-double-transpose
  Target (vLLM native Qwen3)         --native --compile aot_eager --cudagraph on
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import time

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig, CompilationMode
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import CompileConfig
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.generator import VLLMCudagraphConfig
from torchtitan.experiments.rl.examples.alphabet_sort import config_registry
from torchtitan.experiments.rl.models import kernel_ablation
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.tools.utils import has_cuda_capability


logger = init_logger(__name__)


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/benchmark a TorchTitan RL/vLLM generator config standalone."
    )
    parser.add_argument(
        "--config",
        default="rl_grpo_qwen3_0_6b_varlen",
        help="RL config_registry function to instantiate.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Sort these names alphabetically and put the final answer inside "
            "<alphabetical_sorted>...</alphabetical_sorted>: Charlie, Alice, Bob."
        ),
        help="User prompt to generate from (example mode only).",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send --prompt directly to vLLM instead of rendering a chat prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)

    # ---- Benchmark mode ----
    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark generation throughput (excludes engine startup).",
    )
    bench.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests per generation pass.",
    )
    bench.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Synthetic prompt length in tokens (skips the tokenizer). "
        "Required in benchmark mode.",
    )
    bench.add_argument("--warmup-runs", type=int, default=2)
    bench.add_argument("--num-runs", type=int, default=5)
    bench.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force exactly --max-tokens per request (stable, deterministic timing).",
    )
    bench.add_argument(
        "--profile",
        action="store_true",
        help="Capture a torch profiler chrome trace of one post-warmup pass.",
    )
    bench.add_argument(
        "--profile-dir",
        default="/tmp/ablation_logs/traces",
        help="Directory for profiler chrome traces.",
    )
    bench.add_argument(
        "--profile-tag",
        default=None,
        help="Base name for the trace file (default derived from model/workload).",
    )

    # ---- Config / rung overrides ----
    over = parser.add_argument_group("overrides")
    over.add_argument(
        "--model-path",
        default=None,
        help="Override config.hf_assets_path (HF weights/tokenizer directory).",
    )
    over.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Override tensor_parallel_degree (must match torchrun --nproc_per_node).",
    )
    over.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override model max_seq_len and vLLM max_model_len (for long prompts).",
    )
    over.add_argument(
        "--compile",
        choices=["off", "aot_eager", "inductor"],
        default=None,
        help="Override torch.compile: off, or backend (aot_eager/inductor).",
    )
    over.add_argument(
        "--cudagraph",
        choices=["on", "off"],
        default=None,
        help="Override vLLM CUDA graph capture (full mode).",
    )
    over.add_argument(
        "--cudagraph-mode",
        choices=["full_decode_only", "full", "full_and_piecewise"],
        default="full_decode_only",
        help="vLLM CUDAGraphMode. full_decode_only / full keep torchtitan's "
        "per-layer aot_eager compile (vLLM compile = NONE, decode-only / full "
        "graph capture). full_and_piecewise needs vLLM whole-model compile, so "
        "for the torchtitan path it switches per-layer aot_eager off and lets "
        "vLLM compile (eager backend) instead.",
    )
    over.add_argument(
        "--nccl-algo",
        default=None,
        help="Set NCCL_ALGO (e.g. Tree) for tree-based TP all-reduce.",
    )
    over.add_argument(
        "--native",
        action="store_true",
        help="Benchmark vLLM's native model (Target) instead of the TorchTitan model.",
    )

    # ---- Kernel-patch rungs (TorchTitan model only) ----
    kern = parser.add_argument_group("kernel-rungs")
    kern.add_argument(
        "--rope-kernel",
        choices=["torchtitan", "helion", "vllm"],
        default=None,
        help="RoPE implementation: torchtitan (default), helion (PR #3651), or vllm.",
    )
    kern.add_argument(
        "--silu-vllm",
        action="store_true",
        help="Use vLLM's fused SiluAndMul in the FFN.",
    )
    kern.add_argument(
        "--rmsnorm-vllm",
        action="store_true",
        help="Use vLLM's RMSNorm instead of the TorchTitan/Quack RMSNorm.",
    )
    kern.add_argument(
        "--allreduce-vllm",
        action="store_true",
        help="Run wo/w2 output projections as local matmul + vLLM custom "
        "all-reduce (instead of DTensor NCCL all-reduce).",
    )
    kern.add_argument(
        "--merged-gemm",
        action="store_true",
        help="Merge QKV (3->1) and gate_up (2->1) projections into single "
        "GEMMs (gate_up merge requires --allreduce-vllm).",
    )
    kern.add_argument(
        "--fused-qkv-gateup",
        action="store_true",
        help="FusedQKV + Fused gate_up via the PROPER torchtitan mechanisms: "
        "fuse_qkv=True (FusedQKVLinear) + the fused_swiglu @override. Upstream "
        "equivalent of --merged-gemm; applied at config time before build.",
    )
    kern.add_argument(
        "--fused-addnorm",
        action="store_true",
        help="Fuse the residual-add + ffn RMSNorm via vLLM fused_add_rms_norm.",
    )
    kern.add_argument(
        "--attn-backend",
        choices=["custom", "flash"],
        default="custom",
        help="vLLM attention backend for the torchtitan model: custom (PyTorch "
        "varlen, default) or flash (vLLM FlashAttention/FA3, matching native).",
    )
    kern.add_argument(
        "--dtensor-native",
        action="store_true",
        help="Use register_sharding so kernel custom ops run on DTensors "
        "directly (no per-op to_local/from_local round-trip).",
    )
    kern.add_argument(
        "--embed-allreduce-vllm",
        action="store_true",
        help="Route the vocab-parallel embedding's Partial->Replicate reduction "
        "through vLLM's all-reduce instead of DTensor's NCCL ring (drops the "
        "leftover per-forward NCCL all-reduce).",
    )
    kern.add_argument(
        "--model-2d",
        choices=["local", "localfused", "local3d", "dtensor"],
        default=None,
        help="Replace the model with a native-style path (whole-model, mutually "
        "exclusive with kernel rungs): 'local' = pure-local 2D forward (no "
        "DTensor, mirrors native); 'localfused' = local + native-style residual "
        "fusion (threaded residual, in-place fused add+RMSNorm); 'local3d' = "
        "pure-local (1,T,D) 3D linears (isolates tensor-rank); 'dtensor' = "
        "qwen3_vllm.py 2D-DTensor (register_sharding).",
    )
    kern.add_argument(
        "--no-double-transpose",
        action="store_true",
        help="Use the zero-copy attention layout (skip the double transpose).",
    )
    return parser.parse_args()


def _apply_config_overrides(config, args: argparse.Namespace) -> None:
    """Mutate the RL config in place to reflect CLI rung/override flags."""
    if args.model_path is not None:
        config.hf_assets_path = args.model_path

    if args.max_seq_len is not None:
        # Decoder.Config.max_seq_len is a read-only property derived from each
        # attention layer's rope.max_seq_len, so override the per-layer RoPE
        # config (this also sizes the RoPE cache) rather than the model config.
        model_cfg = config.model_spec.model
        new_layers = []
        for layer in model_cfg.layers:
            attn = getattr(layer, "attention", None)
            if attn is not None and getattr(attn, "rope", None) is not None:
                attn = dataclasses.replace(
                    attn,
                    rope=dataclasses.replace(attn.rope, max_seq_len=args.max_seq_len),
                )
                layer = dataclasses.replace(layer, attention=attn)
            new_layers.append(layer)
        config.model_spec.model = dataclasses.replace(model_cfg, layers=new_layers)

    gen = config.generator
    if args.tp is not None:
        gen.parallelism = dataclasses.replace(
            gen.parallelism, tensor_parallel_degree=args.tp
        )

    if args.compile is not None:
        if args.compile == "off":
            config.compile = CompileConfig(enable=False)
        else:
            config.compile = CompileConfig(enable=True, backend=args.compile)

    if args.cudagraph is not None:
        gen.cudagraph = VLLMCudagraphConfig(enable=(args.cudagraph == "on"))

    if args.fused_qkv_gateup:
        _enable_fused_qkv_gateup(config)


def _enable_fused_qkv_gateup(config) -> None:
    """FusedQKV + Fused gate_up via the PROPER torchtitan mechanisms (not the
    --merged-gemm monkeypatch): rebuild each attention with FusedQKVLinear
    (fuse_qkv=True) and apply the registered ``fused_swiglu`` override to fuse
    the FFN gate+up into one ``w13`` weight.
    """
    from torchtitan.config import apply_overrides, OverrideConfig
    from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
    from torchtitan.models.common.nn_modules import Linear

    model_cfg = config.model_spec.model
    new_layers = []
    for layer in model_cfg.layers:
        attn = getattr(layer, "attention", None)
        qkv = getattr(attn, "qkv_linear", None) if attn is not None else None
        if isinstance(qkv, QKVLinear.Config):
            phd = attn.head_dim or (attn.dim // attn.n_heads)
            n_kv = attn.n_kv_heads or attn.n_heads
            fused = FusedQKVLinear.Config(
                head_dim=phd,
                n_heads=attn.n_heads,
                n_kv_heads=n_kv,
                wqkv=Linear.Config(
                    in_features=qkv.wq.in_features,
                    out_features=(attn.n_heads + 2 * n_kv) * phd,
                    param_init=qkv.wq.param_init,
                ),
            )
            attn = dataclasses.replace(attn, qkv_linear=fused)
            layer = dataclasses.replace(layer, attention=attn)
        new_layers.append(layer)
    config.model_spec.model = dataclasses.replace(model_cfg, layers=new_layers)

    # Fuse gate+up via the registered @override (FeedForward.Config -> FusedSwiGLU).
    apply_overrides(
        OverrideConfig(imports=["torchtitan.overrides.fused_swiglu"]), config
    )


def _resolve_sampling(gen_config, args: argparse.Namespace):
    sampling = gen_config.sampling
    temperature = sampling.temperature if args.temperature is None else args.temperature
    top_p = sampling.top_p if args.top_p is None else args.top_p
    max_tokens = sampling.max_tokens if args.max_tokens is None else args.max_tokens
    return temperature, top_p, max_tokens


def _build_torchtitan_engine(config, args, max_num_seqs: int, *, benchmark: bool):
    """Build a vLLM LLMEngine running the TorchTitan unified model."""
    # Imported lazily: vllm_registry/vllm_wrapper monkeypatch vLLM internals
    # (e.g. _codegen._node_ref) at import time, which breaks vLLM's native
    # inductor compile path used by the --native target. The native engine
    # must never import this module.
    from torchtitan.experiments.rl.models.vllm_registry import (
        registry_to_vllm,
        TORCHTITAN_CONFIG_FORMAT,
    )

    gen_config = config.generator
    model_path = config.hf_assets_path

    # FULL_AND_PIECEWISE needs vLLM's whole-model compile (to split the graph
    # around attention for the piecewise part). The torchtitan model is
    # vLLM-compilable via @support_torch_compile, but its per-layer aot_eager
    # compile would then double-compile, so switch it off and let vLLM compile
    # (eager backend) instead. Kernel patches still apply (they monkeypatch the
    # model before build).
    piecewise = (
        args.cudagraph_mode == "full_and_piecewise" and gen_config.cudagraph.enable
    )
    tt_compile_backend = config.compile.backend
    # --model-2d replaces the whole wrapper.forward + block classes AFTER build,
    # so torchtitan's per-layer torch.compile (applied during build) can't wrap
    # the new forwards -- it recompiles past the limit. Disable per-layer compile
    # for the pure-local paths; cudagraph still captures the eager forward.
    if piecewise or args.model_2d is not None:
        config.compile = CompileConfig(enable=False)

    registry_to_vllm(
        config.model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
    )
    logger.info("Registered TorchTitan model with vLLM")

    inner_attn = config.model_spec.model.layers[0].attention.inner_attention
    if not isinstance(inner_attn, (VarlenAttention.Config, FlexAttention.Config)):
        raise ValueError("Only varlen and flex attention backends are supported.")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    set_batch_invariance(gen_config.debug.batch_invariant)
    if gen_config.debug.batch_invariant:
        from torchtitan.experiments.rl.actors.generator import (
            _patch_bmm_for_batch_invariance,
        )

        _patch_bmm_for_batch_invariance()
    enable_ep = gen_config.parallelism.expert_parallel_degree > 1

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        enable_expert_parallel=enable_ep,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not gen_config.cudagraph.enable,
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(inner_attn, FlexAttention.Config)
                else AttentionBackendEnum.FLASH_ATTN
                if args.attn_backend == "flash"
                else AttentionBackendEnum.CUSTOM
            ),
        ),
        disable_log_stats=False,
    )
    engine_kwargs["max_model_len"] = config.model_spec.model.max_seq_len
    engine_kwargs["max_num_seqs"] = max_num_seqs
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    if benchmark:
        # Avoid cross-request / cross-run prefill dedup so prefill is measured.
        engine_kwargs["enable_prefix_caching"] = False
    if gen_config.cudagraph.enable:
        sizes = sorted(
            {1 << i for i in range(max_num_seqs.bit_length())} | {max_num_seqs}
        )
        if piecewise:
            # FULL_AND_PIECEWISE: vLLM owns whole-model compile (per-layer off).
            vllm_backend = (
                "eager" if tt_compile_backend == "aot_eager" else tt_compile_backend
            )
            engine_kwargs["compilation_config"] = CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend=vllm_backend,
                cudagraph_mode="FULL_AND_PIECEWISE",
                cudagraph_capture_sizes=sizes,
            )
        else:
            # full_decode_only / full: vLLM compile = NONE, so torchtitan's
            # per-layer aot_eager is the only compile; vLLM just captures the
            # decode (full_decode_only) or whole (full) forward as a CUDA graph.
            cg_mode = (
                "FULL_DECODE_ONLY"
                if args.cudagraph_mode == "full_decode_only"
                else "FULL"
            )
            engine_kwargs["compilation_config"] = CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=cg_mode,
                cudagraph_capture_sizes=sizes,
            )
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
    logger.info("vLLM LLMEngine (TorchTitan model) initialized")
    return engine


def _build_native_engine(config, args, max_num_seqs: int, *, benchmark: bool):
    """Build a vLLM LLMEngine running vLLM's own native model (Target)."""
    # Importing the rl package applies vllm_wrapper's DTensor-aware monkeypatch
    # of vLLM's codegen._node_ref, whose signature only accepts 1 arg. That
    # breaks vLLM's native piecewise compile, which calls _node_ref with 3
    # args. The native model does not need the patch, so restore the original.
    from vllm.compilation import codegen as _codegen

    from torchtitan.experiments.rl.models import vllm_wrapper as _ttw

    _codegen._node_ref = _ttw._original_node_ref

    gen_config = config.generator
    model_path = config.hf_assets_path

    cudagraph_on = gen_config.cudagraph.enable
    compile_on = config.compile.enable

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not cudagraph_on,
        disable_log_stats=False,
        max_model_len=config.model_spec.model.max_seq_len,
        max_num_seqs=max_num_seqs,
    )
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    if benchmark:
        engine_kwargs["enable_prefix_caching"] = False

    # vLLM-native compile path. The native model uses vLLM's own torch.compile.
    # vLLM's VLLM_COMPILE only accepts backends "", "eager", "inductor"; its
    # "eager" backend is the no-inductor path (dynamo trace + eager exec, NO
    # inductor fusion passes), which is the fair match for torchtitan's
    # aot_eager. So map --compile aot_eager -> vLLM "eager" (the [1/2]/[3/3]
    # target); --compile inductor -> "inductor" (runs fuse_allreduce_rms etc).
    if compile_on:
        sizes = sorted(
            {1 << i for i in range(max_num_seqs.bit_length())} | {max_num_seqs}
        )
        vllm_backend = (
            "eager"
            if config.compile.backend == "aot_eager"
            else (config.compile.backend)
        )
        if not cudagraph_on:
            cg_mode = "none"
        elif args.cudagraph_mode == "full_and_piecewise":
            cg_mode = "FULL_AND_PIECEWISE"
        elif args.cudagraph_mode == "full_decode_only":
            cg_mode = "FULL_DECODE_ONLY"
        else:
            cg_mode = "FULL"
        engine_kwargs["compilation_config"] = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend=vllm_backend,
            cudagraph_mode=cg_mode,
            cudagraph_capture_sizes=sizes,
        )
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
    logger.info("vLLM LLMEngine (native model) initialized")
    return engine


def _make_token_id_lists(
    vocab_size: int, input_len: int, batch_size: int
) -> list[list[int]]:
    """Build synthetic prompt token-id lists, one per request, distinct across
    requests so prefix caching cannot dedup their prefill. This skips the
    tokenizer entirely: ids are fed to vLLM directly."""
    token_id_lists = []
    # Stay clear of low special-token ids; keep ids well within the vocab.
    span = vocab_size - 1024
    for b in range(batch_size):
        base = (b * 1009) % span + 256
        ids = [base + (i % 257) for i in range(input_len)]
        token_id_lists.append(ids)
    return token_id_lists


def _run_pass(engine, prompts, sampling_params) -> int:
    """Add all prompts as concurrent requests, step until finished, return the
    total number of generated tokens."""
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, sampling_params)
    total_generated = 0
    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if request_output.finished:
                total_generated += len(request_output.outputs[0].token_ids)
    return total_generated


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _profile_one_pass(engine, prompts, sampling_params, config, args) -> None:
    """Capture a torch profiler chrome trace of a single generation pass.

    Writes one trace per rank to ``--profile-dir`` so the torchtitan vs native
    paths can be compared (kernel launch counts, per-kernel CUDA time).
    """
    import os as _os

    from torch.profiler import profile, ProfilerActivity

    rank = int(_os.environ.get("RANK", "0"))
    model_tag = "native" if args.native else "torchtitan"
    name = (
        args.profile_tag
        or f"{model_tag}_bs{args.batch_size}_in{args.input_len}_out"
        f"{args.max_tokens}"
    )
    _os.makedirs(args.profile_dir, exist_ok=True)
    out = _os.path.join(args.profile_dir, f"{name}_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        _run_pass(engine, prompts, sampling_params)
        _sync()
    prof.export_chrome_trace(out)
    if rank == 0:
        print(f"  profiler trace written: {out}", flush=True)


def benchmark(config, args: argparse.Namespace) -> None:
    if args.input_len is None:
        raise ValueError("--input-len is required in --benchmark mode.")

    gen_config = config.generator
    temperature, top_p, max_tokens = _resolve_sampling(gen_config, args)
    batch_size = args.batch_size
    # vLLM must be able to run the whole batch concurrently; this also drives
    # the CUDA-graph capture sizes.
    max_num_seqs = max(args.max_num_seqs, batch_size)
    vocab_size = config.model_spec.model.vocab_size

    if not args.native:
        if args.model_2d is not None:
            from torchtitan.experiments.rl.models.qwen3_vllm_2d import apply_2d_model

            apply_2d_model(args.model_2d)
        else:
            kernel_ablation.apply(
                rope_kernel=args.rope_kernel,
                silu_vllm=args.silu_vllm,
                rmsnorm_vllm=args.rmsnorm_vllm,
                allreduce_vllm=args.allreduce_vllm,
                merged_gemm=args.merged_gemm,
                fused_addnorm=args.fused_addnorm,
                embed_allreduce_vllm=args.embed_allreduce_vllm,
                dtensor_native=args.dtensor_native,
                no_double_transpose=args.no_double_transpose,
            )

    # ---- Engine build (EXCLUDED from timing) ----
    build_t0 = time.perf_counter()
    if args.native:
        engine = _build_native_engine(config, args, max_num_seqs, benchmark=True)
    else:
        engine = _build_torchtitan_engine(config, args, max_num_seqs, benchmark=True)
    _sync()
    build_time = time.perf_counter() - build_t0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        ignore_eos=args.ignore_eos,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    token_id_lists = _make_token_id_lists(vocab_size, args.input_len, batch_size)
    # render_cmpl wraps prompt_token_ids without tokenizing (skips tokenizer).
    prompts = engine.renderer.render_cmpl(
        [{"prompt_token_ids": ids} for ids in token_id_lists]
    )

    # ---- Warmup (EXCLUDED) ----
    for _ in range(args.warmup_runs):
        _run_pass(engine, prompts, sampling_params)
    _sync()

    # ---- Optional profiler trace of ONE post-warmup pass (per rank) ----
    if args.profile:
        _profile_one_pass(engine, prompts, sampling_params, config, args)
        _sync()

    # ---- Timed runs ----
    durations: list[float] = []
    generated_counts: list[int] = []
    for _ in range(args.num_runs):
        _sync()
        t0 = time.perf_counter()
        n = _run_pass(engine, prompts, sampling_params)
        _sync()
        durations.append(time.perf_counter() - t0)
        generated_counts.append(n)

    if not _is_rank0():
        return

    throughputs = [n / d for n, d in zip(generated_counts, durations)]
    mean_tps = statistics.mean(throughputs)
    std_tps = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    model_tag = "vllm-native" if args.native else "torchtitan"
    compile_tag = "off" if not config.compile.enable else config.compile.backend
    cg_tag = "on" if gen_config.cudagraph.enable else "off"

    print("\n" + "=" * 72, flush=True)
    print(f"BENCHMARK  config={args.config}  model={model_tag}", flush=True)
    print(
        f"  TP={gen_config.parallelism.tensor_parallel_degree}  "
        f"compile={compile_tag}  cudagraph={cg_tag}  "
        f"nccl_algo={os.environ.get('NCCL_ALGO', 'default')}",
        flush=True,
    )
    if not args.native:
        print(
            f"  rope={args.rope_kernel or 'torchtitan'}  silu_vllm={args.silu_vllm}  "
            f"rmsnorm_vllm={args.rmsnorm_vllm}  "
            f"no_double_transpose={args.no_double_transpose}",
            flush=True,
        )
    print(
        f"  batch_size={batch_size}  input_len={args.input_len}  "
        f"max_tokens={max_tokens}  ignore_eos={args.ignore_eos}  "
        f"greedy={temperature == 0}",
        flush=True,
    )
    print(
        f"  warmup_runs={args.warmup_runs}  timed_runs={args.num_runs}  "
        f"engine_build={build_time:.1f}s (excluded)",
        flush=True,
    )
    for i, (d, n, tps) in enumerate(zip(durations, generated_counts, throughputs)):
        print(
            f"  run {i}: {d * 1000:8.1f} ms  {n:6d} tok  {tps:8.1f} tok/s",
            flush=True,
        )
    print(
        f"  THROUGHPUT: {mean_tps:.1f} +/- {std_tps:.1f} tok/s "
        f"(per-request: {mean_tps / batch_size:.1f} tok/s)",
        flush=True,
    )
    print("=" * 72 + "\n", flush=True)


def generate_example(config, args: argparse.Namespace) -> None:
    """Single-prompt example generation (default, non-benchmark mode).

    Kernel-rung flags also apply here so greedy output can be diffed against the
    unpatched model to validate a rung's numerics end-to-end.
    """
    gen_config = config.generator
    model_path = config.hf_assets_path
    if args.model_2d is not None:
        from torchtitan.experiments.rl.models.qwen3_vllm_2d import apply_2d_model

        apply_2d_model(args.model_2d)
    else:
        kernel_ablation.apply(
            rope_kernel=args.rope_kernel,
            silu_vllm=args.silu_vllm,
            rmsnorm_vllm=args.rmsnorm_vllm,
            allreduce_vllm=args.allreduce_vllm,
            merged_gemm=args.merged_gemm,
            fused_addnorm=args.fused_addnorm,
            embed_allreduce_vllm=args.embed_allreduce_vllm,
            dtensor_native=args.dtensor_native,
            no_double_transpose=args.no_double_transpose,
        )
    engine = _build_torchtitan_engine(config, args, args.max_num_seqs, benchmark=False)

    renderer = config.renderer.build(tokenizer_path=model_path)
    stop_token_ids = list(renderer.get_stop_token_ids())

    temperature, top_p, max_tokens = _resolve_sampling(gen_config, args)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=stop_token_ids or None,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    prompt = args.prompt
    if args.raw_prompt:
        engine_input = prompt
    else:
        prompt_token_ids = renderer.render_ids(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            add_generation_prompt=True,
        )
        engine_input = engine.renderer.render_cmpl(
            [{"prompt_token_ids": prompt_token_ids}]
        )[0]
        if _is_rank0():
            print(f"Prompt token count: {len(prompt_token_ids)}", flush=True)
            print(f"Stop token ids: {stop_token_ids}", flush=True)
    engine.add_request("0", engine_input, sampling_params)

    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                output_token_ids = request_output.outputs[0].token_ids
                if _is_rank0():
                    print(f"\nConfig: {args.config}", flush=True)
                    print(f"Prompt: {prompt}", flush=True)
                    print(f"Generated token count: {len(output_token_ids)}", flush=True)
                    print(f"Generated text: {generated_text!r}\n", flush=True)


def main() -> None:
    args = _parse_args()

    # NCCL_ALGO must be set before the engine initializes NCCL.
    if args.nccl_algo is not None:
        os.environ["NCCL_ALGO"] = args.nccl_algo

    config_factory = getattr(config_registry, args.config, None)
    if not callable(config_factory):
        raise ValueError(f"Unknown RL config {args.config!r}")
    config = config_factory()
    _apply_config_overrides(config, args)

    if args.benchmark:
        benchmark(config, args)
    else:
        generate_example(config, args)


if __name__ == "__main__":
    main()
