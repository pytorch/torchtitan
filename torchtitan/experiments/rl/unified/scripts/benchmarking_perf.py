#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script to compare inference performance across approaches:
1. vLLM engine with native Qwen3 model (HuggingFace)
2. vLLM native with compile(eager) + cudagraph (TARGET for multi-GPU ablation)
3. vLLM engine with TorchTitan Qwen3 model wrapper
4. Direct TorchTitan Qwen3 model inference

Usage:
    # Run vLLM native vs TorchTitan (default, eager mode)
    torchrun --nproc_per_node=1 -m torchtitan.experiments.rl.unified.scripts.benchmarking_perf \\
        --model-path /path/to/Qwen3-1.7B --tp 1

    # Run with compile(eager) + piecewise cudagraph
    torchrun --nproc_per_node=1 -m torchtitan.experiments.rl.unified.scripts.benchmarking_perf \\
        --model-path /path/to/Qwen3-1.7B --tp 1 --use-cuda-graph

    # Run only vLLM TorchTitan
    torchrun --nproc_per_node=2 -m torchtitan.experiments.rl.unified.scripts.benchmarking_perf \\
        --model-path /path/to/Qwen3-1.7B --tp 2 --test-cases vllm-torchtitan

    # Run with profiling
    torchrun --nproc_per_node=2 -m torchtitan.experiments.rl.unified.scripts.benchmarking_perf \\
        --model-path /path/to/Qwen3-1.7B --tp 2 --profile

    # Run all benchmarks (requires TorchTitan checkpoint for native)
    torchrun --nproc_per_node=2 -m torchtitan.experiments.rl.unified.scripts.benchmarking_perf \\
        --model-path /path/to/Qwen3-1.7B --tp 2 --test-cases vllm-native,vllm-torchtitan,torchtitan-native \\
        --torchtitan-checkpoint /path/to/checkpoint

    # Multi-GPU ablation study (run each step separately, same test case, different configs):
    # Target: vLLM native with compile+cudagraph
    torchrun --nproc_per_node=4 ... --tp 4 --use-cuda-graph --test-cases vllm-native

    # Baseline: TorchTitan eager + SP
    torchrun --nproc_per_node=4 ... --tp 4 --test-cases vllm-torchtitan

    # Ablation 1: + compile(eager) + cudagraph
    torchrun --nproc_per_node=4 ... --tp 4 --use-cuda-graph --test-cases vllm-torchtitan

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from typing import Any

import numpy as np

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.profiler import profile, ProfilerActivity, schedule


@dataclass
class _BenchmarkRLConfig:
    """Lightweight config mimicking the fields the benchmark reads from RLTrainer.Config.

    Avoids importing config_registry (which pulls in torchstore/monarch).
    """

    model_spec: Any
    generator: Any
    hf_assets_path: str = ""


def _get_rl_config(model_size: str):
    """Build a benchmark-oriented RL config for the given Qwen3 model size."""
    from torchtitan.config.configs import DebugConfig, ParallelismConfig
    from torchtitan.models.qwen3 import model_registry
    from vllm.config import CompilationConfig

    model_spec = model_registry(model_size, attn_backend="varlen")

    @dataclass(kw_only=True, slots=True)
    class _GeneratorCompileConfig:
        backend: str = "eager"
        cudagraph_mode: str = "piecewise"

        @property
        def is_eager(self) -> bool:
            return self.backend == "none" and self.cudagraph_mode == "none"

        def get_vllm_compilation_config(self) -> CompilationConfig | None:
            if self.is_eager:
                return None
            kwargs: dict = dict(cudagraph_mode=self.cudagraph_mode)
            if self.backend == "none":
                kwargs["mode"] = 0
            else:
                kwargs["backend"] = self.backend
            return CompilationConfig(**kwargs)

    @dataclass(kw_only=True, slots=True)
    class _SamplingConfig:
        temperature: float = 0.8
        top_p: float = 0.95
        max_tokens: int = 100

    @dataclass(kw_only=True, slots=True)
    class _GeneratorConfig:
        model_dtype: str = "bfloat16"
        gpu_memory_limit: float = 0.9
        compile: _GeneratorCompileConfig = field(
            default_factory=_GeneratorCompileConfig
        )
        sampling: _SamplingConfig = field(default_factory=_SamplingConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    hf_paths = {
        "0.6B": "torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        "1.7B": "torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        "14B": "torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
    }

    return _BenchmarkRLConfig(
        model_spec=model_spec,
        generator=_GeneratorConfig(
            compile=_GeneratorCompileConfig(backend="eager", cudagraph_mode="piecewise"),
            sampling=_SamplingConfig(temperature=0.8, top_p=0.95, max_tokens=100),
        ),
        hf_assets_path=hf_paths.get(model_size, ""),
    )


# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking."""

    approach: str
    total_time: float
    tokens_generated: int
    prefill_time: float
    decode_time: float
    throughput_tokens_per_sec: float
    latency_per_token_ms: float
    first_token_latency_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    peak_memory_gb: float
    batch_size: int
    sequence_length: int
    num_prompts: int


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    model_path: str
    prompts_file: str
    model_size: str = "1.7B"  # Model size key for registry (e.g. "0.6B", "1.7B", "4B", "8B", "14B", "32B")
    torchtitan_checkpoint_path: str = None
    torchtitan_config_path: str = None
    from_hf: bool = True
    tp: int = 1
    batch_size: int = 1
    max_tokens: int = 512
    num_runs: int = 5
    warmup_runs: int = 2
    temperature: float | None = None  # None = use gen_config default
    top_p: float | None = None  # None = use gen_config default
    device: str = "cuda"
    use_cuda_graph: bool = False
    compile_backend: str = "eager"  # "eager" or "inductor" for vLLM compile backend
    # NCCL tuning
    nccl_algo: str | None = None  # e.g. "Tree" to force tree-based all-reduce
    # Profiling options
    profile: bool = False
    profile_dir: str = "./profiler_traces"
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 2
    profile_repeat: int = 1


class ProfilerManager:
    """Helper class to manage PyTorch profiler for benchmarks."""

    def __init__(self, config: BenchmarkConfig, approach_name: str):
        self.config = config
        self.approach_name = approach_name
        self.profiler: torch.profiler.profile | None = None

    def get_trace_dir(self) -> str:
        trace_dir = Path(self.config.profile_dir) / self.approach_name.lower().replace(
            " ", "_"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        return str(trace_dir)

    def create_profiler(self) -> torch.profiler.profile | None:
        if not self.config.profile:
            return None

        trace_dir = self.get_trace_dir()
        print(f"  Profiler traces will be saved to: {trace_dir}")

        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.config.profile_wait,
                warmup=self.config.profile_warmup,
                active=self.config.profile_active,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        return self.profiler

    def create_simple_profiler(self) -> torch.profiler.profile | None:
        if not self.config.profile:
            return None

        trace_dir = self.get_trace_dir()
        print(f"  Profiler traces will be saved to: {trace_dir}")

        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        return self.profiler

    def export_chrome_trace(self, run_idx: int):
        if self.profiler is None:
            return

        trace_dir = self.get_trace_dir()
        trace_file = Path(trace_dir) / f"trace_run_{run_idx}.json"
        self.profiler.export_chrome_trace(str(trace_file))
        print(f"  Exported Chrome trace to: {trace_file}")

    def print_summary(self):
        if self.profiler is None:
            return

        print(f"\n  === Profiler Summary for {self.approach_name} ===")
        print(
            self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        )


class VLLMNativeBenchmark:
    """Benchmark vLLM with native Qwen3 model from HuggingFace.

    Uses the same engine settings as VLLMTorchTitanBenchmark (loaded from
    rl_grpo_qwen3_1_7b()) so the two approaches are directly comparable.
    The only difference is the model architecture (native HF vs TorchTitan).
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None
        self.profiling_enabled = False
        self.profile_dir = None

    def setup(self):
        try:
            from vllm import LLM, SamplingParams

            # ── 1. Load RL config (same source of truth as VLLMTorchTitanBenchmark) ──
            rl_config = _get_rl_config(self.config.model_size)
            gen_config = rl_config.generator
            model_path = self.config.model_path or rl_config.hf_assets_path

            # Override compile backend if specified
            if self.config.compile_backend != "eager":
                gen_config.compile.backend = self.config.compile_backend

            use_cuda_graph = self.config.use_cuda_graph
            backend = gen_config.compile.backend
            mode_str = (
                f"compile({backend}) + piecewise cudagraph"
                if use_cuda_graph
                else "eager (no compile, no cudagraph)"
            )

            print("Loading vLLM with native Qwen3 model from HuggingFace...")
            print(f"Model: {model_path}")
            print(f"Tensor Parallel Size: {self.config.tp}")
            print(f"Mode: {mode_str}")

            if self.config.profile:
                profile_name = (
                    f"vllm_native_{self.config.model_size}"
                    f"_tp{self.config.tp}"
                    f"_bsz{self.config.batch_size}"
                    f"_maxtok{self.config.max_tokens}"
                )
                self.profile_dir = Path(self.config.profile_dir) / profile_name
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                os.environ["VLLM_TORCH_PROFILER_DIR"] = str(self.profile_dir)
                self.profiling_enabled = True
                print(
                    f"Profiling ENABLED - traces will be saved to: {self.profile_dir}"
                )

            # Build profiler_config for vLLM versions that require it
            profiler_config_kwargs = {}
            if self.profiling_enabled:
                try:
                    from vllm.config import ProfilerConfig

                    profiler_config_kwargs["profiler_config"] = ProfilerConfig(
                        profiler="torch",
                        torch_profiler_dir=str(self.profile_dir.resolve()),
                    )
                except ImportError:
                    pass  # older vLLM, env var fallback

            # ── 2. Build engine kwargs (identical to VLLMTorchTitanBenchmark) ──
            engine_kwargs = dict(
                model=model_path,
                trust_remote_code=True,
                dtype=gen_config.model_dtype,
                tensor_parallel_size=self.config.tp,
                distributed_executor_backend="external_launcher",
                gpu_memory_utilization=gen_config.gpu_memory_limit,
                enforce_eager=not use_cuda_graph,
            )

            if use_cuda_graph:
                vllm_compilation_config = (
                    gen_config.compile.get_vllm_compilation_config()
                )
                if vllm_compilation_config is not None:
                    engine_kwargs["compilation_config"] = vllm_compilation_config

            if gen_config.debug.seed is not None:
                engine_kwargs["seed"] = gen_config.debug.seed

            self.engine = LLM(**engine_kwargs, **profiler_config_kwargs)

            # ── 3. Sampling params (same as VLLMTorchTitanBenchmark) ──
            sampling = gen_config.sampling
            # Force greedy decoding to eliminate sampling kernel overhead
            # (_topk_topp_kernel) from throughput measurements.
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=self.config.max_tokens or sampling.max_tokens,
            )
            print("✓ vLLM native Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM native model: {e}")
            raise

    def run_inference(
        self, prompts: list[str], use_profiler: bool = False
    ) -> BenchmarkMetrics:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        profile_started = False
        if use_profiler and self.profiling_enabled:
            print(f"  Starting vLLM profiler (traces will go to {self.profile_dir})...")
            if hasattr(self.engine, "start_profile"):
                try:
                    self.engine.start_profile()
                    profile_started = True
                    print("  Profiler started successfully")
                except Exception as e:
                    print(f"  Warning: start_profile() failed: {e}")

        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        if profile_started:
            try:
                self.engine.stop_profile()
                print(
                    f"  Profiler stopped. Traces should be written to: {self.profile_dir}"
                )
            except Exception as e:
                print(f"  Warning: stop_profile() failed: {e}")

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        approach_name = (
            "vLLM Native [TARGET: compile+cudagraph]"
            if self.config.use_cuda_graph
            else "vLLM Native"
        )
        return BenchmarkMetrics(
            approach=approach_name,
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=0.0,
            decode_time=0.0,
            throughput_tokens_per_sec=total_tokens / total_time,
            latency_per_token_ms=(total_time / total_tokens) * 1000,
            first_token_latency_ms=0.0,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class VLLMTorchTitanBenchmark:
    """Benchmark vLLM with TorchTitan Qwen3 model via plugin registration.

    Aligned with inference_example.py: uses rl_grpo_qwen3_1_7b() config as the
    source of truth, with CLI arguments overriding where applicable.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None
        self.profiling_enabled = False
        self.profile_dir = None

    def setup(self):
        try:
            from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
            from torchtitan.experiments.rl.plugin import (
                register_model_to_vllm_model_registry,
                VLLM_MODEL_NAME,
            )

            from vllm import LLM, SamplingParams

            # ── 1. Load RL config (same as inference_example.py) ──
            rl_config = _get_rl_config(self.config.model_size)
            gen_config = rl_config.generator
            # CLI --model-path overrides the config's hf_assets_path
            model_path = self.config.model_path or rl_config.hf_assets_path

            # Override compile backend if specified
            if self.config.compile_backend != "eager":
                gen_config.compile.backend = self.config.compile_backend

            use_cuda_graph = self.config.use_cuda_graph
            backend = gen_config.compile.backend
            mode_str = (
                f"compile({backend}) + piecewise cudagraph"
                if use_cuda_graph
                else "eager (no compile, no cudagraph)"
            )
            print("Loading vLLM with TorchTitan Qwen3 model...")
            print(f"Model: {model_path}")
            print(f"Tensor Parallel Size: {self.config.tp}")
            print(f"Mode: {mode_str}")

            if self.config.profile:
                compile_tag = "compile_cg" if self.config.use_cuda_graph else "eager"
                profile_name = (
                    f"vllm_torchtitan_{self.config.model_size}"
                    f"_tp{self.config.tp}_{compile_tag}"
                    f"_bsz{self.config.batch_size}"
                    f"_maxtok{self.config.max_tokens}"
                )
                self.profile_dir = Path(self.config.profile_dir) / profile_name
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                os.environ["VLLM_TORCH_PROFILER_DIR"] = str(self.profile_dir)
                self.profiling_enabled = True
                print(
                    f"Profiling ENABLED - traces will be saved to: {self.profile_dir}"
                )

            # Build profiler_config for vLLM versions that require it
            profiler_config_kwargs = {}
            if self.profiling_enabled:
                try:
                    from vllm.config import ProfilerConfig

                    profiler_config_kwargs["profiler_config"] = ProfilerConfig(
                        profiler="torch",
                        torch_profiler_dir=str(self.profile_dir.resolve()),
                    )
                except ImportError:
                    pass  # older vLLM, env var fallback

            rl_config.model_spec.parallelize_fn = parallelize_qwen3
            register_model_to_vllm_model_registry(rl_config.model_spec)

            # ── 3. Build engine kwargs (mirroring inference_example.py) ──
            engine_kwargs = dict(
                model=model_path,
                trust_remote_code=True,
                dtype=gen_config.model_dtype,
                tensor_parallel_size=self.config.tp,
                distributed_executor_backend="external_launcher",
                gpu_memory_utilization=gen_config.gpu_memory_limit,
                enforce_eager=not use_cuda_graph,
                hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            )

            if use_cuda_graph:
                vllm_compilation_config = (
                    gen_config.compile.get_vllm_compilation_config()
                )
                if vllm_compilation_config is not None:
                    engine_kwargs["compilation_config"] = vllm_compilation_config

            if gen_config.debug.seed is not None:
                engine_kwargs["seed"] = gen_config.debug.seed

            self.engine = LLM(**engine_kwargs, **profiler_config_kwargs)

            # ── 4. Sampling params (from gen_config, with CLI overrides) ──
            sampling = gen_config.sampling
            # Force greedy decoding to eliminate sampling kernel overhead
            # (_topk_topp_kernel) from throughput measurements.
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=self.config.max_tokens or sampling.max_tokens,
            )
            print("✓ vLLM TorchTitan Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM TorchTitan model: {e}")
            import traceback

            traceback.print_exc()
            raise

    def run_inference(
        self, prompts: list[str], use_profiler: bool = False
    ) -> BenchmarkMetrics:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        profile_started = False
        if use_profiler and self.profiling_enabled:
            if hasattr(self.engine, "start_profile"):
                try:
                    self.engine.start_profile()
                    profile_started = True
                    print("Profiler started successfully")
                except Exception as e:
                    print(f"  Warning: start_profile() failed: {e}")

        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        if profile_started:
            try:
                self.engine.stop_profile()
            except Exception as e:
                print(f"  Warning: stop_profile() failed: {e}")

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        compile_mode = "compile+cudagraph" if self.config.use_cuda_graph else "eager"
        return BenchmarkMetrics(
            approach=f"vLLM TorchTitan ({compile_mode})",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=0.0,
            decode_time=0.0,
            throughput_tokens_per_sec=total_tokens / total_time
            if total_time > 0
            else 0,
            latency_per_token_ms=(total_time / total_tokens) * 1000
            if total_tokens > 0
            else 0,
            first_token_latency_ms=0.0,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class TorchTitanNativeBenchmark:
    """Benchmark direct TorchTitan Qwen3 model inference."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        try:
            print("Loading TorchTitan Qwen3 model directly...")

            import torch.distributed as dist
            import torch.distributed.checkpoint as dcp
            from torchtitan.config import ConfigManager
            from torchtitan.distributed import ParallelDims
            from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
            from torchtitan.protocols.train_spec import get_train_spec

            env_world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if self.config.tp > 1:
                if env_world_size == 1:
                    raise RuntimeError(
                        f"TP={self.config.tp} requested but running in single process mode. "
                        f"Use: torchrun --nproc_per_node={self.config.tp} ..."
                    )
                if env_world_size != self.config.tp:
                    raise RuntimeError(
                        f"TP={self.config.tp} but torchrun launched {env_world_size} processes."
                    )

            world_size = self.config.tp

            print(
                f"TP={self.config.tp}, "
                f"World Size: {world_size}, Local Rank: {local_rank}"
            )

            if not dist.is_initialized():
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "29500")
                if world_size > 1:
                    dist.init_process_group(
                        backend="nccl" if torch.cuda.is_available() else "gloo",
                        init_method="env://",
                    )
                else:
                    os.environ.setdefault("RANK", "0")
                    os.environ.setdefault("WORLD_SIZE", "1")
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://",
                    )

            # Find config file
            if self.config.torchtitan_config_path:
                config_path = self.config.torchtitan_config_path
            else:
                config_search_paths = [
                    Path("./train_configs/qwen3_1.7b.toml"),
                    Path("./torchtitan/models/qwen3/train_configs/qwen3_1.7b.toml"),
                    Path("/data/users/jianiw/torchtitan/train_configs/qwen3_1.7b.toml"),
                    Path(
                        "/data/users/jianiw/torchtitan/torchtitan/models/qwen3/train_configs/debug_model.toml"
                    ),
                    Path("./torchtitan/models/qwen3/train_configs/debug_model.toml"),
                ]
                config_path = None
                for path in config_search_paths:
                    if path.exists():
                        config_path = str(path)
                        break
                if config_path is None:
                    raise RuntimeError(
                        "Could not find model config file. Please specify --torchtitan-config\n"
                        "Searched paths:\n"
                        + "\n".join(f"  - {p}" for p in config_search_paths)
                    )

            print(f"Using config: {config_path}")

            config_manager = ConfigManager()
            tt_config = config_manager.parse_args([f"--job.config_file={config_path}"])

            train_spec = get_train_spec(tt_config.model.name)
            self.tokenizer = train_spec.build_tokenizer_fn(tt_config)

            model_args = train_spec.model_args[tt_config.model.flavor]
            model_args.update_from_config(tt_config)

            if world_size > 1:
                parallel_dims = ParallelDims(
                    dp_replicate=1,
                    dp_shard=-1,
                    cp=1,
                    tp=world_size,
                    pp=1,
                    ep=1,
                    etp=1,
                    world_size=world_size,
                )
            else:
                parallel_dims = ParallelDims(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=1,
                    pp=1,
                    ep=1,
                    etp=1,
                    world_size=1,
                )

            device = torch.device(f"{self.config.device}:{local_rank}")
            init_device = "meta" if world_size > 1 else device

            print(f"Initializing model on device: {init_device}")
            with torch.device(init_device):
                self.model = train_spec.model_cls(model_args)

            if world_size > 1:
                print(f"Applying parallelism with TP={world_size}")
                from torchtitan.config import ParallelismConfig

                parallelism_config = ParallelismConfig()
                parallelize_qwen3(
                    self.model,
                    parallel_dims=parallel_dims,
                    parallelism=parallelism_config,
                )

            # Materialize model
            self.model.to_empty(device=device)
            with torch.no_grad():
                self.model.init_weights()
            self.model.eval()

            # Load checkpoint
            print(
                f"Loading checkpoint from {self.config.torchtitan_checkpoint_path}..."
            )
            checkpoint_path = Path(self.config.torchtitan_checkpoint_path)

            if self.config.from_hf:
                print("Loading from HuggingFace format using state dict adapter...")
                sd_adapter = train_spec.state_dict_adapter(model_args, None)
                assert (
                    sd_adapter is not None
                ), "Trying to load HF checkpoint, but sd_adapter is not available."

                state_dict = get_model_state_dict(self.model)
                hf_state_dict = sd_adapter.to_hf(state_dict)
                dcp.load(
                    hf_state_dict,
                    storage_reader=HuggingFaceStorageReader(path=str(checkpoint_path)),
                )
                state_dict = sd_adapter.from_hf(hf_state_dict)
                set_model_state_dict(
                    self.model,
                    model_state_dict=state_dict,
                    options=StateDictOptions(strict=False),
                )
            else:
                state_dict = get_model_state_dict(self.model)
                dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
                set_model_state_dict(
                    self.model,
                    model_state_dict=state_dict,
                    options=StateDictOptions(strict=False),
                )

            # Apply torch.compile if enabled
            if self.config.use_cuda_graph:
                print("Applying torch.compile to TorchTitan native model...")
                self.model = torch.compile(self.model, backend="inductor")

            if local_rank == 0:
                print("✓ TorchTitan native Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load TorchTitan native model: {e}")
            import traceback

            traceback.print_exc()
            raise

    @torch.inference_mode()
    def generate_with_timing(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> tuple[torch.Tensor, float, float]:
        """Generate tokens with separate prefill/decode timing."""
        device = torch.device(self.config.device)
        temperature = (
            self.config.temperature if self.config.temperature is not None else 0.0
        )

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device)
        generated_tokens = input_ids.clone()

        # Prefill
        prefill_start = time.perf_counter()
        logits = self.model(generated_tokens)
        probs = torch.nn.functional.softmax(
            logits[:, -1, :] / max(temperature, 1e-5), dim=-1
        )

        if temperature == 0.0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Decode
        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            logits = self.model(generated_tokens)
            probs = torch.nn.functional.softmax(
                logits[:, -1, :] / max(temperature, 1e-5), dim=-1
            )

            if temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start

        return generated_tokens, prefill_time, decode_time

    def run_inference(
        self, prompts: list[str], profiler: torch.profiler.profile | None = None
    ) -> BenchmarkMetrics:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        total_tokens = 0
        total_prefill_time = 0
        total_decode_time = 0
        first_token_latency = None

        start_time = time.perf_counter()

        def run_generation():
            nonlocal total_tokens, total_prefill_time, total_decode_time, first_token_latency

            for prompt in prompts:
                if hasattr(self.tokenizer, "encode"):
                    try:
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt, add_bos=True, add_eos=False),
                            dtype=torch.long,
                        )
                    except TypeError:
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt), dtype=torch.long
                        )
                else:
                    input_ids = torch.tensor(
                        self.tokenizer.encode(prompt), dtype=torch.long
                    )

                output_tokens, prefill_time, decode_time = self.generate_with_timing(
                    input_ids, self.config.max_tokens
                )

                if first_token_latency is None:
                    first_token_latency = prefill_time * 1000

                num_generated = output_tokens.size(1) - input_ids.size(0)
                total_tokens += num_generated
                total_prefill_time += prefill_time
                total_decode_time += decode_time

        if profiler is not None:
            with profiler:
                run_generation()
                profiler.step()
        else:
            run_generation()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return BenchmarkMetrics(
            approach="TorchTitan Native",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=total_prefill_time,
            decode_time=total_decode_time,
            throughput_tokens_per_sec=total_tokens / total_time
            if total_time > 0
            else 0,
            latency_per_token_ms=(total_time / total_tokens) * 1000
            if total_tokens > 0
            else 0,
            first_token_latency_ms=first_token_latency or 0,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: dict[str, list[BenchmarkMetrics]] = {
            "vllm_native": [],
            "vllm_torchtitan": [],
            "torchtitan_native": [],
        }

    def load_prompts(self) -> list[str]:
        prompts_path = Path(self.config.prompts_file)
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = [
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the benefits of using Python for data science?",
                "Describe the process of photosynthesis.",
                "How does a neural network work?",
            ]

        # Repeat prompts to fill batch size for benchmarking
        batch_size = self.config.batch_size
        if len(prompts) < batch_size:
            prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
        return prompts[:batch_size]

    def run_benchmark(self, benchmark_cls, key: str):
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {key}")
        if self.config.use_cuda_graph:
            print("Mode: compile(eager) + piecewise cudagraph")
        else:
            print("Mode: eager (no compile, no cudagraph)")
        if self.config.profile:
            print(
                f"Profiling ENABLED - traces will be saved to: {self.config.profile_dir}"
            )
        print(f"{'=' * 60}")

        try:
            benchmark = benchmark_cls(self.config)
            benchmark.setup()

            # Set NCCL algorithm after setup to avoid breaking broadcast
            # during DTensor weight replication. NCCL picks this up for
            # new communicators / operations created after this point.
            if self.config.nccl_algo:
                os.environ["NCCL_ALGO"] = self.config.nccl_algo
                print(f"Set NCCL_ALGO={self.config.nccl_algo} (post-setup)")

            prompts = self.load_prompts()

            is_vllm_benchmark = key.startswith("vllm_")

            # Warmup runs
            print(f"Running {self.config.warmup_runs} warmup iterations...")
            for i in range(self.config.warmup_runs):
                benchmark.run_inference(prompts)
                print(f"  Warmup {i + 1}/{self.config.warmup_runs} completed")

            # Benchmark runs
            print(f"Running {self.config.num_runs} benchmark iterations...")

            if self.config.profile:
                if is_vllm_benchmark:
                    profile_runs = 1
                    print(f"  (Profiling mode: running {profile_runs} iteration only)")
                    for i in range(profile_runs):
                        metrics = benchmark.run_inference(prompts, use_profiler=True)
                        self.results[key].append(metrics)
                        print(
                            f"  Run {i + 1}/{profile_runs} (profiled): "
                            f"{metrics.throughput_tokens_per_sec:.2f} tokens/s, "
                            f"latency: {metrics.latency_per_token_ms:.2f} ms/token"
                        )
                else:
                    approach_name = key.replace("_", " ").title()
                    profiler_manager = ProfilerManager(self.config, approach_name)
                    profiler = profiler_manager.create_simple_profiler()

                    if profiler is not None:
                        profiler.__enter__()

                    for i in range(self.config.num_runs):
                        metrics = benchmark.run_inference(prompts)
                        self.results[key].append(metrics)
                        print(
                            f"  Run {i + 1}/{self.config.num_runs}: "
                            f"{metrics.throughput_tokens_per_sec:.2f} tokens/s, "
                            f"latency: {metrics.latency_per_token_ms:.2f} ms/token"
                        )
                        if profiler is not None:
                            profiler.step()

                    if profiler is not None:
                        profiler.__exit__(None, None, None)
                        profiler_manager.export_chrome_trace(0)
                        profiler_manager.print_summary()
            else:
                for i in range(self.config.num_runs):
                    metrics = benchmark.run_inference(prompts)
                    self.results[key].append(metrics)
                    print(
                        f"  Run {i + 1}/{self.config.num_runs}: "
                        f"{metrics.throughput_tokens_per_sec:.2f} tokens/s, "
                        f"latency: {metrics.latency_per_token_ms:.2f} ms/token"
                    )

            benchmark.cleanup()
            print(f"✓ {key} benchmark completed")

        except Exception as e:
            print(f"✗ {key} benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    def compute_statistics(
        self, metrics_list: list[BenchmarkMetrics]
    ) -> dict[str, Any]:
        if not metrics_list:
            return {}

        throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
        latencies = [m.latency_per_token_ms for m in metrics_list]
        first_token_latencies = [m.first_token_latency_ms for m in metrics_list]
        memory_peaks = [m.peak_memory_gb for m in metrics_list]

        return {
            "approach": metrics_list[0].approach,
            "throughput": {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
                "median": np.median(throughputs),
            },
            "latency_per_token_ms": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "median": np.median(latencies),
            },
            "first_token_latency_ms": {
                "mean": np.mean(first_token_latencies),
                "std": np.std(first_token_latencies),
                "min": np.min(first_token_latencies),
                "max": np.max(first_token_latencies),
                "median": np.median(first_token_latencies),
            },
            "peak_memory_gb": {
                "mean": np.mean(memory_peaks),
                "std": np.std(memory_peaks),
                "min": np.min(memory_peaks),
                "max": np.max(memory_peaks),
                "median": np.median(memory_peaks),
            },
            "num_runs": len(metrics_list),
            "total_tokens_generated": sum(m.tokens_generated for m in metrics_list),
        }

    def print_summary(self):
        print(f"\n{'=' * 80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'=' * 80}\n")

        for key, metrics_list in self.results.items():
            if not metrics_list:
                continue

            stats = self.compute_statistics(metrics_list)

            print(stats["approach"])
            print("-" * 80)
            print("  Throughput (tokens/sec):")
            print(
                f"    Mean:   {stats['throughput']['mean']:>10.2f} ± {stats['throughput']['std']:.2f}"
            )
            print(f"    Median: {stats['throughput']['median']:>10.2f}")
            print(
                f"    Range:  {stats['throughput']['min']:>10.2f} - {stats['throughput']['max']:.2f}"
            )
            print("\n  Latency (ms/token):")
            print(
                f"    Mean:   {stats['latency_per_token_ms']['mean']:>10.2f} ± {stats['latency_per_token_ms']['std']:.2f}"
            )
            print(f"    Median: {stats['latency_per_token_ms']['median']:>10.2f}")
            print(
                f"    Range:  {stats['latency_per_token_ms']['min']:>10.2f} - {stats['latency_per_token_ms']['max']:.2f}"
            )
            print("\n  First Token Latency (ms):")
            print(
                f"    Mean:   {stats['first_token_latency_ms']['mean']:>10.2f} ± {stats['first_token_latency_ms']['std']:.2f}"
            )
            print("\n  Peak Memory (GB):")
            print(
                f"    Mean:   {stats['peak_memory_gb']['mean']:>10.2f} ± {stats['peak_memory_gb']['std']:.2f}"
            )
            print(f"\n  Total tokens: {stats['total_tokens_generated']:,}")
            print(f"  Runs: {stats['num_runs']}")
            print()

        # Comparison
        if len(self.results) > 1:
            print("=" * 80)
            print("RELATIVE PERFORMANCE")
            print("=" * 80 + "\n")

            baseline_key = "vllm_native"
            if self.results[baseline_key]:
                baseline_stats = self.compute_statistics(self.results[baseline_key])
                baseline_throughput = baseline_stats["throughput"]["mean"]

                for key, metrics_list in self.results.items():
                    if not metrics_list or key == baseline_key:
                        continue

                    stats = self.compute_statistics(metrics_list)
                    throughput = stats["throughput"]["mean"]
                    speedup = throughput / baseline_throughput

                    print(f"{stats['approach']} vs {baseline_stats['approach']}:")
                    print(f"  Speedup: {speedup:.2f}x ({speedup * 100 - 100:+.1f}%)")
                    print()

    def save_results(self, output_path: str):
        output_data = {
            "config": asdict(self.config),
            "raw_results": {
                key: [asdict(m) for m in metrics_list]
                for key, metrics_list in self.results.items()
            },
            "statistics": {
                key: self.compute_statistics(metrics_list)
                for key, metrics_list in self.results.items()
                if metrics_list
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference approaches")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path to Qwen3 model from HuggingFace (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="1.7B",
        help="Model size key for registry (e.g. 0.6B, 1.7B, 4B, 8B, 14B, 32B). Default: 1.7B",
    )
    parser.add_argument(
        "--torchtitan-checkpoint",
        type=str,
        default=None,
        help="Path to TorchTitan checkpoint (required for TorchTitan Native benchmark)",
    )
    parser.add_argument(
        "--torchtitan-config",
        type=str,
        default=None,
        help="Path to TorchTitan config file (TOML). If not specified, will auto-detect.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="torchtitan/experiments/rl/unified/scripts/prompts.txt",
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default="vllm-native,vllm-torchtitan",
        help="Comma-separated list of test cases to run. "
        "Options: vllm-native, vllm-torchtitan, torchtitan-native "
        "(default: vllm-native,vllm-torchtitan)",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        default=True,
        help="Load checkpoint from HuggingFace format (default: True)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    # Profiling options
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler to collect performance traces",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profiler_traces",
        help="Directory to save profiler traces (default: ./profiler_traces)",
    )
    parser.add_argument(
        "--profile-wait",
        type=int,
        default=1,
        help="Number of steps to skip before profiling starts (default: 1)",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Number of warmup steps for the profiler (default: 1)",
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=2,
        help="Number of active profiling steps (default: 2)",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Enable compile(eager) + piecewise cudagraph for vLLM engine. "
        "Default is fully eager (no compile, no cudagraph).",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="eager",
        choices=["eager", "inductor"],
        help="torch.compile backend for vLLM (default: eager). "
        "Use 'inductor' to enable Triton kernel fusion.",
    )
    parser.add_argument(
        "--nccl-algo",
        type=str,
        default=None,
        help="NCCL algorithm to use for all-reduce (e.g. 'Tree'). "
        "Set after model setup to avoid breaking broadcast during init.",
    )
    args = parser.parse_args()

    # Parse test cases
    test_cases = [tc.strip() for tc in args.test_cases.split(",")]
    valid_cases = {"vllm-native", "vllm-torchtitan", "torchtitan-native"}
    for tc in test_cases:
        if tc not in valid_cases:
            parser.error(
                f"Unknown test case '{tc}'. Valid options: {', '.join(sorted(valid_cases))}"
            )

    # Validate arguments
    if "torchtitan-native" in test_cases and args.torchtitan_checkpoint is None:
        parser.error(
            "--torchtitan-checkpoint is required for torchtitan-native test case."
        )

    config = BenchmarkConfig(
        model_path=args.model_path,
        model_size=args.model_size,
        torchtitan_checkpoint_path=args.torchtitan_checkpoint,
        torchtitan_config_path=args.torchtitan_config,
        prompts_file=args.prompts,
        from_hf=args.from_hf,
        tp=args.tp,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        profile=args.profile,
        profile_dir=args.profile_dir,
        profile_wait=args.profile_wait,
        profile_warmup=args.profile_warmup,
        profile_active=args.profile_active,
        use_cuda_graph=args.use_cuda_graph,
        compile_backend=args.compile_backend,
        nccl_algo=args.nccl_algo,
    )

    runner = BenchmarkRunner(config)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    benchmark_map = {
        "vllm-native": (VLLMNativeBenchmark, "vllm_native"),
        "vllm-torchtitan": (VLLMTorchTitanBenchmark, "vllm_torchtitan"),
        "torchtitan-native": (TorchTitanNativeBenchmark, "torchtitan_native"),
    }

    for tc in test_cases:
        cls, key = benchmark_map[tc]
        runner.run_benchmark(cls, key)

    # Print and save results (only on rank 0)
    if local_rank == 0:
        runner.print_summary()
        runner.save_results(args.output)


if __name__ == "__main__":
    main()
