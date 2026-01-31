#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script to compare inference performance across three approaches:
1. vLLM engine with native Qwen3 model (HuggingFace)
2. vLLM engine with TorchTitan Qwen3 model wrapper
3. Direct TorchTitan Qwen3 model inference

Usage:
    # Run only vLLM benchmarks (no checkpoint needed)
    python benchmarking_perf.py --skip-torchtitan-native

    # Run only vLLM TorchTitan with profiling and CUDA graph
    python benchmarking_perf.py --skip-vllm-native --skip-torchtitan-native --profile --use-cuda-graph

    # Run all benchmarks (requires TorchTitan checkpoint for native)
    python benchmarking_perf.py --torchtitan-checkpoint /path/to/checkpoint

    # Run with TP=2
    torchrun --nproc_per_node=2 benchmarking_perf.py --torchtitan-checkpoint /path/to/checkpoint --tp 2

Profiling tips:
    - Only send a few requests through when profiling (use --num-runs 2)
    - Use --warmup-runs 1 for faster profiling setup
    - Traces can get large, consider using --max-tokens 64 for debugging
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.profiler import profile, ProfilerActivity, schedule
from torchtitan.experiments.rl import unified  # noqa: F401


# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# NOTE(jianiw): This function is only used for TorchTitan Native model benchmarking
def apply_tp_only(model, tp_mesh):
    """Apply tensor parallelism to model without sequence parallelism."""

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


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
    torchtitan_checkpoint_path: str = (
        None  # Optional, required only for TorchTitan benchmarks
    )
    torchtitan_config_path: str = None  # Optional config path for TorchTitan
    from_hf: bool = True  # Whether checkpoint is in HuggingFace format
    tp: int = 1  # Tensor parallelism size
    batch_size: int = 1
    max_tokens: int = 512
    num_runs: int = 5
    warmup_runs: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "cuda"
    # CUDA graph / compile options
    use_cuda_graph: bool = False  # Enable CUDA graph via compilation config
    # Profiling options
    profile: bool = False
    profile_dir: str = "./profiler_traces"
    profile_wait: int = 1  # Steps to skip before profiling
    profile_warmup: int = 1  # Warmup steps for profiler
    profile_active: int = 2  # Active profiling steps
    profile_repeat: int = 1  # Number of profiling cycles


class ProfilerManager:
    """Helper class to manage PyTorch profiler for benchmarks."""

    def __init__(self, config: BenchmarkConfig, approach_name: str):
        self.config = config
        self.approach_name = approach_name
        self.profiler: Optional[torch.profiler.profile] = None

    def get_trace_dir(self) -> str:
        """Get the trace directory for this approach."""
        trace_dir = Path(self.config.profile_dir) / self.approach_name.lower().replace(
            " ", "_"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        return str(trace_dir)

    def create_profiler(self) -> Optional[torch.profiler.profile]:
        """Create a PyTorch profiler instance if profiling is enabled."""
        if not self.config.profile:
            return None

        trace_dir = self.get_trace_dir()
        print(f"  Profiler traces will be saved to: {trace_dir}")

        # Create profiler with schedule for warmup/active phases
        # Use Chrome trace export instead of TensorBoard to avoid compatibility issues
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

    def create_simple_profiler(self) -> Optional[torch.profiler.profile]:
        """Create a simple profiler without scheduling (for single-run profiling)."""
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
        """Export Chrome trace file for visualization in Perfetto."""
        if self.profiler is None:
            return

        trace_dir = self.get_trace_dir()
        trace_file = Path(trace_dir) / f"trace_run_{run_idx}.json"
        self.profiler.export_chrome_trace(str(trace_file))
        print(f"  Exported Chrome trace to: {trace_file}")

    def print_summary(self):
        """Print profiler summary statistics."""
        if self.profiler is None:
            return

        print(f"\n  === Profiler Summary for {self.approach_name} ===")
        print(
            self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        )


class VLLMNativeBenchmark:
    """Benchmark vLLM with native Qwen3 1.7B model from HuggingFace."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None
        self.profiling_enabled = False
        self.profile_dir = None

    def setup(self):
        """Initialize vLLM engine with native Qwen3 model."""
        try:
            from vllm import LLM, SamplingParams

            print("Loading vLLM with native Qwen3 model from HuggingFace...")
            print(f"Model: {self.config.model_path}")
            print(f"Tensor Parallel Size: {self.config.tp}")
            print(
                f"CUDA Graph: {'Enabled' if self.config.use_cuda_graph else 'Disabled (eager mode)'}"
            )

            # Set up profiling via environment variable (compatible with more vLLM versions)
            if self.config.profile:
                self.profile_dir = Path(self.config.profile_dir) / "vllm_native"
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                os.environ["VLLM_TORCH_PROFILER_DIR"] = str(self.profile_dir)
                self.profiling_enabled = True
                print(
                    f"Profiling ENABLED - traces will be saved to: {self.profile_dir}"
                )

            # Configure compilation based on use_cuda_graph setting
            if self.config.use_cuda_graph:
                compilation_config = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
                enforce_eager = False
            else:
                compilation_config = None
                enforce_eager = True

            # Use external_launcher when launched with torchrun (TP > 1)
            # vLLM will pick up WORLD_SIZE, RANK, etc. from environment
            distributed_backend = "external_launcher" if self.config.tp > 1 else None

            self.engine = LLM(
                model=self.config.model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
                tensor_parallel_size=self.config.tp,
                distributed_executor_backend=distributed_backend,
                compilation_config=compilation_config,
                enforce_eager=enforce_eager,
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            print("✓ vLLM native Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM native model: {e}")
            raise

    def run_inference(
        self, prompts: List[str], use_profiler: bool = False
    ) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Try to use start_profile/stop_profile if available (newer vLLM versions)
        # With VLLM_TORCH_PROFILER_DIR set, profiling may happen automatically
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
                    print(
                        "  Profiling will rely on VLLM_TORCH_PROFILER_DIR environment variable"
                    )
            else:
                print(
                    "  Note: LLM.start_profile() not available, relying on VLLM_TORCH_PROFILER_DIR"
                )

        # Measure prefill (first token) time separately
        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        # Stop profiler if it was started
        if profile_started:
            print("  Stopping profiler...")
            try:
                self.engine.stop_profile()
                print(
                    f"  Profiler stopped. Traces should be written to: {self.profile_dir}"
                )
            except Exception as e:
                print(f"  Warning: stop_profile() failed: {e}")

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        # Memory metrics
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        # vLLM doesn't expose prefill/decode timing directly
        first_token_latency = 0.0
        prefill_time = 0.0
        decode_time = 0.0

        return BenchmarkMetrics(
            approach="vLLM Native",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=prefill_time,
            decode_time=decode_time,
            throughput_tokens_per_sec=total_tokens / total_time,
            latency_per_token_ms=(total_time / total_tokens) * 1000,
            first_token_latency_ms=first_token_latency,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        """Cleanup resources."""
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class VLLMTorchTitanBenchmark:
    """Benchmark vLLM with TorchTitan Qwen3 model via infer.py."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None
        self.profiling_enabled = False
        self.profile_dir = None

    def setup(self):
        """Initialize vLLM engine with TorchTitan Qwen3 model."""
        try:
            # Import unified module to register TorchTitan models with vLLM
            from vllm import LLM, SamplingParams

            print("Loading vLLM with TorchTitan Qwen3 model...")
            print(f"Model: {self.config.model_path}")
            print(f"Tensor Parallel Size: {self.config.tp}")
            print(
                f"CUDA Graph: {'Enabled' if self.config.use_cuda_graph else 'Disabled (eager mode)'}"
            )

            # Set up profiling via environment variable (compatible with more vLLM versions)
            if self.config.profile:
                self.profile_dir = Path(self.config.profile_dir) / "vllm_torchtitan"
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                os.environ["VLLM_TORCH_PROFILER_DIR"] = str(self.profile_dir)
                self.profiling_enabled = True
                print(
                    f"Profiling ENABLED - traces will be saved to: {self.profile_dir}"
                )

            # Configure compilation based on use_cuda_graph setting
            if self.config.use_cuda_graph:
                compilation_config = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
                enforce_eager = False
            else:
                compilation_config = None
                enforce_eager = True

            # Use external_launcher when launched with torchrun (TP > 1)
            # vLLM will pick up WORLD_SIZE, RANK, etc. from environment
            distributed_backend = "external_launcher" if self.config.tp > 1 else None

            self.engine = LLM(
                model=self.config.model_path,
                hf_overrides={
                    # Override architectures to use our registered TorchTitan model class
                    "architectures": ["Qwen3TorchTitanForCausalLM"],
                },
                dtype="bfloat16",
                trust_remote_code=True,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.5,
                tensor_parallel_size=self.config.tp,
                distributed_executor_backend=distributed_backend,
                compilation_config=compilation_config,
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            print("✓ vLLM TorchTitan Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM TorchTitan model: {e}")
            import traceback

            traceback.print_exc()
            raise

    def run_inference(
        self, prompts: List[str], use_profiler: bool = False
    ) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Try to use start_profile/stop_profile if available (newer vLLM versions)
        # With VLLM_TORCH_PROFILER_DIR set, profiling may happen automatically
        profile_started = False
        if use_profiler and self.profiling_enabled:
            if hasattr(self.engine, "start_profile"):
                try:
                    self.engine.start_profile()
                    profile_started = True
                    print("Profiler started successfully")
                except Exception as e:
                    print(f"  Warning: start_profile() failed: {e}")
                    print(
                        "  Profiling will rely on VLLM_TORCH_PROFILER_DIR environment variable"
                    )

        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        # Stop profiler if it was started
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

        # vLLM doesn't expose prefill/decode timing directly
        first_token_latency = 0.0
        prefill_time = 0.0
        decode_time = 0.0

        return BenchmarkMetrics(
            approach="vLLM TorchTitan",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=prefill_time,
            decode_time=decode_time,
            throughput_tokens_per_sec=total_tokens / total_time
            if total_time > 0
            else 0,
            latency_per_token_ms=(total_time / total_tokens) * 1000
            if total_tokens > 0
            else 0,
            first_token_latency_ms=first_token_latency,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        """Cleanup resources."""
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class TorchTitanNativeBenchmark:
    """Benchmark direct TorchTitan Qwen3 model inference using test_generate.py approach."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.config_path = None

    def setup(self):
        """Initialize TorchTitan Qwen3 model using test_generate.py approach."""
        try:
            print("Loading TorchTitan Qwen3 model directly (test_generate.py style)...")

            import os
            import sys
            from pathlib import Path

            import torch.distributed as dist
            import torch.distributed.checkpoint as dcp
            from torchtitan.config import ConfigManager
            from torchtitan.distributed import ParallelDims
            from torchtitan.protocols.train_spec import get_train_spec

            # Get world size and rank from environment (set by torchrun)
            env_world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Use config.tp as the intended TP size
            # Validate that it matches torchrun's world_size when running distributed
            if self.config.tp > 1:
                if env_world_size == 1:
                    raise RuntimeError(
                        f"TP={self.config.tp} requested but running in single process mode. "
                        f"Use: torchrun --nproc_per_node={self.config.tp} scripts/benchmark_inference.py ..."
                    )
                if env_world_size != self.config.tp:
                    raise RuntimeError(
                        f"TP={self.config.tp} but torchrun launched {env_world_size} processes. "
                        f"Use: torchrun --nproc_per_node={self.config.tp} scripts/benchmark_inference.py ..."
                    )

            # world_size should equal TP size. Assume only TP is applied
            world_size = self.config.tp

            print(
                f"TP={self.config.tp}, World Size: {world_size}, Local Rank: {local_rank}"
            )

            # Initialize distributed
            if not dist.is_initialized():
                if world_size > 1:
                    # Multi-process mode (TP > 1) - should be launched with torchrun
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "29500")
                    dist.init_process_group(
                        backend="nccl" if torch.cuda.is_available() else "gloo",
                        init_method="env://",
                    )
                else:
                    # Single-process mode (TP == 1)
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "29500")
                    os.environ.setdefault("RANK", "0")
                    os.environ.setdefault("WORLD_SIZE", "1")
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://",
                    )

            # Add generate module to path
            generate_path = Path(__file__).parent.parent / "scripts" / "generate"
            sys.path.insert(0, str(generate_path))
            from _generation import generate as generate_fn

            self.generate_fn = generate_fn

            # Find config file for the model
            if self.config.torchtitan_config_path:
                self.config_path = self.config.torchtitan_config_path
            else:
                # Try to find a config file in the model directory
                # Default to ./train_configs/qwen3_1.7b.toml
                config_search_paths = [
                    Path("./train_configs/qwen3_1.7b.toml"),
                    Path("./torchtitan/models/qwen3/train_configs/qwen3_1.7b.toml"),
                    Path("/data/users/jianiw/torchtitan/train_configs/qwen3_1.7b.toml"),
                    Path(
                        "/data/users/jianiw/torchtitan/torchtitan/models/qwen3/train_configs/debug_model.toml"
                    ),
                    Path("./torchtitan/models/qwen3/train_configs/debug_model.toml"),
                ]

                self.config_path = None
                for path in config_search_paths:
                    if path.exists():
                        self.config_path = str(path)
                        break

                if self.config_path is None:
                    raise RuntimeError(
                        "Could not find model config file. Please specify --torchtitan-config\n"
                        "Searched paths:\n"
                        + "\n".join(f"  - {p}" for p in config_search_paths)
                    )

            print(f"Using config: {self.config_path}")

            # Load configuration
            config_manager = ConfigManager()
            self.tt_config = config_manager.parse_args(
                [f"--job.config_file={self.config_path}"]
            )

            # Get train spec
            train_spec = get_train_spec(self.tt_config.model.name)

            # Build tokenizer
            self.tokenizer = train_spec.build_tokenizer_fn(self.tt_config)

            # Get model args
            model_args = train_spec.model_args[self.tt_config.model.flavor]
            model_args.update_from_config(self.tt_config)

            # Setup parallel dims
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

            # Initialize model on appropriate device
            device = torch.device(f"{self.config.device}:{local_rank}")
            init_device = "meta" if world_size > 1 else device

            print(f"Initializing model on device: {init_device}")
            with torch.device(init_device):
                self.model = train_spec.model_cls(model_args)

            # Apply TP if world_size > 1
            if world_size > 1:
                print(f"Applying tensor parallelism with TP={world_size}")
                apply_tp_only(self.model, parallel_dims.get_mesh("tp"))

            # Materialize model
            self.model.to_empty(device=device)
            with torch.no_grad():
                self.model.init_weights()
            self.model.eval()

            # Load checkpoint using DCP
            print(
                f"Loading checkpoint from {self.config.torchtitan_checkpoint_path}..."
            )

            # Check if checkpoint is a directory (DCP format) or a file (torch.save format)
            checkpoint_path = Path(self.config.torchtitan_checkpoint_path)

            if self.config.from_hf:
                # Load HuggingFace format checkpoint using state dict adapter
                print("Loading from HuggingFace format using state dict adapter...")
                sd_adapter = train_spec.state_dict_adapter(model_args, None)
                assert (
                    sd_adapter is not None
                ), "Trying to load HF checkpoint, but sd_adapter is not available for this model."

                # Get state dict in torchtitan format using DTensor-aware function
                # This handles DTensor models properly (like train.py does)
                state_dict = get_model_state_dict(self.model)
                # Convert to HF format so that HF weights can be loaded into it
                hf_state_dict = sd_adapter.to_hf(state_dict)
                # Load HF weights
                dcp.load(
                    hf_state_dict,
                    storage_reader=HuggingFaceStorageReader(path=str(checkpoint_path)),
                )
                # Convert back from HF to torchtitan format
                state_dict = sd_adapter.from_hf(hf_state_dict)
                # Use set_model_state_dict for DTensor models (with strict=False for flexibility)
                set_model_state_dict(
                    self.model,
                    model_state_dict=state_dict,
                    options=StateDictOptions(strict=False),
                )
            else:
                # Load DCP format checkpoint
                state_dict = get_model_state_dict(self.model)
                dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
                set_model_state_dict(
                    self.model,
                    model_state_dict=state_dict,
                    options=StateDictOptions(strict=False),
                )

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

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device)
        generated_tokens = input_ids.clone()

        # Prefill: first token generation
        prefill_start = time.perf_counter()
        logits = self.model(generated_tokens)
        probs = torch.nn.functional.softmax(
            logits[:, -1, :] / max(self.config.temperature, 1e-5), dim=-1
        )

        if self.config.temperature == 0.0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Decode: remaining tokens
        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            logits = self.model(generated_tokens)
            probs = torch.nn.functional.softmax(
                logits[:, -1, :] / max(self.config.temperature, 1e-5), dim=-1
            )

            if self.config.temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start

        return generated_tokens, prefill_time, decode_time

    def run_inference(
        self, prompts: List[str], profiler: Optional[torch.profiler.profile] = None
    ) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        total_tokens = 0
        total_prefill_time = 0
        total_decode_time = 0
        first_token_latency = None

        start_time = time.perf_counter()

        # Define the inference loop
        def run_generation():
            nonlocal total_tokens, total_prefill_time, total_decode_time, first_token_latency

            for prompt in prompts:
                # Tokenize (using TorchTitan tokenizer API)
                if hasattr(self.tokenizer, "encode"):
                    # For BaseTokenizer with encode method that takes add_bos/add_eos
                    try:
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt, add_bos=True, add_eos=False),
                            dtype=torch.long,
                        )
                    except TypeError:
                        # Fallback for HF tokenizers
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt), dtype=torch.long
                        )
                else:
                    # Fallback
                    input_ids = torch.tensor(
                        self.tokenizer.encode(prompt), dtype=torch.long
                    )

                # Generate
                output_tokens, prefill_time, decode_time = self.generate_with_timing(
                    input_ids, self.config.max_tokens
                )

                if first_token_latency is None:
                    first_token_latency = prefill_time * 1000  # Convert to ms

                num_generated = output_tokens.size(1) - input_ids.size(0)
                total_tokens += num_generated
                total_prefill_time += prefill_time
                total_decode_time += decode_time

        # Run with or without profiler
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
        """Cleanup resources."""
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
        self.results: Dict[str, List[BenchmarkMetrics]] = {
            "vllm_native": [],
            "vllm_torchtitan": [],
            "torchtitan_native": [],
        }

    def load_prompts(self) -> List[str]:
        """Load prompts from file."""
        prompts_path = Path(self.config.prompts_file)
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default prompts
            prompts = [
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the benefits of using Python for data science?",
                "Describe the process of photosynthesis.",
                "How does a neural network work?",
            ]

        return prompts[: self.config.batch_size]

    def run_benchmark(self, benchmark_cls, key: str):
        """Run benchmark for a specific approach."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {key}")
        if self.config.profile:
            print(
                f"Profiling ENABLED - traces will be saved to: {self.config.profile_dir}"
            )
        print(f"{'=' * 60}")

        try:
            benchmark = benchmark_cls(self.config)
            benchmark.setup()

            prompts = self.load_prompts()

            # Determine if this is a vLLM benchmark (uses built-in profiler) or TorchTitan (uses external profiler)
            is_vllm_benchmark = key.startswith("vllm_")

            # Warmup runs (no profiling during warmup)
            print(f"Running {self.config.warmup_runs} warmup iterations...")
            for i in range(self.config.warmup_runs):
                benchmark.run_inference(prompts)
                print(f"  Warmup {i + 1}/{self.config.warmup_runs} completed")

            # Actual benchmark runs
            print(f"Running {self.config.num_runs} benchmark iterations...")

            if self.config.profile:
                if is_vllm_benchmark:
                    # For vLLM benchmarks, use vLLM's built-in profiler API
                    # Only run 1 iteration when profiling to avoid large trace files
                    profile_runs = 1
                    print(
                        f"  (Profiling mode: running {profile_runs} iteration only, no warmup iterations)"
                    )
                    for i in range(profile_runs):
                        metrics = benchmark.run_inference(prompts, use_profiler=True)
                        self.results[key].append(metrics)
                        print(
                            f"  Run {i + 1}/{profile_runs} (profiled): "
                            f"{metrics.throughput_tokens_per_sec:.2f} tokens/s, "
                            f"latency: {metrics.latency_per_token_ms:.2f} ms/token"
                        )
                else:
                    # For TorchTitan Native, use external PyTorch profiler
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
        self, metrics_list: List[BenchmarkMetrics]
    ) -> Dict[str, Any]:
        """Compute statistics from multiple runs."""
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
        """Print benchmark summary."""
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
        """Save results to JSON file."""
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

    def run_all(self):
        """Run all benchmarks."""
        print("Starting benchmark suite...")
        print("Configuration:")
        print(f"  Model (HuggingFace): {self.config.model_path}")
        if self.config.torchtitan_checkpoint_path:
            print(f"  TorchTitan checkpoint: {self.config.torchtitan_checkpoint_path}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Max tokens: {self.config.max_tokens}")
        print(f"  Warmup runs: {self.config.warmup_runs}")
        print(f"  Benchmark runs: {self.config.num_runs}")
        if self.config.profile:
            print("  Profiling: ENABLED")
            print(f"  Profile dir: {self.config.profile_dir}")
        else:
            print("  Profiling: disabled")

        # Run benchmarks
        self.run_benchmark(VLLMNativeBenchmark, "vllm_native")
        self.run_benchmark(VLLMTorchTitanBenchmark, "vllm_torchtitan")
        self.run_benchmark(TorchTitanNativeBenchmark, "torchtitan_native")

        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference approaches")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path to Qwen3 model from HuggingFace (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--torchtitan-checkpoint",
        type=str,
        default=None,
        help="Path to TorchTitan checkpoint (required only for TorchTitan Native benchmark)",
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
        default="scripts/prompts.txt",
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
        "--skip-vllm-native",
        action="store_true",
        help="Skip vLLM native benchmark",
    )
    parser.add_argument(
        "--skip-vllm-torchtitan",
        action="store_true",
        help="Skip vLLM TorchTitan benchmark",
    )
    parser.add_argument(
        "--skip-torchtitan-native",
        action="store_true",
        help="Skip TorchTitan native benchmark",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        default=True,
        help="Load checkpoint from HuggingFace format using state dict adapter (default: True)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1). When TP > 1, use torchrun --nproc_per_node=TP",
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
        help="Enable CUDA graph via compilation config. If not set, uses eager mode (default: eager)",
    )

    args = parser.parse_args()

    # Validate that torchtitan-checkpoint is provided when needed (only for TorchTitan Native)
    if not args.skip_torchtitan_native and args.torchtitan_checkpoint is None:
        parser.error(
            "--torchtitan-checkpoint is required for TorchTitan Native benchmark. "
            "Use --skip-torchtitan-native if you don't want to run it."
        )

    config = BenchmarkConfig(
        model_path=args.model_path,
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
    )

    runner = BenchmarkRunner(config)

    # Check if running with torchrun (multi-process)
    import os

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Run selected benchmarks
    # When TP > 1: use torchrun, all ranks participate (external_launcher)
    # When TP = 1: use python directly (no torchrun), single process
    if not args.skip_vllm_native:
        runner.run_benchmark(VLLMNativeBenchmark, "vllm_native")

    if not args.skip_vllm_torchtitan:
        runner.run_benchmark(VLLMTorchTitanBenchmark, "vllm_torchtitan")

    if not args.skip_torchtitan_native:
        runner.run_benchmark(TorchTitanNativeBenchmark, "torchtitan_native")

    # Print and save results (only on rank 0)
    if local_rank == 0:
        runner.print_summary()
        runner.save_results(args.output)


if __name__ == "__main__":
    main()
