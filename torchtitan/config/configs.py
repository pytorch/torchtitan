# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared configuration dataclasses for torchtitan.

Some configs live near their owner instead of here:
  - Profiler.Config                 (in observability/profiler.py)
  - OptimizersContainer.Config      (in components/optimizer/optimizer.py)
  - LRSchedulersContainer.Config    (in components/optimizer/lr_scheduler.py)
  - MetricsProcessor.Config         (in observability/metrics.py)
  - CheckpointManager.Config        (in components/checkpointer/dcp.py)

Configs without a clear single owner (or with circular-import constraints)
live here.

Most knobs belong to a component or to the model, not here. But some options
have no suitable home, e.g. the training token-budget settings, and those can
be placed here. Discuss with the maintainers first if you intend to add one.

The command-line surface is frozen either way, so annotate a new field with
``tyro.conf.Suppress``, as ``Trainer.Config.model`` does. See
``torchtitan/config/README.md``.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    num_tokens_per_microbatch_per_dp_rank: int = 16384
    """
    Number of input-token slots processed per data-parallel rank in one model
    forward, before context or tensor parallel sharding.
    """

    num_tokens_per_train_step: int = -1
    """
    Global number of input-token slots across data-parallel ranks, pipeline
    microbatches, and gradient accumulation steps. Defaults to
    `training.num_tokens_per_microbatch_per_dp_rank * num_pp_microbatches *
    data-parallel degree`.
    """

    max_context_length: int = 2048
    """Maximum logical context length used for training."""

    def __post_init__(self) -> None:
        if self.num_tokens_per_microbatch_per_dp_rank <= 0:
            raise ValueError(
                "num_tokens_per_microbatch_per_dp_rank must be greater than 0."
            )
        if self.num_tokens_per_train_step != -1 and self.num_tokens_per_train_step <= 0:
            raise ValueError("num_tokens_per_train_step must be -1 or greater than 0.")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0.")
        if self.max_norm < 0:
            raise ValueError("max_norm must be greater than or equal to 0.")

    max_norm: float | int = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """How many train steps to run"""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    disable_cuda_graphs: bool = False
    """
    Disable CUDA graph capture and replay for the forward+backward step. CUDA
    graphs require fixed-shape inputs and no CPU<->GPU synchronization during
    the captured region. Expert parallelism is supported only with HybridEP
    when ``non_blocking_capacity_factor`` is set. Other EP backends synchronize
    with the host during dispatch. Pipeline parallelism
    is supported with single-stage schedules such as GPipe and 1F1B. CUDA graphs
    are independent of ``torch.compile(mode="reduce-overhead")``, which performs
    its own CUDA graph capture.
    """

    dtype: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype for training. In contrast to mixed precision training, setting training_dtype=bfloat16 will
    put all parameters, gradients, and optimizer states in bfloat16, without an extra copy of fp32 weights.
    In the case of full bf16 training, RoPE calculations and logits will still be in fp32.
    """

    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via fully_shard or torch.autocast.
    This feature takes effect via fully_shard when data_parallel_shard_degree > 1 or
    context_parallel_degree > 1; it takes effect via torch.autocast when data_replicate_degree >= 1
    and no other parallelism is enabled, i.e. under DDP or single-device training.
    """

    mixed_precision_reduce: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    gc_debug: bool = False
    """
    Enable GC debugging mode. This will perform gc.collect() at every step to
    detect if there is a reference cycle that includes a CUDA Tensor.
    Note that you may want to lower the training steps to avoid generating too
    many temporary files.
    """


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    enable_async_tensor_parallel: bool = False
    """Whether to pipeline tensor-parallel collectives with matrix multiplications."""

    components: list[str] = field(default_factory=lambda: ["model", "loss"])
    """Which components to compile"""

    backend: str = "inductor"

    def __post_init__(self) -> None:
        allowed = frozenset({"model", "loss"})
        unknown = [c for c in self.components if c not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown compile.components entries {unknown}; "
                f"allowed values are {sorted(allowed)}"
            )
        if self.enable_async_tensor_parallel and "model" not in self.components:
            raise ValueError("Async TP requires 'model' in --compile.components.")


@dataclass(kw_only=True, slots=True)
class CommConfig:
    init_timeout_seconds: int = 300
    """Timeout for communication operations, during initialization and first train step."""

    train_timeout_seconds: int = 100
    """
    Timeout for communication operations after the first train step --
    usually a tighter bound than during initialization.
    """

    trace_buf_size: int = 20000
    """Flight recorder ring buffer size, >0 means recording by default, 0 means disabled"""

    save_traces_folder: str = "comm_traces"
    """Flight recorder trace files location"""

    save_traces_file_prefix: str = "rank_"
    """Flight recorder trace files prefix"""

    mode: Literal["default", "fake_backend"] = "default"
    """
    Communication mode for distributed training.

    Options:
    - "default": Normal distributed training with real communication
    - "fake_backend": Fake comm backend for dry run mode only (configuration validation without GPU)
    """


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    spmd_typechecking: bool = False
    """Enable global SPMD type checking."""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

    moe_force_load_balance: bool = False
    """If True, we force each experts to get the same amount of tokens via round-robin. This option is for debugging usage only."""

    detect_anomaly: bool = False
    """Enable torch.autograd anomaly detection to help track down NaN/Inf gradients.
    Note: incurs significant overhead; for debugging only."""

    batch_invariant: bool = False
    """Enable batch-invariant mode to use batch-invariant ops in model
    forward and deterministic NCCL collective reduction order"""

    print_config: bool = False
    """Print the job configs to terminal"""

    save_config_file: str | None = None
    """Path to save job config into"""

    enable_structured_logging: bool = True
    """Whether to enable the structured per-rank trace logger (see
    ``torchtitan.observability.structured_logger``). When False, all
    ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` calls
    are no-ops. Disable to fully eliminate trace overhead."""
