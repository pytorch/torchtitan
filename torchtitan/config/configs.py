# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared configuration dataclasses for torchtitan.

Some configs live near their owner instead of here:
  - Profiler.Config                 (in observability/profiler.py)
  - Optim.Config                    (in components/optim/optim.py)
  - OptimizersContainer.Config      (in components/optim/optimizer.py)
  - LRSchedulersContainer.Config    (in components/optim/lr_scheduler.py)
  - MetricsProcessor.Config         (in observability/metrics.py)
  - CheckpointManager.Config        (in components/checkpointer/dcp.py)

Configs without a clear single owner (or with circular-import constraints)
live here.

Most knobs belong to a component or to the model, not here. But some options
have no suitable home, e.g. the training token-budget settings, and those can
be placed here. Discuss with the maintainers first if you intend to add one.

Configuration is provided by Python recipe functions. See
``torchtitan/config/README.md``.
"""

from dataclasses import dataclass, field
from typing import Literal


CommBackend = Literal["default", "fake", "real_pp_fake_spmd"]


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

    steps: int = 10000
    """How many train steps to run"""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    disable_cuda_graphs: bool = False
    """
    Disable CUDA graph capture and replay for the forward and backward pass. CUDA
    graphs require fixed-shape inputs and no CPU<->GPU synchronization during
    the captured region. Expert parallelism is supported only with HybridEP
    when ``non_blocking_capacity_factor`` is set. Other EP backends synchronize
    with the host during dispatch. For pipeline parallelism, TorchTitan
    configures the schedule-derived directed-edge process groups required by
    looped and split-backward schedule replay. CUDA graphs are independent of
    ``torch.compile(mode="reduce-overhead")``, which performs its own CUDA graph
    capture.
    """

    cuda_graph_per_accumulation_group: bool = False
    """Capture and replay one uniform gradient-accumulation group.

    Every group must have the same input structure, tensor metadata, and set of
    parameters receiving gradients. Each replay performs its own FSDP gradient
    reduction and reshard. With HSDP, each replay also performs the replica
    all-reduce. This costs one all-reduce per group and may not be bitwise
    identical to eager accumulation, which all-reduces once.
    This mode supports RL workloads where the number of accumulation groups can
    change.
    """
    # TODO: Remove this option when multiple CUDA graphs support variable group
    # counts without duplicating the gradient accumulation logic.

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

    backend: CommBackend = "default"
    """Communication topology used for training or distributed debugging.

    Options:
    - ``"default"`` uses real process groups for every configured mesh axis.
    - ``"fake"`` represents PP coordinate ``FAKE_PP_RANK`` and SPMD coordinate
      zero in a completely fake logical mesh. It validates configuration,
      shapes, ownership, and PyTorch-managed memory without real transport.
    - ``"real_pp_fake_spmd"`` runs one physical process per PP rank and
      uses a real NCCL PP group while DP, TP, CP, and EP remain fake. It
      exercises pipeline transport, buffers, and CUDA graphs without allocating
      the complete logical world.

    ``NGPU`` is the complete logical world size. ``FAKE_PP_RANK`` applies only
    to ``"fake"``; ``"real_pp_fake_spmd"`` uses physical ``RANK`` as its PP
    coordinate. See ``docs/debugging.md`` for launch examples and limitations.
    """


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    distinct_seed_mesh_axes: list[str] = field(default_factory=lambda: ["pp"])
    """Mesh axes whose ranks each get a distinct RNG seed."""

    spmd_typechecking: bool = False
    """Enable global SPMD type checking."""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

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

    def __post_init__(self):
        # dp_replicate ranks hold replicated params, so distinct seeds there
        # would initialize each replica differently.
        if "dp_replicate" in self.distinct_seed_mesh_axes:
            raise ValueError(
                "debug.distinct_seed_mesh_axes must not contain 'dp_replicate': "
                "its ranks hold replicated parameters and would be initialized "
                "differently."
            )
