# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared configuration dataclasses for torchtitan.

Some configs live near their owner instead of here:
  - ProfilingConfig                 (in tools/profiling.py)
  - OptimizersContainer.Config      (in components/optimizer.py)
  - LRSchedulersContainer.Config    (in components/lr_scheduler.py)
  - MetricsProcessor.Config         (in components/metrics.py)
  - ModelConvertersContainer.Config  (in protocols/model_converter.py)
  - CheckpointManager.Config        (in components/checkpoint.py)

Configs without a clear single owner (or with circular-import constraints)
live here.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    local_batch_size: int = 8
    """Local batch size (i.e., per-device batch size)"""

    global_batch_size: int = -1
    """
    Global batch size (defaults to `training.local_batch_size * data-parallel degree`)
    """

    seq_len: int = 2048
    """Sequence length"""

    max_norm: float | int = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """How many train steps to run"""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
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

    mixed_precision_reduce: Literal["float32"] = "float32"
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
class ParallelismConfig:
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    disable_loss_parallel: bool = False
    """Whether to apply loss parallel when sequence parallel is enabled"""

    enable_async_tensor_parallel: bool = False
    """Whether to apply async tensor parallel (currently only effective when compile is enabled)"""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model chunk.
    Each inner list represents one model chunk and contains the module names that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk compared to split points.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism. This is because
    the first stage has the extra overhead of the embedding layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism. This is because
    the last stage has the extra overhead of the output layer, which is not present in the other stages.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the module_fqns_per_model_part will be
    calculated from the number of layers and pipeline_parallel_degree. If not specified, the
    layers per stage will be inferred from the model, schedule, and pipeline_parallel_degree.
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_schedule_csv: str | None = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use.
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    pipeline_parallel_microbatch_size: int = 1
    """
    The size of each pipeline parallel microbatch (default 1).
    This value is used to compute the total number of microbatches by dividing local_batch_size with
    pipeline_parallel_microbatch_size.
    The global training batch size must be evenly divisible by pipeline_parallel_microbatch_size.
    """

    pipeline_parallel_expert_parallel_overlap: bool = True
    """Whether to turn on the optimization to overlap expert parallel and pipeline parallel
    communication. This is only effective when the pipeline parallel schedule is DualPipeV and
    pipeline_parallel_degree > 1 and expert_parallel_degree > 1.

    TODO: Does not support activation_checkpoint, set mode="none"
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_load_balancer: str | None = "headtail"
    """
    Load balancer type for context parallelism. Options:
    - "headtail": Use HeadTailLoadBalancer for SDPA
    - "ptrr": Use PTRRLoadBalancer for FlexAttention
    - None: Disable load balancing
    """

    def __post_init__(self):
        if self.context_parallel_load_balancer == "":
            raise ValueError(
                "context_parallel_load_balancer cannot be an empty string. "
                "Use None to disable load balancing."
            )

    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    """
    The collective to use in context parallel SDPA for kv shards exchange.
    - 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
    - 'alltoall' means to all-to-all shuffle the kv shards.
    The default value is 'allgather'.
    """

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Currently, etp is either 1 or is the same as tp.

    Note that this is still an experimental feature. Some constraints will be
    relaxed soon when we have more flexible DeviceMesh support.
    """

    expert_tensor_parallel_degree: int = 1
    """
    Expert tensor parallelism degree. 1 means disabled. No effect for non-MoE models, or when ep = 1.
    With this option, the tensor parallel degree on routed experts can be different from that on other params.
    Currently, we only support either
    - [partial dp -> ep] etp = tp
    - [partial dp + all tp -> ep] etp = 1
    Note that this is still an experimental feature.
    """

    expert_parallel_comm_backend: Literal["standard", "deepep"] = "standard"
    """
    Expert-parallel communication backend. No effect for non-MoE models or when ep = 1.

    - "standard": Uses PyTorch all-to-all collectives (default)
    - "deepep": Uses DeepEP custom kernels for more efficient communication

    DeepEP requires installation:
    https://github.com/deepseek-ai/DeepEP.
    """


@dataclass(kw_only=True, slots=True)
class ActivationCheckpointConfig:
    mode: Literal["selective", "full", "memory_budget", "none"] = "selective"
    """Type of activation checkpointing to use"""

    selective_ac_option: str = "2"
    """
    Selective activation checkpointing options ['int', 'op'].
    'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
    """

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    """
    When per-op selective ac is used, this list of fully qualified names is used
    to determine which mm shapes to force recompute, rather than being considered
    by rest of the sac policy, e.g save every other mm. Only nn.Linear modules are
    supported today.

    Note: this config applies to mms not limited to those matching the specified
    fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
    ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
    """

    early_stop: bool = False
    """
    Whether to stop recomputing early when all activations have already been
    rematerialized.
    """

    memory_budget: float = 0.5
    """
    When mode is set to "memory_budget", this value determines how much
    partitioner in the compiler should trade off compute for memory.
    0.0 corresponds to the activation memory from applying
    activation checkpointing to the full compiled region, and 1.0 corresponds to
    the activation memory from the default runtime-optimized strategy. Read here:
    https://pytorch.org/blog/activation-checkpointing-techniques/
    """

    visualize_memory_budget_pareto: bool = False
    """
    This dumps out a SVG visualization of the expected runtime vs. activation
    memory tradeoffs for all memory budget values from 0 to 1 in increments of
    0.05 in {--dump_folder}/memory_budget_pareto folder. See an example here:
    https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
    """

    preserve_rng_state: bool = True
    """
    If deterministic output compared to non-checkpointed passes is required, set
    to true. Results in stashing and restoring the RNG state during each checkpoint,
    may be slower. See https://docs.pytorch.org/docs/stable/checkpoint.html
    for details.
    """

    determinism_check: str = "default"
    """
    A string specifying the determinism function. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """

    debug: bool = False
    """
    Capture ac debug information. Will be slower. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    enable: bool = False
    """Whether to apply torch.compile"""

    components: list[str] = field(default_factory=lambda: ["model", "loss"])
    """Which components to compile"""

    backend: str = "inductor"


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

    mode: Literal["default", "fake_backend", "local_tensor"] = "default"
    """
    Communication mode for distributed training.

    Options:
    - "default": Normal distributed training with real communication
    - "fake_backend": Fake comm backend for dry run mode only (configuration validation without GPU)
    - "local_tensor": Local tensor mode for debugging purposes. There will be only one process
      regardless of the number of GPUs. LocalTensor will simulate the computation by running one
      rank after another. While the performance will be slow, the numerics should be the same.
      This enables us to verify numerics with fewer GPUs. For example, we can directly run 5D
      parallelisms within a single node to reduce the combinations we need to use in integration tests.

    NOTE: local_tensor is an experimental feature and automatically uses fake_backend internally.
    """


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

    moe_force_load_balance: bool = False
    """If True, we force each experts to get the same amount of tokens via round-robin. This option is for debugging usage only."""

    print_config: bool = False
    """Print the job configs to terminal"""

    save_config_file: str | None = None
    """Path to save job config into"""
