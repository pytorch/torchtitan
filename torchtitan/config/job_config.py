# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Job:
    config_file: str | None = None
    """Job config file"""

    dump_folder: str = "./torchtitan/outputs"
    """Folder to dump job outputs"""

    description: str = "default job"
    """Description of the job"""

    use_for_integration_test: bool = False
    """Add this config to the integration test suite"""

    print_args: bool = False
    """Print the args to terminal"""


@dataclass
class Profiling:
    enable_profiling: bool = False
    """Whether to enable pytorch profile"""

    save_traces_folder: str = "profile_traces"
    """Trace files location"""

    profile_freq: int = 10
    """How often to collect profile traces, in interations"""

    enable_memory_snapshot: bool = False
    """Whether to dump memory snapshot"""

    save_memory_snapshot_folder: str = "memory_snapshot"
    """Memory snapshot files location"""


@dataclass
class Metrics:
    log_freq: int = 10
    """How often to log metrics to TensorBoard, in iterations"""

    enable_tensorboard: bool = False
    """Whether to log metrics to TensorBoard"""

    disable_color_printing: bool = False
    """Whether to disable color printing in logs"""

    save_tb_folder: str = "tb"
    """Folder to dump TensorBoard states"""

    save_for_all_ranks: bool = False
    """
    Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
    When this option is False and pipeline_parallel_degree is > 1, the metrics
    component uses the 0th rank of the last stage pipeline group, which is the
    only stage that computes loss metrics.
    """

    enable_wandb: bool = False
    """Whether to log metrics to Weights & Biases"""


@dataclass
class Model:
    name: str = "llama3"
    """Which model to train"""

    flavor: str = "debugmodel"
    """Which model config to train"""

    tokenizer_path: str = "./tests/assets/tokenizer"
    """Tokenizer path"""

    converters: list[str] = field(default_factory=list)
    """
    Comma separated list of converters to apply to the model.
    For instance, the `float8` converter swaps `torch.nn.Linear`
    with `Float8Linear`. This feature requires you to install 'torchao'
    which can be found here: https://github.com/pytorch/ao
    """

    print_after_conversion: bool = False
    """
    If true, model definition will be printed to stdout after all model
    converters have been applied.
    """


@dataclass
class Optimizer:
    name: str = "AdamW"
    """Optimizer to use"""

    lr: float = 8e-4
    """Learning rate to use"""

    beta1: float = 0.9
    beta2: float = 0.95
    """Exponential moving average hyperparameters to use"""

    eps: float = 1e-8
    """Epsilon value to use"""

    weight_decay: float = 0.1
    """Weight decay to use"""

    implementation: Literal["for-loop", "foreach", "fused"] = "fused"
    """
    Specify which optimizer implementation to use:
    - 'fused': Use fused implementation (CUDA only) for best performance.
    - 'foreach': Use some horizontal fusion of tensors for better performance.
    - 'for-loop': Use the default implementation for the optimizer (slowest).
    - more info: https://pytorch.org/docs/stable/optim.html
    """

    early_step_in_backward: bool = False
    """
    Whether to apply optimizer in the backward. Caution, optimizer_in_backward
    is not compatible with gradients clipping, users should not call
    register_post_accumulate_grad_hook after the optimizer is built.
    """


@dataclass
class LRScheduler:
    warmup_steps: int = 200
    """
    Steps for lr scheduler warmup, normally 1/5 of --training.steps
    """

    decay_ratio: float | None = None
    """
    Controls the proportion of the training steps allocated to the learning rate decay phase.
    If `None`, the learning rate will begin decaying immediately after the warmup period.
    Otherwise, the learning rate will remain stable after the warmup period and
    only start decaying during the last `decay_ratio` portion of the total training steps.
    This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
    """

    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    """
    Learning rate decay type to use during training:
    - 'linear': linearly decays learning rate from initial to final value
    - 'sqrt': decays learning rate following a 1 minus square root curve
    - 'cosine': smoothly decays learning rate following a cosine curve
    """

    min_lr_factor: float = 0.0
    """
    Min lr ratio for lr scheduler.
    If provided, the range of decay factor is scaled from 1 to `min_lr_factor`
    to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.min_lr_factor`.
    """


@dataclass
class Training:
    dataset: str = "c4_test"
    """Dataset to use"""

    dataset_path: str | None = None
    """
    Path to the dataset in the file system. If provided, data will be
    loaded from this path instead of downloaded.
    """

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

    compile: bool = False
    """Whether to compile the model"""

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    gc_debug: bool = False
    """
    Enable GC debugging mode. This will perform gc.collect() at every step to
    detect if there is a reference cycle that includes a CUDA Tensor.
    Note that you may want to lower the training steps to avoid generating too
    many temporary files.
    """

    seed: int | None = None
    """Choose the base RNG seed used for training"""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""


@dataclass
class Parallelism:
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

    enable_compiled_autograd: bool = False
    """Enable CompiledAutograd to compile the backward."""

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

    pipeline_parallel_split_points: list[str] = field(default_factory=list)
    """
    DEPRECATED: Use module_fqns_per_model_part instead.
    Specify comma-separated names of modules to use as the beginning of a split point.
    e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
    the first containing all the layers up to layers.0,
    the second containing layers.0 and up to layers.2,
    the third containing layers.2 and all the remaining layers.
    Note: fully-automated splitting may be enabled in the future,
    but currently the split points must be specified manually.
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

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    """
    The collective to use in context parallel SDPA for kv shards exchange.
    - 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
    - 'alltoall' means to all-to-all shuffle the kv shards.
    The default value is 'allgather'.
    """

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled.
    Currently, only "dp2ep" is supported, with the following constraints:
    context_parallel_degree <= expert_parallel_degree <= data_parallel_shard_degree * context_parallel_degree
    Note that this is still an experimental feature.
    """


@dataclass
class Checkpoint:
    enable_checkpoint: bool = False
    """Whether to enable checkpoint"""

    folder: str = "checkpoint"
    """
    The folder to store the checkpoints.
    When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
    """

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """
    This option specifies the path to the initial checkpoint to load, which is
    particularly useful for resuming training from a previous run with a
    different output path or when loading a checkpoint from a pre-trained model.
    If the checkpoint folder for the current run is not empty,
    located at {--job.dump_folder}/{--checkpoint.folder}, this option will be ignored.
    This feature allows users to load an initial checkpoint from a different folder and
    continue training, saving new checkpoints to the specified folder without affecting
    the existing ones.

    Note that the path should contain the full path to the checkpoint folder,
    including the step number, if any; for example,
    "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
    """

    initial_load_model_only: bool = True
    """
    This option specifies if only the model should be loaded during the initial
    checkpoint load. The option is only used when `initial_load_path` is specified.
    If False, the checkpoint at `initial_load_path` is treated as a standard training
    checkpoint, including optimizer, lr scheduler, training states, etc.
    The default setting for this option is True. Note that you will have to use
    `--checkpoint.no_initial_load_model_only` to override the default setting.
    """

    initial_load_in_hf: bool = False
    """
    Enable the use of HuggingFace's safetensors format for checkpointing. The option
    is only used when `initial_load_path` is specified. This will load checkpoints
    in HF's model definition and safetensors format instead of the default torchtitan
    model definition and DCP format, after necessary model state dict transformation.
    `initial_load_model_only` must be true because safetensors doesn't support saving
    non-tensors. The default value is False.
    """

    last_save_model_only: bool = True
    """
    When last_save_model_only=True, only the model will be saved at the end of training,
    the last save.  With this, checkpoints can be loaded using `torch.load(..., weights_only=True)`
    after conversion.  When last_save_model_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is True.
    """

    last_save_in_hf: bool = False
    """
    Enable the use of Hugging Face's safetensors format for checkpointing. This will save the
    final checkpoints in safetensors format instead of the default DCP format, after necessary
    model state dict transformation. There will be a performance cost in using this as we need
    to consolidate the sharded tensors to full tensors as a separate step.
    last_save_model_only must be true because safetensors doesn't support saving
    non-tensors. On load, this argument isn't needed as we will detect whether the loaded
    checkpoint is in safetensors format or not. The default value is False.
    """

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """
    Converts to the specified precision when training completes and last_save_model_only=true.
    """

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """
    Which async checkpoint mode to use. Currently there are 3 different modes.
    - "disabled": synchronized checkpointing will be used.
    - "async": torch.distributed.checkpoint.async_save will be used.
    - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
      separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
      The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
      may degrade due to memory paging. For most users, "async" should suffice as the performance
      overhead is typically small (on the order of tens of seconds) compared to checkpointing
      frequency. This mode can be employed to pursue near-zero checkpointing times
      (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

    "disabled" is the default mode.
    """

    keep_latest_k: int = 10
    """
    Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
    K cannot be 1 as the last one may be in the process of being saved. As a result,
    the metadata of the last one may not be ready yet. The default value is 10 to avoid
    filling up the disk.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """
    Exclude specific keys from being loaded from the checkpoint.
    Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
    This will load the model only, excluding the specified keys.
    """

    enable_first_step_checkpoint: bool = False
    """
    Enable the checkpoint save at first step. This will save a checkpoint immediately
    after the first step to ensure checkpointing functions correctly. This is useful
    when running on a new cluster or storage to verify checkpointing without waiting
    for many steps or checkpointing too frequently. The default value is False.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
    """


@dataclass
class ActivationCheckpoint:
    mode: Literal["selective", "full", "none"] = "selective"
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


@dataclass
class Float8:
    enable_fsdp_float8_all_gather: bool = False
    """Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling"""

    precompute_float8_dynamic_scale_for_fsdp: bool = False
    """Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling"""

    recipe_name: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] | None = None
    """If specified, creates float8 config from recipe name"""

    filter_fqns: list[str] = field(default_factory=list)
    """
    Comma-separated list of fully qualified names of modules to skip applying float8 training to.
    nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
    Example: --float8.filter_fqns "attention.wq,attention.wk,attention.wv,output"
    """

    emulate: bool = False
    """
    If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only,
    as the current CI does not have sm_89 capability, required by Float8.
    Not compatible with torch.compile.
    """

    moe_fqns_prototype: list[str] | str = field(default_factory=list)
    """
    Comma-separated list of fully qualified names of MoE modules to apply float8 rowwise training to.
    This is a prototype feature that requires the torchao nightly build.
    Example: --float8.moe_fqns_prototype="experts"
    """


@dataclass
class MX:
    mxfp8_dim1_cast_kernel_choice: Literal["triton", "cuda", "torch"] = "triton"
    """Temp work around for inductor performance gap"""

    recipe_name: str = "mxfp8_cublas"
    """
    If specified, creates MX config from recipe name. See
    https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats for more information.
    """

    filter_fqns: list[str] = field(default_factory=lambda: ["output"])
    """
    Comma-separated list of fully qualified names of modules to skip applying mxfp8 training to.
    nn.Linear modules with any dim size not divisible by 16 are also always skipped due to hardware requirements.
    By default we always skip the output layer.
    Example: --mx.filter_fqns "attention.wq,attention.wk,attention.wv,output"
    """

    moe_fqns_prototype: list[str] | str = field(default_factory=list)
    """
    Comma-separated list of fully qualified names of MoE modules to apply mxfp8 training to.
    This is a prototype feature that requires the torchao nightly build.
    Example: --mx.moe_fqns_prototype="experts"
    """


@dataclass
class Comm:
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


@dataclass
class MemoryEstimation:
    enabled: bool = False
    """Whether to estimate memory usage for FSDP"""

    disable_fake_mode: bool = False
    """Whether to estimate memory under FakeTensorMode"""


@dataclass
class FaultTolerance:
    enable: bool = False
    """
    Enable TorchFT integration. When TorchFT is enabled, HSDP will be used.
    And --fault_tolerance.data_parallel_replicate_degree should be 1 and
    --fault_tolerance.group_size will be used to control the maximum
    replicate group size as the replicate group size is dynamic.
    Note that this is still an experimental feature.
    """

    process_group: str = "gloo"
    """
    The process group to use for fault tolerance. Currently, only "gloo" and "nccl" are supported.
    """

    process_group_timeout_ms: int = 10000
    """
    The process group will abort if operations don't succeed within this duration.
    Note: This currently only works with gloo process group.
    """

    replica_id: int = 0
    """The TorchFT replica ID of this run."""

    group_size: int = 0
    """
    The number of TorchFT replicate groups. This number will be used for
    dataloader to split the dataset across the replicate groups and FSDP
    dimension
    """

    min_replica_size: int = 1
    """The minimum number of FT replica for each step."""

    semi_sync_method: str | None = None
    """
    The algorithm to use for semi-sync training. Currently, only "local_sgd" and "diloco" from
    torchft are supported
    (https://github.com/pytorch/torchft/blob/360c5c534bdeac959507e9d238ba9f3902d3fda9/torchft/local_sgd.py#L41)
    """

    sync_steps: int = 5
    """
    Number of steps to wait before performing synchronization. This is only used when "semi_sync_method"
    is set.
    """

    should_quantize: bool = False
    """
    Whether to quantize the gradients before allreduce.

    Disabled by default since the quantization does utilize the GPU
    and uses more collectives. Enabling this requires knowing about
    the tradeoffs between GPU utilization and communication.


    This is only used when "semi_sync_method" is set.
    """

    fragment_sync_delay: int = 0
    """
    Controls the number of inner steps to wait before blocking on a
    model fragment's synchronization. This is the "tao" parameter in
    the Streaming DiLoCo paper.

    By default, each model fragment will be synced at the same step
    at which the allreduce is issued. Enabling delay can improve
    communication and computation overlap, but at the cost of compromising
    model quality

    This is only used when "semi_sync_method" is set.
    """

    fragment_update_alpha: float = 0.0
    """
    Determines how to mix the local and global optimized parameters

    By default, we just use the global parameters. This ensures all
    DDP replicas have the same parameters after syncrhonizing on
    the fragment. Tuning this can also affect the model quality.

    This is only used when "semi_sync_method" is set.
    """


@dataclass
class LigerKernel:
    enable_fused_linear_cross_entropy: bool = False
    """Enable Liger-Kernel's fused linear cross entropy loss for improved performance"""


@dataclass
class Experimental:
    custom_import: str = ""
    """
    This option enables the importation of external modules.
    Currently, it only supports dotted import modules (e.g., some_package.model_x).
    It is the user's responsibility to ensure that the specified path can be
    successfully imported. One method to achieve this, you can place your module
    inside the ``torchtitan/torchtitan`` folder and execute ``pip install -e .`` to
    make it available for import.
    """

    custom_args_module: str = ""
    """
    This option allows users to extend TorchTitan's existing JobConfig by extending
    a user defined JobConfig dataclass. Similar to ``--experimental.custom_model_path``, the user
    needs to ensure that the path can be imported.
    """


@dataclass
class Validation:
    enabled: bool = False
    """Enable validation to default run validation after each training loop"""

    dataset: str = "c4_validation"
    """Dataset to use for validation"""

    dataset_path: str | None = None
    """Path to dataset to use for validation"""

    local_batch_size: int = 8
    """Batch size for validation"""

    seq_len: int = 2048
    """Sequence length for validation"""

    freq: int = 10
    """Frequency of validation"""

    steps: int = -1
    """Number of steps to take in the validation set, -1 means consuming all the data in the validation dataset"""

    def __post_init__(self):
        assert (
            self.steps > 0 or self.steps == -1
        ), "validation steps must be positive or -1"


@dataclass
class JobConfig:
    """
    Default container for training configuration.
    """

    job: Job = field(default_factory=Job)
    profiling: Profiling = field(default_factory=Profiling)
    metrics: Metrics = field(default_factory=Metrics)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    float8: Float8 = field(default_factory=Float8)
    mx: MX = field(default_factory=MX)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
    liger_kernel: LigerKernel = field(default_factory=LigerKernel)
    experimental: Experimental = field(default_factory=Experimental)
    validation: Validation = field(default_factory=Validation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
