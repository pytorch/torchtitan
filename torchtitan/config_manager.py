# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import tyro

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Job:
    config_file: Optional[str] = None
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

    rank_0_only: bool = True
    """
    Whether to save TensorBoard metrics only for rank 0 or for all ranks.
    When pipeline_parallel_degree is > 1, this option uses the 0th rank of the last stage pipeline group,
    which is the only stage that computes loss metrics.
    """

    enable_wandb: bool = False
    """Whether to log metrics to Weights & Biases"""


@dataclass
class Model:
    name: str = "llama"
    """Which model to train"""

    flavor: str = "debugmodel"
    """Which model config to train"""

    norm_type: str = "rmsnorm"
    """Type of layer normalization to use [layernorm, np_layernorm, rmsnorm, fused_rmsnorm]"""

    tokenizer_path: str = "./torchtitan/datasets/tokenizer/tokenizer.model"
    """Tokenizer path"""


@dataclass
class Optimizer:
    name: str = "AdamW"
    """Optimizer to use"""

    lr: float = 8e-4
    """Learning rate to use"""

    fused: bool = False
    """Whether the fused implementation (CUDA only) is used"""

    early_step_in_backward: bool = False
    """
    Whether to apply optimizer in the backward. Caution, optimizer_in_backward
    is not compatible with gradients clipping, users should not call
    register_post_accumulate_grad_hook after the optimizer is built.
    """


@dataclass
class Training:
    dataset: str = "c4_mini"
    """Dataset to use"""

    dataset_path: Optional[str] = None
    """
    Path to the dataset in the file system. If provided, data will be
    loaded from this path instead of downloaded.
    """

    batch_size: int = 8
    """Batch size"""

    seq_len: int = 2048
    """Sequence length"""

    warmup_steps: int = 200
    """Steps for lr scheduler warmup, normally 1/5 of --training.steps"""

    max_norm: Union[float, int] = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """How many train steps to run"""

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

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    disable_loss_parallel: bool = False
    """Whether to apply loss parallel when sequence parallel is enabled"""

    mixed_precision_param: str = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    mixed_precision_reduce: str = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    compile: bool = False
    """Whether to compile the model"""

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    seed: Optional[int] = None
    """Choose the base RNG seed used for training"""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""


@dataclass
class Experimental:
    enable_async_tensor_parallel: bool = False
    """Whether to apply async tensor parallel (currently only effective when compile is enabled)"""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    pipeline_parallel_split_points: List[str] = field(default_factory=list)
    """
    Specify comma-separated names of modules to use as the beginning of a split point.

    e.g. "layers.0" "layers.2" will cause the model to be split into 3 stages,
    the first containing all the layers up to layers.0,
    the second containing layers.0 and up to layers.2,
    the third containing layers.2 and all the remaining layers.

    Note: fully-automated splitting may be enabled in the future,
    but currently the split points must be specified manually.
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.

    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1.
    """

    pipeline_parallel_schedule_csv: str = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use.
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    pipeline_parallel_microbatches: Optional[int] = None
    """
    How many microbatches to split the global training batch into when using pipeline parallelism.

    The global training batch size must be evenly divisible by the number of microbatches.

    The default value will be the number of pipeline stages, if unspecified.
    """

    enable_compiled_autograd: bool = False
    """Enable CompiledAutograd to compile the backward."""

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_rotate_method: str = "allgather"
    """
    The collective to use in context parallel SDPA for kv shards exchange.

    'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,

    'alltoall' means to all-to-all shuffle the kv shards.

    The default value is 'allgather'.
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

    interval_type: str = "steps"
    """Checkpointing interval unit of measurement ['step', 'seconds']"""

    interval: int = 500
    """Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type"""

    model_weights_only: bool = False
    """
    When model_weights_only=True, only model weights will be saved at the end of training.
    With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
    When model_weights_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is false.
    """

    export_dtype: str = "float32"
    """
    Converts to the specified precision when training completes and model_weights_only=true.
    Currently supports float32, float16, and bfloat16.
    The default value is float32.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
    """

    async_mode: str = "disabled"
    """
    Which async checkpoint mode to use. Currently there are 3 different modes.
    1. "disabled": synchronized checkpointing will be used.
    2. "async": torch.distributed.checkpoint.async_save will be used.
    3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
       space and creates a separate process for faster GPU->CPU transfer
       performance and eliminating GIL contention. The cost is increased CPU
       memory usage. If insufficient CPU memory is available, performance may
       degrade due to memory paging. For most users, "async" should suffice as
       the performance overhead is typically small (on the order of tens of
       seconds) compared to checkpointing frequency. This mode can be employed
       to pursue near-zero checkpointing times (e.g., < 1 second) given
       appropriate hardware support such as ample CPU memory and fast PCIe.

    "disabled" is the default mode.
    """

    keep_latest_k: int = 0
    """
    Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
    0 is the default value.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""


@dataclass
class ActivationCheckpoint:
    mode: str = "selective"
    """Type of activation checkpointing to use ['none', 'full', 'selective']"""

    selective_ac_option: str = "2"
    """
    Selective activation checkpointing options ['int', 'op'].
    'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
    """


@dataclass
class Float8:
    enable_float8_linear: bool = False
    """
    If true, swaps `torch.nn.Linear` with `Float8Linear`.
    This feature requires you to install 'torchao' which can be found
    here: https://github.com/pytorch/ao
    """

    enable_fsdp_float8_all_gather: bool = False
    """Whether enable float8 all-gather in FSDP"""

    precompute_float8_dynamic_scale_for_fsdp: bool = False
    """Whether precompute float8 scales dynamically for FSDP"""

    scaling_type_input: str = "dynamic"
    """float8 scaling for input, dynamic (default) or delayed"""

    scaling_type_weight: str = "dynamic"
    """float8 scaling for weight, dynamic (default) or delayed"""

    scaling_type_grad_output: str = "dynamic"
    """float8 scaling for grad output, dynamic (default) or delayed"""


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


@dataclass
class MemoryEstimation:
    enabled: bool = False
    """Whether to estimate memory usage for FSDP"""

    disable_fake_mode: bool = False
    """Whether to estimate memory under FakeTensorMode"""


@dataclass
class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    job: Job = field(default_factory=Job)
    profiling: Profiling = field(default_factory=Profiling)
    metrics: Metrics = field(default_factory=Metrics)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    training: Training = field(default_factory=Training)
    experimental: Experimental = field(default_factory=Experimental)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    float8: Float8 = field(default_factory=Float8)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_instance(self, instance) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(instance, f.name, getattr(self, f.name)))

    def parse_args(self, args=None) -> None:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        toml_data = self.maybe_load_toml(args or sys.argv[1:])
        defaults = self.__class__
        if toml_data:
            defaults = self.dict_to_dataclass(defaults, toml_data)
        final_config = tyro.cli(self.__class__, args=args, default=defaults)
        self.update_instance(final_config)
        self.validate_config()

    def maybe_load_toml(self, args):
        config_flags = {"--job.config-file", "--job.config_file"}
        for i, arg in enumerate(args[:-1]):
            if arg in config_flags:
                file_path = args[i + 1]
                try:
                    with open(file_path, "rb") as f:
                        return tomllib.load(f)
                except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                    logger.exception(f"Error while loading config file: {file_path}")
                    raise e
        return None

    def dict_to_dataclass(self, config_class, data) -> Any:
        """Recursively convert dictionaries to nested dataclasses."""
        if not is_dataclass(config_class):
            return data
        kwargs = {}
        for f in fields(config_class):
            if f.name in data:
                value = data[f.name]
                if is_dataclass(f.type) and isinstance(value, dict):
                    kwargs[f.name] = self.dict_to_dataclass(f.type, value)
                else:
                    kwargs[f.name] = value
        return config_class(**kwargs)

    def validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert self.model.name, "Model name is required"
        assert self.model.flavor, "Model flavor is required"
        assert self.model.tokenizer_path, "Model tokenizer path is required"
