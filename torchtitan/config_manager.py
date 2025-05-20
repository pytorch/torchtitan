# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys

from dataclasses import asdict, dataclass, field, fields, is_dataclass, make_dataclass
from typing import Any, Literal, Type

import torch
import tyro

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.tools.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

custom_registry = tyro.constructors.ConstructorRegistry()


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

    tokenizer_path: str = "./torchtitan/datasets/tokenizer/tokenizer.model"
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

    eps: float = 1e-8
    """Epsilon value to use"""

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

    lr_min: float = 0.0
    """
    Min lr ratio for lr scheduler.
    If provided, the range of decay factor is scaled from 1 to `lr_min`
    to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.lr_min`.
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

    batch_size: int = 8
    """Batch size"""

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
    torch dtype to use for parameters when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
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
    Specify comma-separated names of modules to use as the beginning of a split point.
    e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
    the first containing all the layers up to layers.0,
    the second containing layers.0 and up to layers.2,
    the third containing layers.2 and all the remaining layers.
    Note: fully-automated splitting may be enabled in the future,
    but currently the split points must be specified manually.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the split points will be
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
    This value is used to compute the total number of microbatches by dividing batch_size with
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

    model_weights_only: bool = False
    """
    When model_weights_only=True, only model weights will be saved at the end of training.
    With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
    When model_weights_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is false.
    """

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """
    Converts to the specified precision when training completes and model_weights_only=true.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
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


@dataclass
class ActivationCheckpoint:
    mode: Literal["selective", "full", "none"] = "selective"
    """Type of activation checkpointing to use"""

    selective_ac_option: str = "2"
    """
    Selective activation checkpointing options ['int', 'op'].
    'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
    """


@dataclass
class Float8:
    enable_fsdp_float8_all_gather: bool = False
    """Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling"""

    precompute_float8_dynamic_scale_for_fsdp: bool = False
    """Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling"""

    force_recompute_fp8_weight_in_bwd: bool = False
    """
    Whether to force the recomputation of FP8 weights during backward pass.
    When using FSDP with tensorwise scaling, it is recommended to enable
    `force_recompute_fp8_weight_in_bwd` to prevent saving unsharded FP8 weights
    for backward computation.
    """

    recipe_name: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] | None = None
    """If specified, creates float8 config from recipe name"""

    filter_fqns: list[str] = field(default_factory=list)
    """
    Comma-separated list of fully qualified names of modules to skip applying float8 training to.
    nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
    Example: --float8.filter_fqns "attention.wq,attention.wk,attention.wv,output"
    """


@dataclass
class MX:
    use_fp8_dim1_cast_triton_kernel: bool = True
    """Temp work around for inductor performance gap"""

    recipe_name: Literal["mxfp8"] = "mxfp8"
    """If specified, creates float8 config from recipe name"""

    filter_fqns: list[str] = field(default_factory=lambda: ["output"])
    """
    Comma-separated list of fully qualified names of modules to skip applying mxfloat8 training to.
    nn.Linear modules with any dim size not divisible by 16 are also always skipped due to hardware requirements.
    By default we always skip the output layer.
    Example: --mx.filter_fqns "attention.wq,attention.wk,attention.wv,output"
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
    experimental: Experimental = field(default_factory=Experimental)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """
    Parses, merges, and validates a JobConfig from TOML and CLI sources.

    Configuration precedence:
        CLI args > TOML file > JobConfig defaults

    CLI arguments use the format <section>.<key> to map to TOML entries.
    Example:
        model.name →

        [model]
        name
    """

    def __init__(self, config_cls: Type[JobConfig] = JobConfig):
        self.config_cls = config_cls
        self.config: JobConfig = config_cls()
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] = sys.argv[1:]) -> JobConfig:
        toml_values = self._maybe_load_toml(args)
        config_cls = self._maybe_add_custom_args(args, toml_values)

        base_config = (
            self._dict_to_dataclass(config_cls, toml_values)
            if toml_values
            else config_cls()
        )

        self.config = tyro.cli(
            config_cls, args=args, default=base_config, registry=custom_registry
        )

        self._validate_config()

        return self.config

    def _maybe_load_toml(self, args: list[str]) -> dict[str, Any] | None:

        # 1. Check CLI
        valid_keys = {"--job.config-file", "--job.config_file"}
        for i, arg in enumerate(args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    file_path = value
                    break
            elif i < len(args) - 1 and arg in valid_keys:
                file_path = args[i + 1]
                break
        else:
            return None

        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            logger.exception(f"Error while loading config file: {file_path}")
            raise e

    def _maybe_add_custom_args(
        self, args: list[str], toml_values: dict[str, Any] | None
    ) -> Type[JobConfig]:  # noqa: B006
        """Find and merge custom arguments module with current JobConfig class"""
        module_path = None

        # 1. Check CLI
        valid_keys = {
            "--experimental.custom_args_module",
            "--experimental.custom-args-module",
        }
        for i, arg in enumerate(args):
            key = arg.split("=")[0]
            if key in valid_keys:
                module_path = arg.split("=", 1)[1] if "=" in arg else args[i + 1]
                break

        # 2. If not found in CLI, check TOML
        if not module_path and toml_values:
            experimental = toml_values.get("experimental", {})
            if isinstance(experimental, dict):
                module_path = experimental.get("custom_args_module")

        if not module_path:
            return self.config_cls

        JobConfigExtended = importlib.import_module(module_path).JobConfig
        return self._merge_configs(self.config_cls, JobConfigExtended)

    @staticmethod
    def _merge_configs(base, custom) -> Type:
        """
        Merges a base JobConfig class with user-defined extensions.

        This method creates a new dataclass type that combines fields from both `base` and `custom`,
        allowing users to extend or override JobConfig configuration structure.

        Merge behavior:
        - If a field exists in both `base` and `custom`:
            - If both field types are dataclasses, they are merged recursively.
            - Otherwise, the field from `custom` overrides the one in `base` (type, default, etc.).
        - Fields only present in `base` or `custom` are preserved as-is.
        """
        result = []
        b_map = {f.name: f for f in fields(base)}
        c_map = {f.name: f for f in fields(custom)}

        for name, f in b_map.items():
            if (
                name in c_map
                and is_dataclass(f.type)
                and is_dataclass(c_map[name].type)
            ):
                m_type = ConfigManager._merge_configs(f.type, c_map[name].type)
                result.append((name, m_type, field(default_factory=m_type)))

            # Custom field overrides base type
            elif name in c_map:
                result.append((name, c_map[name].type, c_map[name]))

            # Only in Base
            else:
                result.append((name, f.type, f))

        # Only in Custom
        for name, f in c_map.items():
            if name not in b_map:
                result.append((name, f.type, field(default_factory=f.type)))

        return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Convert dictionary to dataclass, handling nested structures."""
        if not is_dataclass(cls):
            return data

        result = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                if is_dataclass(f.type) and isinstance(value, dict):
                    result[f.name] = self._dict_to_dataclass(f.type, value)
                else:
                    result[f.name] = value
        return cls(**result)

    def _validate_config(self) -> None:
        # TODO: temporary mitigation of BC breaking change in
        #       tokenizer default path, need to remove later
        if not os.path.exists(self.config.model.tokenizer_path):
            logger.warning(
                f"Tokenizer path {self.config.model.tokenizer_path} does not exist!"
            )
            old_tokenizer_path = (
                "torchtitan/datasets/tokenizer/original/tokenizer.model"
            )
            if os.path.exists(old_tokenizer_path):
                self.config.model.tokenizer_path = old_tokenizer_path
                logger.warning(
                    f"Temporarily switching to previous default tokenizer path {old_tokenizer_path}. "
                    "Please update your config."
                )

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma seperated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Run this module directly to debug or inspect configuration parsing.
    #
    # Examples:
    #   Show help message:
    #     > python -m torchtitan.config_manager --help
    #
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config_manager --profiling.enable_memory_snapshot
    #
    # -----------------------------------------------------------------------------

    from rich import print as rprint
    from rich.pretty import Pretty

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    rprint(Pretty(config))
