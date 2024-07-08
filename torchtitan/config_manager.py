# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict
from typing import Tuple, Union

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.logging_utils import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    return raw_arg.split(",")


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

    def __init__(self):
        # main parser
        self.parser = argparse.ArgumentParser(description="torchtitan arg parser.")

        self.parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="Job config file",
        )

        # job level configs
        self.parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./torchtitan/outputs",
            help="Folder to dump job outputs",
        )
        self.parser.add_argument(
            "--job.description",
            type=str,
            default="default job",
            help="Description of the job",
        )
        self.parser.add_argument(
            "--job.use_for_integration_test",
            default=False,
            action="store_true",
            help="Add this config to the integration test suite",
        )

        # profiling configs
        self.parser.add_argument(
            "--profiling.enable_profiling",
            action="store_true",
            help="Whether to enable pytorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profile_traces",
            help="Trace files location",
        )
        self.parser.add_argument(
            "--profiling.profile_freq",
            type=int,
            default=10,
            help="How often to collect profiler traces, in iterations",
        )
        self.parser.add_argument(
            "--profiling.enable_memory_snapshot",
            action="store_true",
            default=False,
            help="Whether to dump memory snapshot",
        )
        self.parser.add_argument(
            "--profiling.save_memory_snapshot_folder",
            type=str,
            default="memory_snapshot",
            help="Memeory snapshot files location",
        )

        # metrics configs
        self.parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="How often to log metrics to TensorBoard, in iterations",
        )
        self.parser.add_argument(
            "--metrics.enable_color_printing",
            default=False,
            action="store_true",
            help="Whether to enable color printing",
        )
        self.parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            help="Whether to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="Folder to dump TensorBoard states",
        )
        self.parser.add_argument(
            "--metrics.rank_0_only",
            default=True,
            action="store_true",
            help="""
                Whether to save TensorBoard metrics only for rank 0 or for all ranks.
                When pipeline_parallel_degree is > 1, this option uses the 0th rank of the last stage pipeline group,
                which is the only stage that computes loss metrics.
            """,
        )

        # model configs
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="llama",
            help="Which model to train",
        )
        self.parser.add_argument(
            "--model.flavor",
            type=str,
            default="debugmodel",
            help="Which model config to train",
        )
        self.parser.add_argument(
            "--model.norm_type",
            type=str,
            default="rmsnorm",
            help="Type of layer normalization to use [layernorm, np_layernorm, rmsnorm, fused_rmsnorm]",
        )
        self.parser.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="./torchtitan/datasets/tokenizer/tokenizer.model",
            help="Tokenizer path",
        )

        # optimizer configs
        self.parser.add_argument(
            "--optimizer.name", type=str, default="AdamW", help="Optimizer to use"
        )
        self.parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="Learning rate to use"
        )
        self.parser.add_argument(
            "--optimizer.fused",
            default=False,
            action="store_true",
            help="Whether the fused implementation(CUDA only) is used.",
        )

        # training configs
        self.parser.add_argument(
            "--training.dataset", type=str, default="c4_mini", help="Dataset to use"
        )
        self.parser.add_argument(
            "--training.dataset_path",
            type=str,
            help="""
                Path to the dataset in the file system. If provided, data will be
                loaded from this path instead of downloaded.""",
        )
        self.parser.add_argument(
            "--training.batch_size", type=int, default=8, help="Batch size"
        )
        self.parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="Sequence length"
        )
        self.parser.add_argument(
            "--training.warmup_steps",
            type=int,
            default=200,
            help="Steps for lr scheduler warmup, normally 1/5 of --training.steps",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=Union[float, int],
            default=1.0,
            help="Max norm for gradient clipping",
        )
        self.parser.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="How many train steps to run",
        )
        self.parser.add_argument(
            "--training.data_parallel_degree",
            type=int,
            default=-1,
            help="Data Parallelism degree. -1 means leftover ranks will be used (After SP/PP). 1 means disabled.",
        )
        self.parser.add_argument(
            "--training.tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor Parallelism degree. 1 means disabled.",
        )
        self.parser.add_argument(
            "--training.enable_loss_parallel",
            default=True,
            action="store_true",
            help="Whether to apply loss parallel when sequence parallel is enabled",
        )
        self.parser.add_argument(
            "--experimental.enable_async_tensor_parallel",
            default=False,
            action="store_true",
            help="Whether to apply async tensor parallel (currently only effective when compile is enabled)",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_degree",
            type=int,
            default=1,
            help="""
                Pipeline Parallelism degree, or number of ranks. 1 means disabled.
                If using looped schedules, this still specifies the number of physical ranks, not the number
                of stages.  Stages per rank are inferred from split points degree, and schedule.""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_split_points",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Specify comma-separated names of modules to use as the beginning of a split point.

                e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
                the first containing all the layers up to layers.0,
                the second containing layers.0 and up to layers.2,
                the third containing layers.2 and all the remaining layers.

                Note: fully-automated splitting may be enabled in the future,
                but currently the split points must be specified manually for both manual and tracer.""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_schedule",
            type=str,
            choices=["1f1b", "gpipe", "interleaved_1f1b"],
            default="1f1b",
            help="""
                Specify the Pipeline Parallel schedule to use.

                The schedule must be compatible with the split points and stages_per_rank.

                Looped schedules (e.g. interleaved_1f1b) require specifying pipeline_paralle_degree = number of ranks,
                and split_points = number of stages - 1""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_split_mode",
            type=str,
            choices=["manual", "tracer"],
            default="manual",
            help="""
                Specify the split method (e.g. the Pipeline Parallelism Front End)

                "manual" means each rank will construct an nn.Module with the appropriate layers and .forward
                implementation manually, and then wrap it in a PipelineStage.

                "tracer" means the full model will be initialized (via meta device) and then traced into a graph,
                split via the provided split points, unflattened into an nn.Module,
                and finally wrapped in a PipelineStage.  tracer frontend is currently more experimental.""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_microbatches",
            type=int,
            default=None,
            help="""
                How many microbatches to split the global training batch into when using pipeline parallelism.

                The global training batch size must be evenly divisible by the number of microbatches.

                The default value will be the number of pipeline stages, if unspecified.
            """,
        )
        self.parser.add_argument(
            "--training.mixed_precision_param",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="""
                torch dtype to use for parameters when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_degree > 1
            """,
        )
        self.parser.add_argument(
            "--training.mixed_precision_reduce",
            type=str,
            default="float32",
            choices=["float32"],
            help="""
                torch dtype to use for reductions when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_degree > 1
            """,
        )
        self.parser.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model",
        )
        self.parser.add_argument(
            "--training.fp8_linear",
            type=str,
            default="",
            choices=[
                "dynamic",
                "",
            ],  # TODO: add "delayed" option back in when supported
            help="""
                Type of fp8 linear quantization to apply to the model ['', 'dynamic'].
                This features requires you to install 'float8_experimental' which can be found
                here: https://github.com/pytorch-labs/float8_experimental
            """,
        )
        self.parser.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )

        # checkpointing configs
        self.parser.add_argument(
            "--checkpoint.enable_checkpoint",
            action="store_true",
            help="Whether to enable checkpoint",
        )
        self.parser.add_argument(
            "--checkpoint.folder",
            type=str,
            default="checkpoint",
            help="""
                The folder to store the checkpoints.
                When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.interval_type",
            type=str,
            default="steps",
            help="Checkpointing interval unit of measurement ['step', 'seconds']",
        )
        self.parser.add_argument(
            "--checkpoint.interval",
            type=int,
            default=500,
            help="Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type",
        )
        self.parser.add_argument(
            "--checkpoint.model_weights_only",
            action="store_true",
            help="""
                When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.export_dtype",
            type=str,
            default="float32",
            choices=["float16", "bfloat16", "float32"],
            help="""
                Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.create_seed_checkpoint",
            action="store_true",
            help="""
                Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.async_mode",
            type=str,
            default="disabled",
            help="""
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
            """,
        )
        self.parser.add_argument(
            "--checkpoint.keep_latest_k",
            type=int,
            default=0,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value.
            """,
        )
        # activation checkpointing configs
        self.parser.add_argument(
            "--activation_checkpoint.mode",
            type=str,
            default="selective",
            help="Type of activation checkpointing to use ['none', 'full', 'selective']",
        )
        self.parser.add_argument(
            "--activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        # communications library settings
        self.parser.add_argument(
            "--comm.init_timeout_seconds",
            type=int,
            default=300,
            help="Timeout for communication operations, during initialization and first train step.",
        )
        self.parser.add_argument(
            "--comm.train_timeout_seconds",
            type=int,
            default=100,
            help=(
                "Timeout for communication operations after the first train step -- "
                "usually a tighter bound than during initialization."
            ),
        )
        self.parser.add_argument(
            "--comm.trace_buf_size",
            type=int,
            default=20000,
            help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
        )

        # memory estimation settings
        self.parser.add_argument(
            "--memory_estimation.enabled",
            help="Whether to estimate memory usage for FSDP",
            action="store_true",
        )

        self.parser.add_argument(
            "--memory_estimation.disable_fake_mode",
            help="Whether to estimate memory under FakeTensorMode",
            default=False,
            action="store_true",
        )

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(
                    f"Error while loading the configuration file: {config_file}"
                )
                logger.exception(f"Error details: {str(e)}")
                raise e

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> bool:
        # TODO: Add more mandatory validations
        assert self.model.name and self.model.flavor and self.model.tokenizer_path
        return True

    def parse_args_from_command_line(
        self, args_list
    ) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    "--" + arg, action="store_true" if val else "store_false"
                )
            elif arg == "experimental.pipeline_parallel_split_points":
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
