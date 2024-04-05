# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import sys
from collections import defaultdict
from typing import Tuple, Union

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtrain.logging_utils import logger


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
        self.parser = argparse.ArgumentParser(description="TorchTrain arg parser.")
        self.parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="job config file",
        )

        # job level configs
        self.parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./torchtrain/outputs",
            help="folder to dump job outputs",
        )
        self.parser.add_argument(
            "--job.description",
            type=str,
            default="default job",
            help="description of the job",
        )
        self.parser.add_argument(
            "--job.use_for_integration_test",
            default=False,
            action="store_true",
            help="add this config to integration test suite",
        )
        # profiling configs
        self.parser.add_argument(
            "--profiling.run_profiler",
            action="store_true",
            help="enable pytorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profiling/traces",
            help="trace file location",
        )
        self.parser.add_argument(
            "--profiling.profile_every_x_iter",
            type=int,
            default=10,
            help="collect profiler traces every x iterations",
        )
        # metrics configs
        self.parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            help="whether to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="how often to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="folder to dump tensorboard state",
        )

        # model configs
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="llama",
            help="which model to train",
        )
        self.parser.add_argument(
            "--model.flavor",
            type=str,
            default="debugmodel",
            help="which model config to train",
        )
        self.parser.add_argument(
            "--model.norm_type",
            type=str,
            default="rmsnorm",
            help="Layer Normalization type to use [layernorm, np_layernorm, rmsnorm, fused_rmsnorm]",
        )
        self.parser.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="./torchtrain/datasets/tokenizer/tokenizer.model",
            help="tokenizer path",
        )

        # optimizer configs
        self.parser.add_argument(
            "--optimizer.name", type=str, default="AdamW", help="optimizer to use"
        )
        self.parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="learning rate to use"
        )

        # training configs
        self.parser.add_argument(
            "--training.dataset", type=str, default="alpaca", help="dataset to use"
        )
        self.parser.add_argument(
            "--training.dataset_path",
            type=str,
            help=(
                "Path to the dataset in the file system. If provided, data will be"
                "loaded from this path instead of downloaded.",
            ),
        )
        self.parser.add_argument(
            "--training.batch_size", type=int, default=8, help="batch size"
        )
        self.parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="sequence length"
        )
        self.parser.add_argument(
            "--training.warmup_steps",
            type=int,
            default=200,
            help="steps for lr scheduler warmup",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=Union[float, int],
            default=1.0,
            help="max norm for gradient clipping",
        )
        self.parser.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="how many train steps to run",
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
            help="whether to enable loss parallel when sequence parallel is enabled",
        )
        self.parser.add_argument(
            "--training.pipeline_parallel_degree",
            type=int,
            default=1,
            help="Pipeline Parallelism degree (default of 1 means disabled)",
        )
        self.parser.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model.",
        )
        self.parser.add_argument(
            "--training.checkpoint_interval",
            type=int,
            default=3600,
            help=(
                "Checkpointing interval. The unit of measurement is in seconds or "
                "steps depending on --training.checkpoint-internval-type."
            ),
        )
        self.parser.add_argument(
            "--training.checkpoint_interval_type",
            type=str,
            default="steps",
            help=(
                "The checkpointing interval unit of measurement."
                "The default value is step."
            ),
        )
        self.parser.add_argument(
            "--training.checkpoint_folder",
            type=str,
            default="",
            help=(
                "The folder to store the checkpoints. If this is not specified or "
                "is an empty string, checkpointing is disabled."
            ),
        )
        self.parser.add_argument(
            "--training.fp8_linear",
            type=str,
            default="",
            choices=[
                "dynamic",
                "",
            ],  # TODO: add "delayed" option back in when supported
            help="Type of fp8 linear quantization to apply to the model",
        )
        self.parser.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )

        # activation checkpointing
        self.parser.add_argument(
            "--activation_checkpoint.mode",
            type=str,
            default="selective",
            help=" ['none', 'full', 'selective'] = type of activation checkpointing to use",
        )
        self.parser.add_argument(
            "--activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="['int', 'op'] = selective activation checkpointing options, 'int' for every nth layer, or 'op' for op level ac.",
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
            default=5,
            help=(
                "Timeout for communication operations after the first train step-"
                "usually a tighter bound than during initialization."
            ),
        )
        self.parser.add_argument(
            "--comm.trace_buf_size",
            type=int,
            default=20000,
            help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
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
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
