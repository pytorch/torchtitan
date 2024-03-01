# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import sys
from collections import defaultdict
from typing import Union

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    """

    def parse_args(self, args_list: list = sys.argv[1:]):
        args = JobConfig.init_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            with open(config_file, "rb") as f:
                for k, v in tomllib.load(f).items():
                    # to prevent overwrite of non-specified keys
                    args_dict[k] |= v
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

    def _validate_config(self):
        # TODO: Add more mandatory validations
        assert self.model.name and self.model.flavor and self.model.tokenizer_path
        return True

    @staticmethod
    def init_args_from_command_line(
        args_list: list = sys.argv[1:],
    ) -> argparse.Namespace:
        """
        Each argument starts with <prefix>_ which is the section name in the toml file
        followed by name of the option in the toml file. For ex,
        model.name translates to:
            [model]
            name
        in the toml file
        """
        parser = argparse.ArgumentParser(description="TorchTrain arg parser.")
        parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="job config file",
        )

        # misc configs
        parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./torchtrain/outputs",
            help="folder to dump job outputs",
        )
        parser.add_argument(
            "--job.description",
            type=str,
            default="default job",
            help="description of the job",
        )
        # profiling configs
        parser.add_argument(
            "--profiling.run_profiler",
            action="store_true",
            help="enable pytorch profiler",
        )
        parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profiling/traces",
            help="trace file location",
        )
        parser.add_argument(
            "--profiling.profile_every_x_iter",
            type=int,
            default=10,
            help="collect profiler traces every x iterations",
        )
        # metrics configs
        parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            help="whether to log metrics to TensorBoard",
        )
        parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="how often to log metrics to TensorBoard",
        )
        parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="folder to dump tensorboard state",
        )

        # model configs
        parser.add_argument(
            "--model.name",
            type=str,
            default="llama",
            help="which model to train",
        )
        parser.add_argument(
            "--model.flavor",
            type=str,
            default="debugmodel",
            help="which model config to train",
        )
        parser.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="./torchtrain/datasets/tokenizer/tokenizer.model",
            help="tokenizer path",
        )

        # optimizer configs
        parser.add_argument(
            "--optimizer.name", type=str, default="AdamW", help="optimizer to use"
        )
        parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="learning rate to use"
        )

        # training configs
        parser.add_argument(
            "--training.dataset", type=str, default="alpaca", help="dataset to use"
        )
        parser.add_argument(
            "--training.batch_size", type=int, default=8, help="batch size"
        )
        parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="sequence length"
        )
        parser.add_argument(
            "--training.warmup_steps",
            type=int,
            default=200,
            help="steps for lr scheduler warmup",
        )
        parser.add_argument(
            "--training.max_norm",
            type=Union[float, int],
            default=1.0,
            help="max norm for gradient clipping",
        )
        parser.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="how many train steps to run",
        )
        parser.add_argument(
            "--training.data_parallel_degree",
            type=int,
            default=-1,
            help="Data Parallelism degree. -1 means leftover ranks will be used (After SP/PP). 1 means disabled.",
        )
        parser.add_argument(
            "--training.sequence_parallel_degree",
            type=int,
            default=1,
            help="Sequence Parallelism degree.  1 means disabled.",
        )
        parser.add_argument(
            "--training.pipeline_parallel_degree",
            type=int,
            default=1,
            help="Pipeline Parallelism degree (default of 1 means disabled)",
        )
        parser.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model.",
        )
        parser.add_argument(
            "--training.checkpoint_interval",
            type=int,
            default=3600,
            help=(
                "Checkpointing interval. The unit of measurement is in seconds or "
                "steps depending on --training.checkpoint-internval-type."
            ),
        )
        parser.add_argument(
            "--training.checkpoint_interval_type",
            type=str,
            default="steps",
            help=(
                "The checkpointing interval unit of measurement."
                "The default value is step."
            ),
        )
        parser.add_argument(
            "--training.checkpoint_folder",
            type=str,
            default="",
            help=(
                "The folder to store the checkpoints. If this is not specified or "
                "is an empty string, checkpointing is disabled."
            ),
        )
        parser.add_argument(
            "--training.fp8_linear_type",
            type=str,
            default="",
            choices=[
                "dynamic",
                "",
            ],  # TODO: add "delayed" option back in when supported
            help="Type of fp8 linear quantization to apply to the model",
        )
        parser.add_argument(
            "--training.enable_selective_ac",
            action="store_true",
            help="whether to enable selective activation checkpointing",
        )
        return parser.parse_args(args_list)
