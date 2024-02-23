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

class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - Then, Override config is loaded from command line arguments.
    """

    def __init__(self, args_list: list = sys.argv[1:]):
        self._args_list = args_list
        self._import_config()

    def _import_config(self):
        default_args, override_args = self._init_args()
        config_path = default_args.global_config_file
        args_dict = defaultdict(defaultdict)
        override_dict = self._args_to_two_level_dict(override_args)
        if config_path is None:
            args_dict = self._args_to_two_level_dict(default_args)
        else:
            with open(config_path, "rb") as f:
                args_dict = tomllib.load(f)
        for first_level_key , d in override_dict.items():
            for second_level_key, v in d.items():
                args_dict[first_level_key][second_level_key] = v
        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key , second_level_key = k.split("_", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict


    def _validate_config(self):
        # TODO: Add more mandatory validations
        assert self.model.name and self.model.config and self.model.tokenizer_path
        return True

    def _init_args(self) -> Tuple:
        """
        Each argument starts with <prefix>_ which is the section name in the toml file
        followed by name of the option in the toml file. For ex,
        model_name translates to:
            [model]
            name
        in the toml file
        """
        parser = argparse.ArgumentParser(description="TorchTrain arg parser.")
        parser.add_argument(
            "--global_config_file",
            type=str,
            default=None,
            help="job config file",
        )

        # global configs
        parser.add_argument(
            "--global_dump_folder",
            type=str,
            default="./torchtrain/outputs",
            help="folder to dump job outputs",
        )

        # profiling configs
        parser.add_argument(
            "--profiling_run_profiler",
            action="store_true",
            help="enable pytorch profiler",
        )
        parser.add_argument(
            "--profiling_save_traces_folder",
            type=str,
            default="profiling/traces",
            help="trace file location",
        )
        parser.add_argument(
            "--profiling_profile_every_x_iter",
            type=int,
            default=10,
            help="collect profiler traces every x iterations",
        )
        # metrics configs
        parser.add_argument(
            "--metrics_log_freq",
            type=int,
            default=10,
            help="how often to log metrics to TensorBoard",
        )
        parser.add_argument(
            "--metrics_enable_tensorboard",
            action="store_true",
            help="how often to log metrics to TensorBoard",
        )
        parser.add_argument(
            "--metrics_save_tb_folder",
            type=str,
            default="tb",
            help="folder to dump tensorboard state",
        )

        # model configs
        parser.add_argument(
            "--model_name",
            type=str,
            default="llama",
            help="which model to train",
        )
        parser.add_argument(
            "--model_config",
            type=str,
            default="debugmodel",
            help="which model config to train",
        )
        parser.add_argument(
            "--model_tokenizer_path",
            type=str,
            default="./torchtrain/datasets/tokenizer/tokenizer.model",
            help="tokenizer path",
        )


        # optimizer configs
        parser.add_argument(
            "--optimizer_name", type=str, default="AdamW", help="optimizer to use"
        )
        parser.add_argument("--optimizer_lr", type=float, default=8e-4, help="learning rate to use")

        # training configs
        parser.add_argument("--training_dataset", type=str, default="alpaca", help="dataset to use")
        parser.add_argument("--training_batch_size", type=int, default=8, help="batch size")
        parser.add_argument("--training_seq_len", type=int, default=2048, help="sequence length")
        parser.add_argument(
            "--training_warmup_pct",
            type=float,
            default=0.20,
            help="percentage of total training steps to use for warmup",
        )
        parser.add_argument(
            "--training_max_norm",
            type=Union[float, int],
            default=1.0,
            help="max norm for gradient clipping",
        )
        parser.add_argument(
            "--training_steps", type=int, default=-1, help="how many train steps to run"
        )
        parser.add_argument(
            "--training_data_parallel_degree",
            type=int,
            default=-1,
            help="Data Parallelism degree. -1 means leftover ranks will be used (After SP/PP). 1 means disabled.",
        )
        parser.add_argument(
            "--training_sequence_parallel_degree",
            type=int,
            default=1,
            help="Sequence Parallelism degree.  1 means disabled.",
        )
        parser.add_argument(
            "--training_pipeline_parallel_degree",
            type=int,
            default=1,
            help="Pipeline Parallelism degree (default of 1 means disabled)",
        )
        parser.add_argument(
            "--training_compile", action="store_true", help="Whether to compile the model."
        )
        parser.add_argument(
            "--training_checkpoint_interval",
            type=int,
            default=3600,
            help=(
                "Checkpointing interval. The unit of measurement is in seconds or "
                "steps depending on --training_checkpoint-internval-type."
            ),
        )
        parser.add_argument(
            "--training_checkpoint_interval_type",
            type=str,
            default="steps",
            help=(
                "The checkpointing interval unit of measurement."
                "The default value is step."
            ),
        )
        parser.add_argument(
            "--training_checkpoint_folder",
            type=str,
            default="",
            help=(
                "The folder to store the checkpoints. If this is not specified or "
                "is an empty string, checkpointing is disabled."
            ),
        )
        args = parser.parse_args(self._args_list)
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg , val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument('--'+arg,
                                        action='store_true' if val else 'store_false')
            else:
                aux_parser.add_argument('--'+arg, type=type(val))
        override_args, _ = aux_parser.parse_known_args(self._args_list)
        return args, override_args
