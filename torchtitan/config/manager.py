# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys

from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import Any, Type

import tyro

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.tools.logging import logger

from .job_config import JobConfig


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
        config_cls = self._maybe_add_custom_config(args, toml_values)

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

    def _maybe_add_custom_config(
        self, args: list[str], toml_values: dict[str, Any] | None
    ) -> Type[JobConfig]:  # noqa: B006
        """
        Find and merge custom config module with current JobConfig class, if it is given.
        The search order is first searching CLI args, then toml config file.
        """
        module_path = None

        # 1. Check CLI
        valid_keys = {
            "--job.custom_config_module",
            "--job.custom-config-module",
        }
        for i, arg in enumerate(args):
            key = arg.split("=")[0]
            if key in valid_keys:
                module_path = arg.split("=", 1)[1] if "=" in arg else args[i + 1]
                break

        # 2. If not found in CLI, check TOML
        if not module_path and toml_values:
            job = toml_values.get("job", {})
            if isinstance(job, dict):
                module_path = job.get("custom_config_module")

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
                result.append((name, f.type, f))

        return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Convert dictionary to dataclass, handling nested structures."""
        if not is_dataclass(cls):
            return data

        valid_fields = set(f.name for f in fields(cls))
        if invalid_fields := set(data) - valid_fields:
            raise ValueError(
                f"Invalid field names in {cls} data: {invalid_fields}.\n"
                "Please modify your .toml config file or override these fields from the command line.\n"
                "Run `NGPU=1 ./run_train.sh --help` to read all valid fields."
            )

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
        if self.config.experimental.custom_args_module:
            logger.warning(
                "This field is being moved to --job.custom_config_module and "
                "will be deprecated soon. Setting job.custom_config_module to "
                "experimental.custom_args_module temporarily."
            )
            self.config.job.custom_config_module = (
                self.config.experimental.custom_args_module
            )
        # TODO: temporary mitigation of BC breaking change in hf_assets_path
        #       tokenizer default path, need to remove later
        if self.config.model.tokenizer_path:
            logger.warning(
                "tokenizer_path is deprecated, use model.hf_assets_path instead. "
                "Setting hf_assets_path to tokenizer_path temporarily."
            )
            self.config.model.hf_assets_path = self.config.model.tokenizer_path
        if not os.path.exists(self.config.model.hf_assets_path):
            logger.warning(
                f"HF assets path {self.config.model.hf_assets_path} does not exist!"
            )
            old_tokenizer_path = (
                "torchtitan/datasets/tokenizer/original/tokenizer.model"
            )
            if os.path.exists(old_tokenizer_path):
                self.config.model.hf_assets_path = old_tokenizer_path
                logger.warning(
                    f"Temporarily switching to previous default tokenizer path {old_tokenizer_path}. "
                    "Please download the new tokenizer files (python scripts/download_hf_assets.py) and update your config."
                )
        else:
            # Check if we are using tokenizer.model, if so then we need to alert users to redownload the tokenizer
            if self.config.model.hf_assets_path.endswith("tokenizer.model"):
                raise Exception(
                    "You are using the old tokenizer.model, please redownload the tokenizer ",
                    "(python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer) ",
                    " and update your config to the directory of the downloaded tokenizer.",
                )

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma separated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )


# Initialize the custom registry for tyro
custom_registry = tyro.constructors.ConstructorRegistry()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Run this module directly to debug or inspect configuration parsing.
    #
    # Examples:
    #   Show help message:
    #     > python -m torchtitan.config.manager --help
    #
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config.manager --profiling.enable_memory_snapshot
    #
    # -----------------------------------------------------------------------------

    from rich import print as rprint
    from rich.pretty import Pretty

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    rprint(Pretty(config))
