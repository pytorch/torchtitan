# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys

from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import Type

import tyro

from torchtitan.tools.logging import logger


class ConfigManager:
    """
    Parses, merges, and validates a config from Python config files and CLI sources.

    Configuration precedence:
        CLI args > Python config file > config defaults

    Python config files define a `default_config` variable.
    CLI arguments use the format <section>.<key> to override config values.
    """

    def __init__(self, config_cls: Type | None = None):
        if config_cls is None:
            from torchtitan.trainer import Trainer

            config_cls = Trainer.Config
        self.config_cls = config_cls
        self.config = config_cls()
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] = sys.argv[1:]):
        loaded_config = self._maybe_load_python_config(args)

        if loaded_config:
            # Use the loaded config's type (which may already be a merged type
            # if the Python config did its own merge)
            config_cls = type(loaded_config)
            base_config = loaded_config
        else:
            config_cls = self.config_cls
            base_config = config_cls()

        self.config = tyro.cli(
            config_cls, args=args, default=base_config, registry=custom_registry
        )

        self._validate_config()

        return self.config

    def _maybe_load_python_config(self, args: list[str]):
        """Load a Python config file that defines a `default_config` variable.

        If multiple --job.config_file args are present, the last one wins
        (consistent with standard CLI override behavior).
        """
        valid_keys = {
            "--job.config-file",
            "--job.config_file",
        }
        file_path = None
        for i, arg in enumerate(args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    file_path = value
            elif i < len(args) - 1 and arg in valid_keys:
                file_path = args[i + 1]

        if file_path is None:
            return None

        try:
            spec = importlib.util.spec_from_file_location("_user_config", file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load config file: {file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.exception(f"Error while loading config file: {file_path}")
            raise e

        if not hasattr(module, "default_config"):
            raise ValueError(
                f"Config file {file_path} must define a 'default_config' variable"
            )
        return module.default_config

    @staticmethod
    def _merge_configs(base, custom) -> Type:
        """
        Merges a base config class with user-defined extensions.

        This method creates a new dataclass type that combines fields from both `base` and `custom`,
        allowing users to extend or override configuration structure.

        Merge behavior:
        - If a field exists in both `base` and `custom`:
            - If both field types are dataclasses, they are merged recursively.
            - Otherwise, the field from `custom` overrides the one in `base` (type, default, etc.).
        - Fields only present in `base` or `custom` are preserved as-is.
        """
        result: list[str | tuple[str, Any] | tuple[str, Any, Any]] = []
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

    def _validate_config(self) -> None:
        # TODO: temporary mitigation of BC breaking change in hf_assets_path
        #       tokenizer default path, need to remove later
        if (
            hasattr(self.config.model, "tokenizer_path")
            and self.config.model.tokenizer_path
        ):
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

    try:

        # pyrefly: ignore[missing-import]
        from rich import print as rprint

        # pyrefly: ignore[missing-import]
        from rich.pretty import Pretty

        config_manager = ConfigManager()
        config = config_manager.parse_args()

        rprint(Pretty(config))
    except ImportError:
        config_manager = ConfigManager()
        config = config_manager.parse_args()
        logger.info(config)
        logger.warning("rich is not installed, show the raw config")
