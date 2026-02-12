# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import Type

import tyro

from torchtitan.tools.logging import logger


class ConfigManager:
    """
    Parses, merges, and validates a config from --model/--config and CLI sources.

    Configuration precedence:
        CLI args > config_registry function defaults

    --model selects the model (e.g., llama3, deepseek_v3).
    --config selects a config_registry function (e.g., llama3_debugmodel).
    CLI arguments use the format <section>.<key> to override config values.
    """

    def __init__(self):
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] = sys.argv[1:]):
        loaded_config, args = self._load_config(args)
        config_cls = type(loaded_config)

        self.config = tyro.cli(
            config_cls, args=args, default=loaded_config, registry=custom_registry
        )

        self._validate_config()

        return self.config

    def _load_config(self, args: list[str]) -> tuple[object, list[str]]:
        """Parse --model and --config from args, load config from config_registry.

        Both --model and --config are required.
        Returns (loaded_config, filtered_args) with --model/--config stripped.
        """
        model_name = None
        config_name = None
        filtered_args = []

        i = 0
        while i < len(args):
            arg = args[i]

            # Handle --model=X and --model X forms
            if arg.startswith("--model="):
                model_name = arg.split("=", 1)[1]
            elif arg == "--model":
                if i + 1 < len(args):
                    model_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--model requires a value")
            # Handle --config=X and --config X forms
            elif arg.startswith("--config="):
                config_name = arg.split("=", 1)[1]
            elif arg == "--config":
                if i + 1 < len(args):
                    config_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--config requires a value")
            else:
                filtered_args.append(arg)

            i += 1

        if model_name is None:
            raise ValueError(
                "--model is required. Example: --model llama3 --config llama3_debugmodel"
            )
        if config_name is None:
            raise ValueError(
                "--config is required. Example: --model llama3 --config llama3_debugmodel"
            )

        # Validate model name
        from torchtitan.models import _supported_models

        if model_name not in _supported_models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported models: {sorted(_supported_models)}"
            )

        # Import config_registry module
        module_path = f"torchtitan.models.{model_name}.config_registry"
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import config_registry for model '{model_name}': {e}"
            ) from e

        # Get the config function
        config_fn = getattr(module, config_name, None)
        if config_fn is None or not callable(config_fn):
            available = [
                name
                for name in dir(module)
                if not name.startswith("_")
                and callable(getattr(module, name))
                and name[0].islower()
            ]
            raise ValueError(
                f"Config function '{config_name}' not found in {module_path}. "
                f"Available config functions: {available}"
            )

        loaded_config = config_fn()
        return loaded_config, filtered_args

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
        if not os.path.exists(self.config.job.hf_assets_path):
            logger.warning(
                f"HF assets path {self.config.job.hf_assets_path} does not exist!"
            )
            old_tokenizer_path = (
                "torchtitan/datasets/tokenizer/original/tokenizer.model"
            )
            if os.path.exists(old_tokenizer_path):
                self.config.job.hf_assets_path = old_tokenizer_path
                logger.warning(
                    f"Temporarily switching to previous default tokenizer path {old_tokenizer_path}. "
                    "Please download the new tokenizer files (python scripts/download_hf_assets.py) and update your config."
                )
        else:
            # Check if we are using tokenizer.model, if so then we need to alert users to redownload the tokenizer
            if self.config.job.hf_assets_path.endswith("tokenizer.model"):
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
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config.manager --model llama3 --config llama3_debugmodel
    #
    #   Show help message:
    #     > python -m torchtitan.config.manager --model llama3 --config llama3_debugmodel --help
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
