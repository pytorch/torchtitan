# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load Python configuration recipes from the command line."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import pprint
from typing import Any

from torchtitan.config.configurable import Configurable
from torchtitan.config.override import (
    apply_overrides,
    OverrideConfig,
    parse_cli_imports,
)


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load a Python config recipe with a deliberately small CLI surface."""

    def load(self, args: list[str] | None = None) -> Configurable.Config:
        namespace = self._parser().parse_args(args)
        config = self._load_config(namespace.module, namespace.config)

        if namespace.override:
            apply_overrides(
                OverrideConfig(imports=parse_cli_imports(namespace.override)), config
            )

        self._apply_launch_options(config, namespace)
        self._validate_assets_path(config)

        if namespace.print_config:
            pprint.pprint(config)
            raise SystemExit(0)
        return config

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Load and run a TorchTitan Python configuration recipe."
        )
        parser.add_argument(
            "--module",
            required=True,
            help="Config module shorthand or fully qualified Python module.",
        )
        parser.add_argument(
            "--config",
            required=True,
            help="Config factory function in the selected module.",
        )
        parser.add_argument(
            "--override",
            action="append",
            default=[],
            metavar="TARGET[=JSON]",
            help=(
                "Apply a registered config override. Repeat this flag to apply "
                "multiple overrides."
            ),
        )
        parser.add_argument(
            "--comm-backend",
            help="Override the distributed communication backend.",
        )
        parser.add_argument(
            "--output-dir",
            help="Override the root output directory.",
        )
        parser.add_argument(
            "--resume-step",
            type=int,
            help="Checkpoint step to load; -1 selects the latest checkpoint.",
        )
        parser.add_argument(
            "--print-config",
            action="store_true",
            help="Print the resolved config and exit.",
        )
        return parser

    @staticmethod
    def _load_config(module_name: str, config_name: str) -> Configurable.Config:
        """Call ``config_name`` from a built-in shorthand or Python module."""
        from torchtitan.experiments import _supported_experiments
        from torchtitan.models import _supported_models

        all_supported = _supported_models | _supported_experiments
        candidates = (
            (
                f"torchtitan.models.{module_name}.config_registry",
                f"torchtitan.experiments.{module_name}.config_registry",
                f"torchtitan.rl.examples.{module_name}.config_registry",
            )
            if module_name in all_supported
            else (f"{module_name}.config_registry", module_name)
        )

        module = None
        module_path = None
        import_errors: list[ImportError] = []
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
                module_path = candidate
                break
            except ImportError as error:
                missing_candidate = error.name is not None and (
                    error.name == candidate or candidate.startswith(f"{error.name}.")
                )
                if not missing_candidate:
                    raise
                import_errors.append(error)

        if module is None:
            message = (
                f"Cannot import config module {module_name!r}. Tried: "
                f"{', '.join(candidates)}."
            )
            if module_name in all_supported:
                message += f" Supported shorthands: {sorted(all_supported)}"
            raise ImportError(message) from import_errors[-1]

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
                f"Config function {config_name!r} not found in {module_path}. "
                f"Available config functions: {available}"
            )
        config = config_fn()
        if not isinstance(config, Configurable.Config):
            raise TypeError(
                f"{module_path}.{config_name} must return Configurable.Config, "
                f"got {type(config).__qualname__}."
            )
        return config

    @staticmethod
    def _apply_launch_options(
        config: Configurable.Config, args: argparse.Namespace
    ) -> None:
        if args.output_dir is not None:
            ConfigLoader._set_attr(config, "dump_folder", args.output_dir)

        if args.comm_backend is not None:
            trainer_config = ConfigLoader._trainer_config(config)
            trainer_config.comm.backend = args.comm_backend

        if args.resume_step is not None:
            trainer_config = ConfigLoader._trainer_config(config)
            checkpointer = trainer_config.checkpointer
            if checkpointer is None:
                raise ValueError(
                    "--resume-step requires checkpointing to be configured."
                )
            checkpointer.load_step = args.resume_step

    @staticmethod
    def _trainer_config(config: Configurable.Config):
        from torchtitan.trainer import Trainer

        if isinstance(config, Trainer.Config):
            return config

        from torchtitan.rl.controller import Controller

        if isinstance(config, Controller.Config):
            return config.trainer
        raise ValueError(
            "--comm-backend and --resume-step require a Trainer.Config or "
            "Controller.Config recipe."
        )

    @staticmethod
    def _set_attr(config: object, name: str, value: Any) -> None:
        if not hasattr(config, name):
            raise ValueError(
                f"--{name.replace('_', '-')} is not supported by "
                f"{type(config).__qualname__}."
            )
        setattr(config, name, value)

    @staticmethod
    def _validate_assets_path(config: object) -> None:
        assets_path = getattr(config, "hf_assets_path", None)
        if assets_path is None:
            return
        if not os.path.exists(assets_path):
            logger.warning("HF assets path %s does not exist!", assets_path)
        elif assets_path.endswith("tokenizer.model"):
            raise ValueError(
                "hf_assets_path must name an assets directory, not the legacy "
                "tokenizer.model file. Download the Hugging Face assets and "
                "update the recipe."
            )
