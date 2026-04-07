import importlib
import os
import sys
from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import Any, Type, get_args, get_origin

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found]

from torchtitan.tools.logging import logger

from .job_config import JobConfig


class ConfigManager:
    """
    Parses, merges, and validates a JobConfig from TOML and CLI sources.

    Configuration precedence:
        CLI args > TOML file > JobConfig defaults

    CLI arguments use the format --<section>.<key> <value> to map to TOML entries.
    Example:
        --model.name llama3 maps to:

        [model]
        name = "llama3"
    """

    def __init__(self, config_cls: Type[JobConfig] = JobConfig):
        self.config_cls = config_cls
        self.config: JobConfig = config_cls()

    def parse_args(self, args: list[str] = sys.argv[1:]) -> JobConfig:
        toml_values = self._maybe_load_toml(args)
        config_cls = self._maybe_add_custom_config(args, toml_values)

        base_config = (
            self._dict_to_dataclass(config_cls, toml_values)
            if toml_values
            else config_cls()
        )

        # Apply CLI overrides on top of the TOML-loaded config
        self.config = self._apply_cli_overrides(config_cls, base_config, args)

        self._validate_config()

        return self.config

    def _apply_cli_overrides(
        self, config_cls: Type[JobConfig], base_config: JobConfig, args: list[str]
    ) -> JobConfig:
        """Parse CLI args in --section.key value format and apply as overrides."""
        # Skip known meta-args that are handled separately
        skip_keys = {
            "--job.config-file",
            "--job.config_file",
            "--job.custom_config_module",
            "--job.custom-config-module",
        }

        i = 0
        while i < len(args):
            arg = args[i]
            if not arg.startswith("--"):
                i += 1
                continue

            # Parse key=value or key value
            if "=" in arg:
                key, value = arg[2:].split("=", 1)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                key = arg[2:]
                value = args[i + 1]
                i += 1
            else:
                # Boolean flag (--some.flag means True)
                key = arg[2:]
                value = "true"

            # Normalize key: convert hyphens to underscores
            key = key.replace("-", "_")

            if f"--{key}" in skip_keys or f"--{key.replace('_', '-')}" in skip_keys:
                i += 1
                continue

            parts = key.split(".")
            if len(parts) != 2:
                i += 1
                continue

            section, field_name = parts
            self._set_field(base_config, section, field_name, value)

            i += 1

        return base_config

    def _set_field(
        self, config: JobConfig, section: str, field_name: str, value_str: str
    ) -> None:
        """Set a field on the config object, coercing the string value to the correct type."""
        if not hasattr(config, section):
            logger.warning(f"Unknown config section: {section}")
            return

        section_obj = getattr(config, section)
        if not is_dataclass(section_obj):
            logger.warning(f"Config section '{section}' is not a dataclass")
            return

        # Find the field type
        field_type = None
        for f in fields(section_obj):
            if f.name == field_name:
                field_type = f.type
                break

        if field_type is None:
            logger.warning(f"Unknown config field: {section}.{field_name}")
            return

        coerced = self._coerce_value(value_str, field_type)
        setattr(section_obj, field_name, coerced)

    @staticmethod
    def _coerce_value(value_str: str, field_type: Any) -> Any:
        """Coerce a string value to the appropriate Python type."""
        # Handle Optional / Union types (e.g., str | None)
        origin = get_origin(field_type)
        if origin is type(int | str):  # UnionType
            # Try the non-None types
            for t in get_args(field_type):
                if t is type(None):
                    continue
                try:
                    return ConfigManager._coerce_value(value_str, t)
                except (ValueError, TypeError):
                    continue
            return value_str

        if field_type is bool:
            return value_str.lower() in ("true", "1", "yes")
        elif field_type is int:
            return int(value_str)
        elif field_type is float:
            return float(value_str)
        elif field_type is str:
            return value_str
        elif field_type == list[str]:
            return value_str.split(",")
        else:
            return value_str

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
        return cls(**result)  # pyrefly: ignore[not-callable, bad-instantiation]  # type: ignore

    def _validate_config(self) -> None:
        if not os.path.exists(self.config.model.hf_assets_path):
            logger.warning(
                f"HF assets path {self.config.model.hf_assets_path} does not exist!"
            )


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Run this module directly to debug or inspect configuration parsing.
    #
    # Examples:
    #   Parse and print a config with CLI arguments:
    #     > python -m torchtitan.config.manager --job.config_file path/to/config.toml
    #
    # -----------------------------------------------------------------------------

    # pyrefly: ignore [missing-import]
    from rich import print as rprint

    # pyrefly: ignore [missing-import]
    from rich.pretty import Pretty

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    rprint(Pretty(config))
