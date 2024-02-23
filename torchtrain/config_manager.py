# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

class JobConfig:
    """
    A helper class to manage the train configuration.
    """

    def __init__(self, config_path: str = None):
        self._config_path = "./torchtrain/train_configs/train_config.toml" if config_path is None else config_path
        self._import_config()

    def _import_config(self):
        with open(self._config_path, "rb") as f:
            self._config = tomllib.load(f)
            for k, v in self._config.items():
                class_type = type(k.title(), (), v)
                setattr(self, k, class_type())
            self._validate_config()

    def _validate_config(self):
        # TODO: Add more mandatory validations
        assert self.model.name and self.model.model_conf and self.model.tokenizer_path
        return True
