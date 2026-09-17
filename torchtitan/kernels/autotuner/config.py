# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product
from typing import Callable


class AutotuneConfig:
    def __init__(self, config: dict, condition: Callable = None) -> None:
        self.config = config
        self.condition = condition

    def get_key_values(self) -> dict:
        return self.config

    def is_condition_valid(self, **kwargs) -> bool:
        # note that here we override the values from the args passed by the user
        kwargs.update(self.get_key_values())
        return True if self.condition is None else self.condition(**kwargs)

    def __repr__(self) -> str:
        return str(self.config)


def get_cartesian_product_autotune_configs(
    condition: Callable = None, **kwargs: dict[str, list]
) -> list[AutotuneConfig]:
    configs = []
    all_values = product(*list(kwargs.values()))

    for values in all_values:
        config = {key: value for key, value in zip(kwargs.keys(), values)}
        config = AutotuneConfig(config, condition=condition)
        configs.append(config)

    return configs
