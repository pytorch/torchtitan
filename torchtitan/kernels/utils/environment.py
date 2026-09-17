# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
from contextlib import contextmanager


def get_boolean_env_variable(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ["1", "true"]


@contextmanager
def environment(env: dict):
    original_env = {}
    for key, value in env.items():
        original_env[key] = os.environ.get(key, "")
        os.environ[key] = value

    try:
        yield
    finally:
        for key, value in original_env.items():
            os.environ[key] = value
