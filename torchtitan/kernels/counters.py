# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from collections import defaultdict
from contextlib import contextmanager
from typing import Any


_COUNTERS = defaultdict(int)
_IS_COUNTER_ENABLED = False


def is_counter_enabled() -> bool:
    global _IS_COUNTER_ENABLED
    return _IS_COUNTER_ENABLED


@contextmanager
def enable_counters():
    global _IS_COUNTER_ENABLED
    _IS_COUNTER_ENABLED = True

    yield

    _IS_COUNTER_ENABLED = False


def increment_counter(key: Any, value: int = 1) -> None:
    if not is_counter_enabled():
        return

    global _COUNTERS
    _COUNTERS[key] += value


def reset_counters() -> None:
    global _COUNTERS
    _COUNTERS = defaultdict(int)


def get_counter_value(key: Any) -> int:
    global _COUNTERS
    return _COUNTERS[key]
