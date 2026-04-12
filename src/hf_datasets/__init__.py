from dataclasses import dataclass
from typing import Callable

__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable
