import warnings
from enum import Enum
from typing import  Type
import torch
from pydantic import ValidationError


def print_rank_0(message: str):
    """If torch.distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def warn_rank_0(message: str):
    """If torch.distributed is initialized, print only on rank 0."""
    message_with_color_code = f"\033[91m {message} \033[00m"
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            warnings.warn(message_with_color_code)
    else:
        warnings.warn(message_with_color_code)


def parse_enum_by_name(name: str, enum_type: Type[Enum]) -> Enum:
    try:
        return enum_type[name]
    except KeyError:
        raise ValidationError(f"Invalid {enum_type} member name: {name}")

