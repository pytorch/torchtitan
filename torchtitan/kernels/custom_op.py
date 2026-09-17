# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import functools
import inspect
from typing import Any, Callable, Iterable, Sequence

import torch

from .accelerator import KernelBackend, get_kernel_backend
from .autotuner import AutotunedFunction
from .constants import LIBRARY_NAME
from .counters import increment_counter


def ctx_needs_gradients(ctx) -> bool:
    return any(ctx.needs_input_grad)


def ctx_save_for_backward(ctx, *args) -> None:
    if ctx_needs_gradients(ctx):
        ctx.save_for_backward(*args)


class _CustomOpMeta(type(torch.autograd.Function)):
    """lets `_Op[kernel_backend] = ...` register an op's implementation for that backend directly on
    the class - either a real `torch.autograd.Function` subclass (for a compiled kernel, `.apply()`'d),
    or a plain callable (for `KernelBackend.torch`, which needs no custom backward since it's built from
    already-differentiable torch ops). Assignment via `[]` requires a metaclass in Python, there's no
    per-class `__setitem__` equivalent."""

    def __setitem__(cls, kernel_backend: KernelBackend, function: type[torch.autograd.Function] | Callable) -> None:
        cls.functions[kernel_backend] = function

    def __getitem__(cls, kernel_backend: KernelBackend) -> type[torch.autograd.Function] | Callable:
        if kernel_backend not in cls.functions:
            raise NotImplementedError(f"{cls.__name__} has nothing registered for kernel_backend ({kernel_backend})")

        return cls.functions[kernel_backend]


class CustomOp(torch.autograd.Function, metaclass=_CustomOpMeta):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.functions: dict[KernelBackend, type[torch.autograd.Function] | Callable] = {}

    @classmethod
    def run(cls, kernel_backend: KernelBackend | None = None, **kwargs) -> Any:
        if kernel_backend is None:
            kernel_backend = get_kernel_backend()
        else:
            assert kernel_backend.verify_accelerator()

        if kernel_backend is None:
            raise ValueError("code is not supposed to reach here! kernel_backend was not inferrable")

        increment_counter(cls._get_key(kernel_backend))
        function = cls.functions[kernel_backend]

        if kernel_backend == KernelBackend.torch:
            return function(**kwargs)

        if kernel_backend in cls.functions:
            return function.apply(*tuple(kwargs.values()))

    @classmethod
    def _get_key(cls, kernel_backend: KernelBackend) -> str:
        return f"{cls.__name__}-{kernel_backend.value}"


def xma_op(
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        # support for autotuned function with custom op
        if isinstance(func, AutotunedFunction):
            autotuned_function = func

            @functools.wraps(autotuned_function.function)
            def func(*args, **kwargs):
                return autotuned_function(*args, **kwargs)

            func.__signature__ = autotuned_function.exposed_signature

        custom_op = torch.library.custom_op(
            f"{LIBRARY_NAME}::{func.__name__}",
            func,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )

        if fake_func is not None:
            custom_op.register_fake(fake_func)

        def _run(*args, **kwargs):
            return custom_op(*args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__

        return _run

    return _inner
