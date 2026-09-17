# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import inspect
import os
from shutil import rmtree
from typing import Callable
from uuid import uuid4

import torch
from torch.utils.cpp_extension import load as load_cpp_extension


_CPP_MODULE_PREFIX = "xma"
_GLOBAL_RANK = int(os.getenv("RANK", 0))
_WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

_ALL_COMPILED_MODULES = {}


def _get_cpp_function(
    function_name: str, module_name: str, source_files: list[str], build_directory: str, is_cuda: bool, is_mps: bool
) -> Callable:
    module_name = f"{_CPP_MODULE_PREFIX}_{module_name}"

    if is_cuda:
        extra_cflags = ["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"]
        extra_cuda_cflags = ["-O3", "-lineinfo"]
        extra_ldflags = None
        extra_include_paths = [
            os.path.dirname(__file__),  # xma/include
            os.path.dirname(os.path.dirname(__file__)) + "/cutlass/include",  # cutlass
            os.path.dirname(os.path.dirname(__file__)) + "/cutlass/tools/util/include",  # cutlass
        ]
    elif is_mps:
        extra_cflags = ["-O3", "-Wall", "-std=c++17"]
        extra_cuda_cflags = None
        extra_ldflags = ["-framework", "Metal", "-framework", "Foundation"]
        extra_include_paths = [os.path.dirname(__file__)]
    else:
        raise ValueError

    module = _ALL_COMPILED_MODULES.get(module_name, None)

    if module is None:
        if torch.distributed.is_initialized():
            os.makedirs(build_directory, exist_ok=True)

            if _GLOBAL_RANK == 0:
                module = load_cpp_extension(
                    module_name,
                    sources=source_files,
                    with_cuda=is_cuda,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory,
                    verbose=True,
                )

            torch.distributed.barrier()

            if _GLOBAL_RANK != 0:
                module = load_cpp_extension(
                    module_name,
                    sources=source_files,
                    with_cuda=is_cuda,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory,
                    verbose=True,
                )
        else:
            if _WORLD_SIZE > 1:
                build_directory = os.path.join(build_directory, str(uuid4()))

            os.makedirs(build_directory, exist_ok=True)

            module = load_cpp_extension(
                module_name,
                sources=source_files,
                with_cuda=is_cuda,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
                verbose=True,
            )

            if _WORLD_SIZE > 1:
                rmtree(build_directory, ignore_errors=True)

        _ALL_COMPILED_MODULES[module_name] = module

    return getattr(module, function_name)


def cpp_jit(
    function_name: str | None = None,
    extra_source_files: list[str] = [],
    build_directory: str | None = None,
    depth: int = 1,
    is_cuda: bool = False,
    is_mps: bool = False,
) -> Callable:
    """wrapper to compile C++/CUDA source code at runtime.

    Args:
        function_name (str | None, optional): name of the function to expose from the C++ file, the python function
            name should match the funcion name in the C++ file if this is not specified. Defaults to None.
        extra_source_files (list[str], optional): any extra files to use for compilation, by default it scans the
            directory of the python stub file. Defaults to [].
        build_directory (str | None, optional): directory in which to place the build artifacts. Defaults to None.
        depth (int, optional): number of times dirname is called to get the build path. Defaults to 2.

    Returns:
        Callable: returns the wrapped function that can be used to call the C++ functions from python
    """
    cpp_function = None
    args_spec = None

    source_files = []
    source_files.extend(extra_source_files)

    calling_filename = inspect.stack()[1].filename
    calling_directory = os.path.dirname(calling_filename)

    if is_cuda:
        extensions = [".cu", ".cpp"]
        assert not is_mps
    elif is_mps:
        extensions = [".mm"]
        assert not is_cuda
    else:
        raise ValueError

    for dirname, _, filenames in os.walk(calling_directory):
        filenames = [os.path.join(dirname, f) for f in filenames]
        filenames = filter(lambda f: os.path.splitext(f)[1] in extensions, filenames)
        source_files.extend(filenames)

    if build_directory is None:
        module_name = calling_directory
        for _ in range(depth):
            module_name = os.path.dirname(module_name)
        module_name = os.path.basename(module_name)

        build_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", module_name)

    def _run(*args, **kwargs):
        nonlocal cpp_function

        if cpp_function is None:
            cpp_function = _get_cpp_function(
                function_name=_run.__name__,
                module_name=module_name,
                source_files=source_files,
                build_directory=build_directory,
                is_cuda=is_cuda,
                is_mps=is_mps,
            )

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return cpp_function(*full_args)

    def _wrapper(function: Callable) -> Callable:
        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        _run.__doc__ = function.__doc__
        _run.__name__ = function.__name__ if function_name is None else function_name
        _run.__signature__ = inspect.signature(function)

        return _run

    return _wrapper
