# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from ..device import Accelerator
from ..utils import get_boolean_env_variable
from .config import AutotuneConfig


_XMA_PRINT_AUTOTUNING = get_boolean_env_variable("XMA_PRINT_AUTOTUNING", False)
_SEPARATOR = "."
_DEFAULT_WARMUP_ITERATIONS = 5
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_ITERATIONS = 10


def _parse_trigger(trigger: str) -> tuple[str, str, Callable]:
    split_trigger = trigger.split(_SEPARATOR)
    variable_name = split_trigger[0]

    if len(split_trigger) == 1:
        func_name = "info"
        func = None
    elif len(split_trigger) == 2:
        func_name = split_trigger[1]

        if func_name == "dtype":
            func = lambda tensor: tensor.dtype
        elif func_name in ["size()", "shape"]:
            func = lambda tensor: tensor.size()
        elif func_name == "stride()":
            func = lambda tensor: tensor.stride()
        elif func_name.startswith("size"):
            dim = int(func_name[5:][:-1])
            func = lambda tensor: tensor.size(dim)
        elif func_name.startswith("shape"):
            dim = int(func_name[6:][:-1])
            func = lambda tensor: tensor.size(dim)
        elif func_name.startswith("stride"):
            dim = int(func_name[7:][:-1])
            func = lambda tensor: tensor.stride(dim)
        else:
            raise ValueError(f"unexpected triggeer found ({trigger})")

    return variable_name, func_name, func


class AutotunedFunction:
    def __init__(
        self,
        function: Callable,
        configs: list[AutotuneConfig],
        triggers: set[str],
        warmup_iterations: int,
        benchmark_iterations: int,
        functional_triggers: dict[str, Callable] = {},
        reset_to_zero: dict = {},
    ) -> AutotunedFunction:
        assert len(configs) > 0, "no autotune config is passed"

        self.function = function
        self.configs = configs
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        self.signature = inspect.getfullargspec(function)
        self.autotuneable_parameters = set(self.configs[0].get_key_values().keys())

        self._setup_trigger_map(triggers)

        for config in self.configs:
            assert (
                set(config.get_key_values().keys()) == self.autotuneable_parameters
            ), "autotune configs don't match the expected function signature"

        self.functional_triggers = functional_triggers
        self.reset_to_zero = reset_to_zero

        caller_path = Path(inspect.stack()[2].filename).resolve()

        try:
            self.filename = str(caller_path.relative_to(_REPO_ROOT))
        except ValueError:
            self.filename = str(caller_path)

        self.function_hash = f"{self.filename}->{function.__name__}"

        self.function_cache = {}

    @property
    def exposed_signature(self) -> inspect.Signature:
        # the tuneable parameters are chosen internally by the tuner, so callers should never need to pass them
        # and they shouldn't be part of the signature exposed to the outside world (e.g. for custom op schema
        # inference)
        full_signature = inspect.signature(self.function)

        return full_signature.replace(
            parameters=[
                parameter
                for name, parameter in full_signature.parameters.items()
                if name not in self.autotuneable_parameters
            ]
        )

    def __call__(self, *args, **kwargs) -> Any:
        lookup_key = self._get_lookup_key(*args, **kwargs)
        best_config = self.function_cache.get(lookup_key, None)

        if best_config is None:
            # bypass autotune for single config
            if len(self.configs) == 1:
                best_config = self.configs[0]
                best_time = 0
            else:
                best_config, best_time, _ = self._autotune(*args, **kwargs)

            self.function_cache[lookup_key] = best_config

            if _XMA_PRINT_AUTOTUNING:
                print(
                    f"config {best_config} achieved the best time ({best_time:.3f} sec) for {lookup_key} for "
                    f"function {self.function.__name__}"
                )

        return self.function(**self._get_function_arguments(config=best_config, args=args, kwargs=kwargs))

    def _get_function_arguments(self, config: AutotuneConfig, args: list, kwargs: dict) -> dict:
        # copy the best_config first so we can override with args or kwargs
        result = {variable_name: value for variable_name, value in config.get_key_values().items()}

        for i in range(len(args)):
            variable_name = self.signature.args[i]
            result[variable_name] = args[i]

        # accessing kwargs.items() breaks torch.compile in backwards of a custom autograd function
        for variable_name in kwargs:
            result[variable_name] = kwargs.get(variable_name)

        return result

    @torch.compiler.set_stance("force_eager")
    @torch.inference_mode()
    def _autotune(self, *args, **kwargs) -> tuple[AutotuneConfig, float, list[tuple[AutotuneConfig, float]]]:
        best_config = None
        best_time = float("inf")
        timed_configs = []

        for config in self.configs:
            if not config.is_condition_valid(
                **self._get_function_arguments(config=AutotuneConfig({}), args=args, kwargs=kwargs)
            ):
                if _XMA_PRINT_AUTOTUNING:
                    print(f"Skipping config {config} for function {self.function.__name__}")

                continue

            elapsed_time = self._run_benchmark(
                **self._get_function_arguments(config=config, args=args, kwargs=kwargs),
            )

            if _XMA_PRINT_AUTOTUNING:
                print(f"config {config} took {elapsed_time:.3f} sec for function {self.function.__name__}")

            timed_configs.append((config, elapsed_time))

            if elapsed_time < best_time:
                best_config = config
                best_time = elapsed_time

        assert best_config is not None, "no best_config found, check that at least 1 autotune config is valid"

        return best_config, best_time, timed_configs

    def _get_lookup_key(self, *args, **kwargs) -> Any:
        lookup_key = []

        def _maybe_add_key(variable_name: str, value) -> None:
            if variable_name not in self.variable_name_trigger_map:
                return

            triggers = self.variable_name_trigger_map[variable_name]

            if isinstance(value, torch.Tensor):
                for func_name, func in triggers:
                    if func is None:
                        assert len(triggers) == 1
                        func = lambda tensor: (tensor.dtype, tensor.size(), tensor.stride())

                    lookup_key.append(f"{variable_name}.{func_name} = {func(value)}")
            else:
                assert len(triggers) == 1
                func_name, func = triggers[0]
                assert (
                    func is None
                ), f"trigger ({variable_name}) is not a tensor and shouldn't have a functional trigger"

                lookup_key.append(f"{variable_name} = {value}")

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]
            _maybe_add_key(variable_name, value)

        for variable_name in kwargs:
            _maybe_add_key(variable_name, kwargs[variable_name])

        # now run the functional triggers
        if len(self.functional_triggers) > 0:
            kwargs = self._get_function_arguments(config=AutotuneConfig({}), args=args, kwargs=kwargs)

            for variable_name, func in self.functional_triggers.items():
                lookup_key.append(f"{variable_name} = {func(**kwargs)}")

        return str(lookup_key)[1:-1]

    def _run_benchmark(self, **kwargs: dict) -> float:
        Accelerator.synchronize()

        for _ in range(self.warmup_iterations):
            self.function(**kwargs)

        # TODO generalize to device Event
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if len(self.reset_to_zero) > 0:
            elapsed_time = 0

            for _ in range(self.benchmark_iterations):
                start.record()
                self.function(**kwargs)
                end.record()

                Accelerator.synchronize()
                elapsed_time += start.elapsed_time(end)

                for variable_name, function in self.reset_to_zero.items():
                    if function is None or function(**kwargs):
                        variable = kwargs[variable_name]
                        assert isinstance(variable, torch.Tensor)

                        variable.zero_()
        else:
            start.record()
            for _ in range(self.benchmark_iterations):
                self.function(**kwargs)
            end.record()

            Accelerator.synchronize()
            elapsed_time = start.elapsed_time(end)

        return elapsed_time / self.benchmark_iterations

    def _setup_trigger_map(self, triggers: set[str]) -> None:
        assert isinstance(triggers, set), "triggers should be a set"
        self.variable_name_trigger_map = defaultdict(list)

        for trigger in triggers:
            variable_name, func_name, func = _parse_trigger(trigger)
            self.variable_name_trigger_map[variable_name].append((func_name, func))

        # filter to remove all triggers if None, this is useful for Tensor based triggers
        for variable_name in self.variable_name_trigger_map:
            if ("info", None) in self.variable_name_trigger_map[variable_name]:
                self.variable_name_trigger_map[variable_name] = [("info", None)]

            assert (
                variable_name in self.signature.args
            ), f"unexpected variable_name ({variable_name}) found in triggers"

        for variable_name in self.autotuneable_parameters:
            assert variable_name not in self.variable_name_trigger_map, "trigger can't be a tuneable parameter"

    def __repr__(self):
        return f"""AutotunedFunction(
    function_cache = {self.function_cache}
    configs = {self.configs}
    warmup iterations = {self.warmup_iterations}
    benchmark iterations = {self.benchmark_iterations}
    autotuneable parameters = {self.autotuneable_parameters}
    functional triggers = {self.functional_triggers}
    reset to zero = {self.reset_to_zero}
    function hash = {self.function_hash}
)"""


def autotune(
    configs: list[AutotuneConfig],
    triggers: set[str] = set(),
    functional_triggers: dict[str, Callable] = {},
    warmup_iterations: int = _DEFAULT_WARMUP_ITERATIONS,
    benchmark_iterations: int = _BENCHMARK_ITERATIONS,
    reset_to_zero: dict = {},
) -> AutotunedFunction:
    """
    autotuner for any function or kernel

    :param configs: list of configs to autotune over
    :type configs: list[AutotuneConfig]
    :param triggers: change in these parameters will trigger autotuning
    :type triggers: set[str]
    :param functional_triggers: key, function mapping. change in the function outputs will trigger autotuning.
    :type functional_triggers: dict[str, Callable]
    :param warmup_iterations: iterations for warmup. Defaults to 5.
    :type warmup_iterations: int
    :param benchmark_iterations: iterations for benchmarking. Defaults to 10.
    :type benchmark_iterations: int
    :param reset_to_zero: A dictionary mapping tensor argument names to an optional callable condition. The
        specified tensors will be zeroed out after each benchmark iteration if the condition (if provided) returns
        True.
    :type reset_to_zero: dict
    :return: autotuned version of the function
    :rtype: _Autotune
    """

    def inner(function: Callable) -> Callable:
        return AutotunedFunction(
            function=function,
            configs=configs,
            triggers=triggers,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            functional_triggers=functional_triggers,
            reset_to_zero=reset_to_zero,
        )

    return inner
