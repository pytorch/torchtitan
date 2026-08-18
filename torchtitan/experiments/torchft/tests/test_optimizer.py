# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.experiments.torchft.optimizer import (
    _OptimizerAdapter,
    TorchFTOptimizersContainer,
)


class _RecordingOptimizer:
    def __init__(self) -> None:
        self.calls = []

    def step(self, *args, **kwargs) -> None:
        self.calls.append(("step", args, kwargs))

    def zero_grad(self, *args, **kwargs) -> None:
        self.calls.append(("zero_grad", args, kwargs))


class _Wrapper:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.calls = []

    def step(self, *args, **kwargs) -> None:
        self.calls.append("step")
        return self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self.calls.append("zero_grad")
        return self.optimizer.zero_grad(*args, **kwargs)


def test_ft_optimizer_dispatches_to_inner_adapter_without_recursion():
    optimizers = [_RecordingOptimizer(), _RecordingOptimizer()]
    container = object.__new__(TorchFTOptimizersContainer)
    container.optimizers = optimizers
    container.cache_state_dict = {"cached": object()}
    container._inner_optimizer = _OptimizerAdapter(container)
    container._ft_optimizer = _Wrapper(container._inner_optimizer)
    container._use_ft_optimizer = True

    container.step()
    container.zero_grad(set_to_none=True)

    assert container._ft_optimizer.calls == ["step", "zero_grad"]
    for optimizer in optimizers:
        assert optimizer.calls == [
            ("step", (), {}),
            ("zero_grad", (), {"set_to_none": True}),
        ]
    assert container._inner_optimizer.state_dict() is container.cache_state_dict
    assert container._use_ft_optimizer is True


def test_ft_optimizer_error_does_not_change_dispatch_policy():
    class _FailingWrapper:
        def step(self):
            raise RuntimeError("step failed")

    container = object.__new__(TorchFTOptimizersContainer)
    container._ft_optimizer = _FailingWrapper()
    container._use_ft_optimizer = True

    with pytest.raises(RuntimeError, match="step failed"):
        container.step()

    assert container._use_ft_optimizer is True


def test_single_replica_dispatches_directly_to_inner_optimizer():
    optimizer = _RecordingOptimizer()
    container = object.__new__(TorchFTOptimizersContainer)
    container.optimizers = [optimizer]
    container._inner_optimizer = _OptimizerAdapter(container)
    container._ft_optimizer = mock_ft_optimizer = _Wrapper(container._inner_optimizer)
    container._use_ft_optimizer = False

    container.step()
    container.zero_grad()

    assert mock_ft_optimizer.calls == []
    assert [call[0] for call in optimizer.calls] == ["step", "zero_grad"]
