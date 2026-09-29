# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.components.optimization import AdamW, Optimization, OptimizersContainer


@dataclass
class _OptimizerConfig:
    enable_cuda_graph: bool


@dataclass
class _OptimizersConfig:
    optimizers: list[_OptimizerConfig]
    build: MagicMock


def test_optimization_config_rejects_negative_max_norm() -> None:
    with pytest.raises(ValueError, match="max_norm"):
        Optimization.Config(max_norm=-1)


def test_optimization_config_rejects_mismatched_cuda_graph_setting() -> None:
    with pytest.raises(ValueError, match="must match every optimizer"):
        Optimization.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*")]
            ),
            enable_cuda_graph=True,
        )


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_optimization_builds_owned_components(enable_cuda_graph: bool) -> None:
    model = MagicMock(spec=torch.nn.Module)
    parameter = MagicMock(spec=torch.nn.Parameter)
    parameter.device = torch.device("cuda" if enable_cuda_graph else "cpu")
    model.parameters.return_value = [parameter]
    optimizers = MagicMock()
    lr_schedulers = MagicMock()
    ema = MagicMock()
    optimizer_config = _OptimizersConfig(
        optimizers=[_OptimizerConfig(enable_cuda_graph=enable_cuda_graph)],
        build=MagicMock(return_value=optimizers),
    )
    config = SimpleNamespace(
        optimizer=optimizer_config,
        lr_scheduler=SimpleNamespace(build=MagicMock(return_value=lr_schedulers)),
        ema=SimpleNamespace(build=MagicMock(return_value=ema)),
        max_norm=1.0,
        enable_cuda_graph=enable_cuda_graph,
    )
    wrapped_update = MagicMock()

    with patch(
        "torchtitan.components.optimization.optimization.wrap_with_cuda_graph",
        return_value=wrapped_update,
    ) as wrap, patch(
        "torchtitan.components.optimization.optimization.cuda_graphs_supported",
        return_value=enable_cuda_graph,
    ):
        optimization = Optimization(
            config,
            model_parts=[model],
            parallelism_context=MagicMock(),
            training_steps=10,
            pp_has_last_stage=True,
        )

    optimizer_config.build.assert_called_once_with(model_parts=[model])
    config.lr_scheduler.build.assert_called_once_with(
        optimizers=optimizers,
        training_steps=10,
    )
    config.ema.build.assert_called_once_with(model_parts=[model])
    if enable_cuda_graph:
        wrap.assert_called_once_with(
            optimization._update,
            num_warmup_iterations=2,
        )
        assert optimization._run_update is wrapped_update
    else:
        wrap.assert_not_called()
        assert optimization._run_update == optimization._update


@pytest.mark.parametrize("runtime_supported", [False, True])
def test_optimization_cuda_graph_falls_back_for_unsupported_device(
    runtime_supported: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizers = MagicMock()
    optimizer_build = MagicMock(return_value=optimizers)
    config = SimpleNamespace(
        optimizer=_OptimizersConfig(
            optimizers=[_OptimizerConfig(enable_cuda_graph=True)],
            build=optimizer_build,
        ),
        lr_scheduler=SimpleNamespace(build=MagicMock()),
        ema=None,
        max_norm=1.0,
        enable_cuda_graph=True,
    )

    with patch(
        "torchtitan.components.optimization.optimization.cuda_graphs_supported",
        return_value=runtime_supported,
    ), patch(
        "torchtitan.components.optimization.optimization.wrap_with_cuda_graph"
    ) as wrap:
        optimization = Optimization(
            config,
            model_parts=[model],
            parallelism_context=MagicMock(),
            training_steps=10,
            pp_has_last_stage=True,
        )

    optimizer_build.assert_called_once_with(model_parts=[model])
    wrap.assert_not_called()
    assert optimization._run_update == optimization._update
    assert "Optimization CUDA graph is disabled" in caplog.text


def test_optimization_step_keeps_eager_state_outside_update() -> None:
    events = []
    optimization = object.__new__(Optimization)
    optimization._run_update = lambda loss: (
        events.append("update"),
        torch.testing.assert_close(loss, torch.tensor(1.0)),
        torch.tensor(2.0),
    )[2]
    optimization.lr_schedulers = SimpleNamespace(
        step=lambda: events.append("lr_scheduler")
    )
    optimization.ema = SimpleNamespace(step=lambda step: events.append(f"ema_{step}"))

    grad_norm = optimization.step(torch.tensor(1.0), current_step=3)

    torch.testing.assert_close(grad_norm, torch.tensor(2.0))
    assert events == ["update", "lr_scheduler", "ema_3"]
