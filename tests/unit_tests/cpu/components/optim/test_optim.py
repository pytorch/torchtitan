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

from torchtitan.components.optim import GradientClipper, Optim
from torchtitan.tools.utils import device_type


@dataclass
class _OptimizerConfig:
    pass


@dataclass
class _OptimizersConfig:
    optimizers: list[_OptimizerConfig]
    build: MagicMock


class _TestGradientClipper(GradientClipper):
    @dataclass(kw_only=True, slots=True)
    class Config(GradientClipper.Config):
        grad_norm: float = 2.0

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls = []

    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float | int,
        *,
        parallelism_context,
    ) -> torch.Tensor:
        self.calls.append((parameters, max_norm, parallelism_context))
        return torch.tensor(self.config.grad_norm)


def test_default_gradient_clipper_uses_distributed_clip_grad_norm() -> None:
    parameters = [torch.nn.Parameter(torch.tensor(1.0))]
    parallelism_context = MagicMock()
    pp_mesh = MagicMock()
    parallelism_context.get_optional_mesh.return_value = pp_mesh
    gradient_clipper_config = Optim.Config().gradient_clipper
    assert isinstance(gradient_clipper_config, GradientClipper.Config)
    gradient_clipper = gradient_clipper_config.build()
    expected_grad_norm = torch.tensor(2.0)

    with patch(
        "torchtitan.components.optim.optim.dist_utils.clip_grad_norm_",
        return_value=expected_grad_norm,
    ) as clip_grad_norm:
        grad_norm = gradient_clipper.clip_grad_norm_(
            parameters,
            1.0,
            parallelism_context=parallelism_context,
        )

    clip_grad_norm.assert_called_once_with(
        parameters,
        1.0,
        foreach=True,
        pp_mesh=pp_mesh,
    )
    assert grad_norm is expected_grad_norm


def test_optim_config_rejects_negative_max_norm() -> None:
    with pytest.raises(ValueError, match="max_norm"):
        Optim.Config(max_norm=-1)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_optim_builds_owned_components(enable_cuda_graph: bool) -> None:
    model = MagicMock(spec=torch.nn.Module)
    parameter = MagicMock(spec=torch.nn.Parameter)
    parameter.device = torch.device(device_type if enable_cuda_graph else "cpu")
    model.parameters.return_value = [parameter]
    optimizers = MagicMock()
    lr_schedulers = MagicMock()
    ema = MagicMock()
    gradient_clipper = MagicMock()
    optimizer_config = _OptimizersConfig(
        optimizers=[_OptimizerConfig()],
        build=MagicMock(return_value=optimizers),
    )
    gradient_clipper_config = SimpleNamespace(
        build=MagicMock(return_value=gradient_clipper)
    )
    config = SimpleNamespace(
        optimizer=optimizer_config,
        lr_scheduler=SimpleNamespace(build=MagicMock(return_value=lr_schedulers)),
        ema=SimpleNamespace(build=MagicMock(return_value=ema)),
        max_norm=1.0,
        gradient_clipper=gradient_clipper_config,
        enable_cuda_graph=enable_cuda_graph,
    )
    wrapped_update = MagicMock()

    with patch(
        "torchtitan.components.optim.optim.wrap_with_accelerator_graph",
        return_value=wrapped_update,
    ) as wrap, patch(
        "torchtitan.components.optim.optim.accelerator_graphs_supported",
        return_value=enable_cuda_graph,
    ):
        optim = Optim(
            config,
            model_parts=[model],
            parallelism_context=MagicMock(),
            training_steps=10,
            pp_has_last_stage=True,
        )

    optimizer_config.build.assert_called_once_with(
        model_parts=[model],
        enable_cuda_graph=enable_cuda_graph,
    )
    config.lr_scheduler.build.assert_called_once_with(
        optimizers=optimizers,
        training_steps=10,
    )
    config.ema.build.assert_called_once_with(model_parts=[model])
    gradient_clipper_config.build.assert_called_once_with()
    assert optim.gradient_clipper is gradient_clipper
    assert optim._is_first_step_call
    if enable_cuda_graph:
        wrap.assert_called_once_with(
            optim._update,
            num_warmup_iterations=2,
            name="optimizer",
            annotation_config=None,
        )
        assert optim._run_update is wrapped_update
    else:
        wrap.assert_not_called()
        assert optim._run_update == optim._update


@pytest.mark.parametrize("runtime_supported", [False, True])
def test_optim_cuda_graph_falls_back_for_unsupported_device(
    runtime_supported: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizers = MagicMock()
    optimizer_build = MagicMock(return_value=optimizers)
    gradient_clipper_config = SimpleNamespace(build=MagicMock())
    config = SimpleNamespace(
        optimizer=_OptimizersConfig(
            optimizers=[_OptimizerConfig()],
            build=optimizer_build,
        ),
        lr_scheduler=SimpleNamespace(build=MagicMock()),
        ema=None,
        max_norm=1.0,
        gradient_clipper=gradient_clipper_config,
        enable_cuda_graph=True,
    )

    with patch(
        "torchtitan.components.optim.optim.accelerator_graphs_supported",
        return_value=runtime_supported,
    ), patch("torchtitan.components.optim.optim.wrap_with_accelerator_graph") as wrap:
        optim = Optim(
            config,
            model_parts=[model],
            parallelism_context=MagicMock(),
            training_steps=10,
            pp_has_last_stage=True,
        )

    optimizer_build.assert_called_once_with(
        model_parts=[model],
        enable_cuda_graph=False,
    )
    gradient_clipper_config.build.assert_called_once_with()
    wrap.assert_not_called()
    assert optim._run_update == optim._update
    assert "Optim graph capture is disabled" in caplog.text


def test_optim_builds_configured_grad_clip() -> None:
    model = torch.nn.Linear(2, 2)
    config = SimpleNamespace(
        optimizer=_OptimizersConfig(
            optimizers=[_OptimizerConfig()],
            build=MagicMock(return_value=MagicMock()),
        ),
        lr_scheduler=SimpleNamespace(build=MagicMock()),
        ema=None,
        max_norm=1.0,
        gradient_clipper=_TestGradientClipper.Config(grad_norm=3.0),
        enable_cuda_graph=False,
    )

    optim = Optim(
        config,
        model_parts=[model],
        parallelism_context=MagicMock(),
        training_steps=10,
        pp_has_last_stage=True,
    )

    assert isinstance(optim.gradient_clipper, _TestGradientClipper)
    assert optim.gradient_clipper.config.grad_norm == 3.0


def test_optim_update_uses_configured_grad_clip() -> None:
    parameters = [torch.nn.Parameter(torch.tensor(1.0))]
    parallelism_context = MagicMock()
    parallelism_context.pp_enabled = False
    parallelism_context.get_optional_mesh.return_value = None
    gradient_clipper = _TestGradientClipper(_TestGradientClipper.Config())
    optim = object.__new__(Optim)
    optim.config = SimpleNamespace(max_norm=1.0)
    optim.parallelism_context = parallelism_context
    optim.pp_has_last_stage = True
    optim.parameters = parameters
    optim.optimizers = MagicMock()
    optim.gradient_clipper = gradient_clipper

    with patch(
        "torchtitan.components.optim.optim.dist_utils.clip_grad_norm_"
    ) as eager_grad_clip:
        grad_norm = optim._update(torch.tensor(1.0))

    eager_grad_clip.assert_not_called()
    assert gradient_clipper.calls == [(parameters, 1.0, parallelism_context)]
    torch.testing.assert_close(grad_norm, torch.tensor(2.0))
    optim.optimizers.step.assert_called_once_with()


@pytest.mark.parametrize("device_initialized", [False, True])
def test_optim_step_keeps_eager_state_outside_update(device_initialized: bool) -> None:
    events = []
    optim = object.__new__(Optim)
    optim._is_first_step_call = True
    optim._run_update = lambda loss: (
        events.append("update"),
        torch.testing.assert_close(loss, torch.tensor(1.0)),
        torch.tensor(2.0),
    )[2]
    optim.lr_schedulers = SimpleNamespace(step=lambda: events.append("lr_scheduler"))
    optim.ema = SimpleNamespace(step=lambda step: events.append(f"ema_{step}"))

    device_module = MagicMock()
    device_module.is_initialized.return_value = device_initialized
    device_module.synchronize.side_effect = lambda: events.append("synchronize")
    device_module.empty_cache.side_effect = lambda: events.append("empty_cache")

    with (
        patch(
            "torchtitan.components.optim.optim.torch.accelerator.is_available",
            return_value=True,
        ),
        patch("torchtitan.components.optim.optim.device_module", device_module),
    ):
        grad_norm = optim.step(torch.tensor(1.0), current_step=3)
        optim.step(torch.tensor(1.0), current_step=4)

    device_module.empty_cache.assert_called_once_with()

    torch.testing.assert_close(grad_norm, torch.tensor(2.0))
    assert events == (["synchronize"] if device_initialized else []) + [
        "empty_cache",
        "update",
        "lr_scheduler",
        "ema_3",
        "update",
        "lr_scheduler",
        "ema_4",
    ]
