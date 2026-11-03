# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import math

import pytest
import torch
import torch_remat as remat
from torch import nn

from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.observability import tensor_logger
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

D, H, T = 16, 32, 8
_LINEAR_INIT = {
    "weight": functools.partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


def _build_logger(model: nn.Module, log_freq: int) -> tensor_logger.TensorLogger:
    return tensor_logger.TensorLogger.Config(log_freq=log_freq).build(
        model_parts=[model],
        device=torch.device("cpu"),
        accelerator_graph_enabled=False,
        pp_schedule=None,
    )


class _Block(Module):
    """A llama3-like block whose logged tensors are outputs of remat regions."""

    def __init__(self):
        super().__init__()
        self.first_ffn = make_ffn_config(
            dim=D,
            hidden_dim=H,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_LINEAR_INIT,
            enable_sp=False,
        ).build()
        self.feed_forward = make_ffn_config(
            dim=D,
            hidden_dim=H,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_LINEAR_INIT,
            enable_sp=False,
        ).build()
        tensor_logger.register_fwd_bwd(self, "first_out", "ffn_out")

    def forward(self, x_TD):
        first_out_TD = self.first_ffn(x_TD)
        tensor_logger.log_fwd_bwd(self, first_out=first_out_TD)
        remat.recompute_needs_tensor(first_out_TD)
        h_TD = x_TD + first_out_TD
        ffn_out_TD = self.feed_forward(h_TD)
        tensor_logger.log_fwd_bwd(self, ffn_out=ffn_out_TD)
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h_TD, ffn_out_TD)


class _Model(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": _Block(), "1": _Block()})

    def forward(self, x_TD):
        for layer in self.layers.values():
            x_TD = layer(x_TD)
        return x_TD


def test_remat_records_each_observation_once():
    """torch_remat replays each block's forward in backward; each tensor counts once.

    Without the `remat.is_recomputing()` check, Full remat doubles every forward
    row and Default remat raises when replay reads a saved region's output. Both
    orders matter: gradient accumulation (backward right after its forward) and
    pipeline order (all forwards, then all backwards).
    """
    torch.manual_seed(0)
    initial_model = _Model().double()
    initial_model.init_states()
    initial_state = initial_model.state_dict()
    microbatches = [torch.randn(T, D, dtype=torch.float64) for _ in range(2)]

    def run(activation_remat, *, forwards_first):
        model = _Model().double()
        model.load_state_dict(initial_state)
        if activation_remat is not None:
            activation_remat.apply(model)
        logger = _build_logger(model, log_freq=1)
        try:
            with logger.maybe_record(step=1):
                if forwards_first:
                    outputs = [model(x_TD).square().sum() for x_TD in microbatches]
                    for output in outputs:
                        output.backward()
                else:
                    for x_TD in microbatches:
                        model(x_TD).square().sum().backward()
            metrics = logger.collect()
        finally:
            logger.close()
        return metrics

    for forwards_first in (False, True):
        ref_metrics = run(None, forwards_first=forwards_first)
        assert (
            ref_metrics["tensor_logger/layers.0.first_out.fwd.observation_count"] == 2
        )
        assert ref_metrics["tensor_logger/layers.1.ffn_out.bwd.observation_count"] == 2
        for activation_remat in (
            FullActivationRematConfig(),
            DefaultActivationRematConfig(),
        ):
            metrics = run(activation_remat, forwards_first=forwards_first)
            assert metrics == ref_metrics, (type(activation_remat), forwards_first)


class _Logged(nn.Module):
    def __init__(self):
        super().__init__()
        tensor_logger.register_fwd(self, "scores", "with_nonfinite")
        tensor_logger.register_fwd_bwd(self, "hidden")


def test_collect_follows_cadence_and_derives_stats():
    module = _Logged()
    logger = _build_logger(module, log_freq=2)
    scores = torch.tensor([0.0, 1.0, -2.0, 3.0])
    try:
        with logger.maybe_record(step=1):
            tensor_logger.log_fwd(module, scores=scores)
        assert logger.collect() == {}

        # Outside maybe_record nothing is recorded, e.g. a validation forward.
        tensor_logger.log_fwd(module, scores=scores)
        with logger.maybe_record(step=2):
            tensor_logger.log_fwd(
                module,
                scores=scores,
                with_nonfinite=torch.tensor([math.inf, math.nan, 2.0, 0.0]),
            )
            x = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
            # A leaf would collect one more gradient hook per call.
            with pytest.raises(ValueError, match="leaf tensor"):
                tensor_logger.log_fwd_bwd(module, hidden=x)
            hidden = x * 3
            tensor_logger.log_fwd_bwd(module, hidden=hidden)
            hidden.sum().backward()
        metrics = logger.collect()
    finally:
        logger.close()

    def fields(
        prefix, numel, nonfinite_count, zero_frac, abs_mean, rms, kurtosis, abs_max
    ):
        values = {
            "numel": numel,
            "observation_count": 1,
            "nonfinite_count": nonfinite_count,
            "zero_frac": zero_frac,
            "abs_mean": abs_mean,
            "rms": rms,
            "kurtosis": kurtosis,
            "abs_max": abs_max,
        }
        return {f"tensor_logger/{prefix}.{name}": v for name, v in values.items()}

    assert metrics == {
        # [0, 1, -2, 3]: abs_mean 6/4, rms sqrt(14/4), kurtosis 98/4/3.5**2 - 3.
        **fields("scores.fwd", 4, 0, 0.25, 1.5, math.sqrt(3.5), -1.0, 3.0),
        # Finite values [2, 0]: inf and nan count only in numel and nonfinite_count.
        **fields("with_nonfinite.fwd", 4, 2, 0.5, 1.0, math.sqrt(2.0), -1.0, 2.0),
        # [3, -3, 6]: abs_mean 12/3, rms sqrt(54/3), kurtosis 1458/3/18**2 - 3.
        **fields("hidden.fwd", 3, 0, 0.0, 4.0, math.sqrt(18.0), -1.5, 6.0),
        # The gradient of hidden.sum() is all ones.
        **fields("hidden.bwd", 3, 0, 0.0, 1.0, 1.0, -2.0, 1.0),
    }


_PP2 = {"parallelism.pipeline_parallel_degree": 2}


@pytest.mark.parametrize(
    "overrides, match",
    [
        # Each case would otherwise publish wrong statistics without an error.
        ({"metrics.log_freq": 2}, "multiple of metrics.log_freq"),
        (
            {"debug.deterministic": True, "sdc_replayer": SDCReplayer.Config()},
            "sdc_replayer",
        ),
        (
            {**_PP2, "parallelism.pipeline_parallel_schedule": "ZBVZeroBubble"},
            "one full backward",
        ),
        (
            {**_PP2, "parallelism.pipeline_parallel_schedule_csv": "schedule.csv"},
            "one full backward",
        ),
    ],
)
def test_unsupported_configs_raise(overrides, match):
    config = llama3_debugmodel()
    config.tensor_logger = tensor_logger.TensorLogger.Config(log_freq=3)
    for path, value in overrides.items():
        *parents, name = path.split(".")
        setattr(functools.reduce(getattr, parents, config), name, value)
    with pytest.raises(ValueError, match=match):
        # replace() runs Trainer.Config.__post_init__ again on the changed fields.
        dataclasses.replace(config)
