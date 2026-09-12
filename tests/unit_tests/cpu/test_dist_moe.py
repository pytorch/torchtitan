# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from torchtitan.components.dist_moe import DistMoeConverter
from torchtitan.components.dist_moe.backend import _DistMoeRuntime, DistMoeRoutedExperts
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import RoutedExperts


def _runtime(prefetch: Any) -> _DistMoeRuntime:
    return _DistMoeRuntime(
        config=cast(Any, object()),
        group=cast(Any, object()),
        prefetch=prefetch,
    )


def test_dist_moe_runtime_closes_prefetch_after_initialization_failure():
    prefetch = Mock()
    runtime = _runtime(prefetch)

    with (
        patch(
            "torchtitan.components.dist_moe.backend.create_context",
            side_effect=RuntimeError("context creation failed"),
        ),
        pytest.raises(RuntimeError, match="context creation failed"),
    ):
        runtime.initialize(torch.device("cuda"))

    prefetch.close.assert_called_once_with()
    assert runtime.prefetch is None


def test_dist_moe_runtime_close_releases_pending_prefetch():
    prefetch = Mock()
    runtime = _runtime(prefetch)

    runtime.close()
    runtime.close()

    prefetch.close.assert_called_once_with()
    assert runtime.prefetch is None


def test_dist_moe_converter_rejects_specialized_routed_experts():
    """Conversion must not silently discard subclass-specific semantics."""

    @dataclass(kw_only=True, slots=True)
    class SpecializedConfig(RoutedExperts.Config):
        """Routed-expert config carrying behavior unknown to the converter."""

        extra_policy: bool = True

    stock = make_routed_experts_config(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init={},
        comm_backend="standard",
    )
    specialized = SpecializedConfig(
        inner_experts=stock.inner_experts,
        token_dispatcher=stock.token_dispatcher,
    )

    with pytest.raises(TypeError, match="specialized routed-experts config"):
        DistMoeConverter(DistMoeConverter.Config()).convert(specialized)


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8"])
def test_dist_moe_forwards_expert_output_postprocess(dtype):
    """The adapter preserves the common route-output callback contract."""
    stock = make_routed_experts_config(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init={},
        comm_backend="standard",
    )
    config = DistMoeConverter(
        DistMoeConverter.Config(backend=DistMoeBackendConfig(dtype=dtype))
    ).convert(stock)
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module._runtime = _runtime(None)
    module._runtime.context = cast(Any, object())
    postprocess = Mock(side_effect=lambda value: value)

    with (
        patch(
            "torchtitan.components.dist_moe.backend._dynamic_prepared_weight",
            side_effect=lambda weight, **_kwargs: weight,
        ),
        patch(
            "torchtitan.components.dist_moe.backend.run_dist_moe",
            return_value=torch.empty(2, 32),
        ) as run,
    ):
        module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
            expert_output_postprocess=postprocess,
        )

    configured = run.call_args.kwargs["options"].experts_output_postprocess
    assert configured is not None
    value = torch.randn(2, 32)
    assert configured.fn(value) is value
    postprocess.assert_called_once_with(value)
    assert not configured.includes_scale_and_sum
