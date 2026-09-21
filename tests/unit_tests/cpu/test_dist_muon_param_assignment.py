# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.distributed.flex_shard import ComputeLayout, Owned
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


def _model_part(layer_id, *, wrapped=False):
    block = nn.ModuleDict(
        {
            "feed_forward": FeedForward.Config(
                w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
                w2=Linear.Config(in_features=8, out_features=4),
            ).build(),
            "norm": nn.LayerNorm(4),
        }
    )
    if wrapped:
        block = checkpoint_wrapper(block)
    return nn.ModuleDict({"layers": nn.ModuleDict({str(layer_id): block})})


def _group(pattern, optimizer_name="DistMuon"):
    return ParamGroupConfig(
        pattern=pattern,
        optimizer_name=optimizer_name,
        optimizer_kwargs={"lr": 1e-3},
    )


def _config(*groups, mapped_names):
    return OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=list(groups),
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "compute_sharding_by_fqn": {
                    name: ComputeLayout(shardings_by_mesh_axis={"dp_shard": Owned()})
                    for name in mapped_names
                }
            }
        },
    )


@pytest.fixture
def muon_factory():
    # Exercise real grouping and container construction; only replace the
    # distributed optimizer factory so these assignment tests need no mesh.
    with patch(
        "torchtitan.components.optimizer.optimizer.build_dist_muon",
        side_effect=lambda groups, **kwargs: torch.optim.SGD(groups, lr=1e-3),
    ) as factory:
        yield factory


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_dist_muon_rejects_fused_weight_regex_fallback(
    optimizer_name, wrapped, muon_factory
):
    model = _model_part(0, wrapped=wrapped)
    fqn = "layers.0.feed_forward.w13.weight"
    config = _config(
        _group(r"feed_forward\.w[123]\.weight$"),
        _group(r".*", optimizer_name),
        mapped_names=[fqn, "layers.0.feed_forward.w2.weight"],
    )

    with pytest.raises(ValueError) as error:
        config.build(model_parts=[model])

    assert str(error.value) == (
        f"{fqn} has a DistMuon compute layout but is assigned to {optimizer_name}"
    )
    muon_factory.assert_not_called()


def test_dist_muon_rejects_first_match_shadowing(muon_factory):
    fqn = "layers.0.feed_forward.w13.weight"
    config = _config(
        _group(r"\.w13\.weight$", "AdamW"),
        _group(r"feed_forward\.(?:w13|w2)\.weight$"),
        _group(r".*", "AdamW"),
        mapped_names=[fqn],
    )

    with pytest.raises(ValueError) as error:
        config.build(model_parts=[_model_part(0)])

    assert str(error.value) == (
        f"{fqn} has a DistMuon compute layout but is assigned to AdamW"
    )
    muon_factory.assert_not_called()


def test_dist_muon_rejects_layout_without_muon_group(muon_factory):
    fqn = "layers.0.feed_forward.w13.weight"
    config = _config(_group(r".*", "AdamW"), mapped_names=[fqn])

    with pytest.raises(ValueError) as error:
        config.build(model_parts=[_model_part(0)])

    assert str(error.value) == (
        f"{fqn} has a DistMuon compute layout but is assigned to AdamW"
    )
    muon_factory.assert_not_called()


def test_dist_muon_rejects_mapped_parameter_without_optimizer(muon_factory):
    fqn = "layers.0.feed_forward.w13.weight"
    config = _config(_group(r"\.norm\.", "AdamW"), mapped_names=[fqn])

    with pytest.raises(ValueError) as error:
        config.build(model_parts=[_model_part(0)])

    assert str(error.value) == (
        f"{fqn} has a DistMuon compute layout but is not assigned to an optimizer"
    )
    muon_factory.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
def test_dist_muon_accepts_local_assignments_with_absent_and_frozen_entries(
    wrapped, muon_factory
):
    model_parts = [_model_part(layer_id, wrapped=wrapped) for layer_id in (0, 2)]
    for model in model_parts:
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.requires_grad_(False)
    config = _config(
        _group(r"feed_forward\.(?:w13|w2)\.weight$"),
        _group(r".*", "AdamW"),
        mapped_names=[
            f"layers.{layer_id}.{suffix}"
            for layer_id in range(3)
            for suffix in (
                "feed_forward.w13.weight",
                "feed_forward.w2.weight",
                "norm.weight",
            )
        ],
    )

    container = config.build(model_parts=model_parts)

    assert muon_factory.call_count == 2
    assert len(container.optimizers) == 4
    for layer_id, call in zip((0, 2), muon_factory.call_args_list):
        assert {name for group in call.args[0] for name in group["param_names"]} == {
            f"layers.{layer_id}.feed_forward.w13.weight",
            f"layers.{layer_id}.feed_forward.w2.weight",
        }
    assert (
        sum(
            len(group["params"])
            for optimizer in container.optimizers
            for group in optimizer.param_groups
        )
        == 6
    )
