# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-scoped activation retention preserves the global checkpoint policy."""

from copy import deepcopy

import pytest
import torch
import torch_remat as remat

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.linear import Linear


@pytest.fixture(autouse=True)
def deterministic_numerics():
    """Exercise real finite computations with reproducible CPU inputs."""
    torch.manual_seed(42)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class _Block(Module):
    def __init__(self):
        super().__init__()
        self.inner_compute = Linear.Config(in_features=32, out_features=32).build()
        self.inner_forwards = 0
        self.activation_forwards = 0

    def _inner(self, x_TD: torch.Tensor) -> torch.Tensor:
        self.inner_forwards += 1
        return torch.nn.functional.linear(x_TD, self.inner_compute.weight)

    def _activation(self, x_TD: torch.Tensor) -> torch.Tensor:
        self.activation_forwards += 1
        return x_TD.tanh()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        output_TD = remat.region(
            self._inner,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(x_TD)
        output_TD = remat.region(
            self._activation,
            self.remat_region_name("activation"),
            recompute=self.remat_should_recompute("activation"),
        )(output_TD)
        remat.recompute_needs_tensor(output_TD)
        return output_TD.sin()


class _Model(Module):
    def __init__(self, layer_ids: tuple[str, ...] = ("0", "1", "2")):
        super().__init__()
        self.layers = ModuleDict({layer_id: _Block() for layer_id in layer_ids})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        for block in self.layers.values():
            x_TD = block(x_TD)
        return x_TD


def _run(model: _Model, input_TD: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return exact loss, input gradient, and parameter gradients."""
    model.zero_grad(set_to_none=True)
    x_TD = input_TD.detach().clone().requires_grad_(True)
    loss = model(x_TD).square().sum()
    loss.backward()
    return (
        loss.detach(),
        x_TD.grad.detach().clone(),
        *(parameter.grad.detach().clone() for parameter in model.parameters()),
    )


def test_only_selected_block_avoids_recomputation() -> None:
    """One retained region skips replay only in its selected layer, exactly."""
    reference = _Model()
    actual = deepcopy(reference)
    ActivationRematConfig(
        save_regions=[], block_save_regions={"layers.1": ["inner_attention"]}
    ).apply(actual)
    input_TD = torch.randn(8, 32)
    for actual_tensor, expected in zip(
        _run(actual, input_TD), _run(reference, input_TD), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected, rtol=0, atol=0)
    assert [block.inner_forwards for block in actual.layers.values()] == [2, 1, 2]


def test_block_globs_preserve_pipeline_layer_fqns() -> None:
    """A PP subset retains original IDs rather than renumbering its local layers."""
    model = _Model(("81", "82", "91"))
    ActivationRematConfig(
        save_regions=[], block_save_regions={"layers.8?": ["inner_attention"]}
    ).apply(model)
    _run(model, torch.randn(8, 32))
    assert [block.inner_forwards for block in model.layers.values()] == [1, 1, 2]


def test_global_recompute_overrides_block_retention() -> None:
    """An explicit recompute pattern still wins over additional block saves."""
    model = _Model()
    ActivationRematConfig(
        save_regions=[],
        recompute_regions=["inner_attention"],
        block_save_regions={"layers.*": ["inner_attention"]},
    ).apply(model)
    _run(model, torch.randn(8, 32))
    assert [block.inner_forwards for block in model.layers.values()] == [2, 2, 2]


@pytest.mark.parametrize(
    "policy", [FullActivationRematConfig, DefaultActivationRematConfig]
)
def test_fixed_checkpoint_presets_reject_block_overrides(policy) -> None:
    """Fixed Full and default remat presets cannot silently acquire custom saves."""
    with pytest.raises(ValueError, match="ActivationRematConfig"):
        policy(block_save_regions={"layers.1": ["inner_attention"]})


def test_block_rules_are_additive_with_global_saves() -> None:
    """Overlapping block globs never replace global or earlier matching saves."""
    reference = _Model()
    actual = deepcopy(reference)
    ActivationRematConfig(
        save_regions=["activation"],
        block_save_regions={
            "layers.*": ["inner_attention"],
            "layers.1": [],
        },
    ).apply(actual)
    input_TD = torch.randn(8, 32)
    for actual_tensor, expected in zip(
        _run(actual, input_TD), _run(reference, input_TD), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected, rtol=0, atol=0)
    assert [block.inner_forwards for block in actual.layers.values()] == [1, 1, 1]
    assert [block.activation_forwards for block in actual.layers.values()] == [1, 1, 1]


def test_block_retention_respects_saved_tensor_hooks() -> None:
    """Selected region outputs use the existing saved-tensor hook ownership."""
    reference = _Model()
    actual = deepcopy(reference)
    saved_outputs = {}

    def hooks(module, block_fqn):
        def pack(tensor):
            if (
                remat.current_saved_tensor_info().kind
                is remat.SavedTensorKind.SAVE_OUTPUT
            ):
                saved_outputs[block_fqn] = (
                    saved_outputs.get(block_fqn, 0) + tensor.numel()
                )
            return tensor.detach().clone()

        return pack, lambda payload: payload

    ActivationRematConfig(
        save_regions=[], block_save_regions={"layers.1": ["inner_attention"]}
    ).apply(actual, saved_tensors_hooks=hooks)
    input_TD = torch.randn(8, 32)
    for actual_tensor, expected in zip(
        _run(actual, input_TD), _run(reference, input_TD), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected, rtol=0, atol=0)
    assert saved_outputs == {"layers.1": 8 * 32}
