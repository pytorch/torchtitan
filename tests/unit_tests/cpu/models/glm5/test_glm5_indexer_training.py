# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.glm5 import build_model_config


@pytest.fixture(autouse=True)
def isolated_aux_metrics():
    with patch.dict(AuxLoss.group_acc, clear=True), patch.dict(
        AuxLoss._group_counts, clear=True
    ):
        yield


def build_stage(stage, coefficient=1.0):
    config = build_model_config(
        "glm5_2_debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=16,
        indexer_stage=stage,
        indexer_loss_coeff=coefficient,
    )
    config.layers = config.layers[:4]
    config.index_sources = (0, 1, 2, 2)
    torch.manual_seed(42)
    model = config.build()
    model.init_states()
    return config, model


def backward(model, *, checkpoint=False):
    model.zero_grad(set_to_none=True)
    for module in model.modules():
        if isinstance(module, AuxLoss):
            module.instance_acc.zero_()
    positions = torch.cat((torch.arange(12), torch.arange(4)))
    padding = torch.zeros(16, dtype=torch.bool)
    padding[11] = True
    inputs = torch.arange(16)
    if checkpoint:
        DefaultActivationRematConfig().apply(model)
    output = model(
        inputs,
        positions=positions,
        padding_mask=padding,
        aux_loss_denominators=torch.tensor([15.0]),
    )
    output.float().square().mean().backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients.values())
    metrics = {
        name: module.instance_acc.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, AuxLoss)
    }
    return output.detach().clone(), gradients, metrics


@pytest.mark.parametrize("stage", ["warmup", "sparse"])
def test_stage_gradient_ownership_and_rematerialization(stage):
    _, model = build_stage(stage)
    expected = backward(model)
    assert any(
        ".indexer." in name and value.abs().sum() > 0
        for name, value in expected[1].items()
    )
    backbone_gradients = [name for name in expected[1] if ".indexer." not in name]
    assert bool(backbone_gradients) == (stage == "sparse")
    actual = backward(model, checkpoint=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_coefficient_does_not_leak_lm_gradients_into_indexer():
    _, model = build_stage("sparse", coefficient=0.0)
    _, gradients, _ = backward(model)
    indexer_gradients = [
        value for name, value in gradients.items() if ".indexer." in name
    ]
    assert indexer_gradients
    assert all(torch.count_nonzero(value) == 0 for value in indexer_gradients)
    assert any(
        value.abs().sum() > 0
        for name, value in gradients.items()
        if ".indexer." not in name
    )


def test_warmup_weights_load_strictly_into_sparse_stage():
    _, warmup = build_stage("warmup")
    optimizer = torch.optim.SGD(
        (p for p in warmup.parameters() if p.requires_grad), lr=0.01
    )
    before = {
        name: value.detach().clone() for name, value in warmup.state_dict().items()
    }
    backward(warmup)
    optimizer.step()
    state = warmup.state_dict()
    assert any(
        not torch.equal(before[name], value)
        for name, value in state.items()
        if ".indexer." in name
    )
    for name, value in state.items():
        if ".indexer." not in name:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    _, sparse = build_stage("sparse")
    sparse.load_state_dict(state, strict=True)
    for name, value in sparse.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    backward(sparse)


def test_teacher_matches_head_averaged_dense_probabilities_without_gradients():
    from torchtitan.models.glm5.dsa import Glm5IndexerLoss

    torch.manual_seed(42)
    q = torch.randn(4, 2, 8, requires_grad=True)
    k = torch.randn(4, 2, 8, requires_grad=True)
    indices = torch.arange(4).expand(4, -1).clone()
    valid = indices <= torch.arange(4)[:, None]
    indices.masked_fill_(~valid, -1)
    logits = torch.einsum("qhd,khd->qhk", q, k) * 0.25
    logits = logits.masked_fill(~valid[:, None], -torch.inf)
    lse = logits.logsumexp(-1)
    expected = logits.softmax(-1).mean(1)
    loss = Glm5IndexerLoss.Config(
        coeff=1.0, num_heads=2, softmax_scale=0.25, chunk_size=2
    ).build()
    actual = loss._teacher(q, k, indices, lse, None)
    expected = expected.gather(1, indices.clamp_min(0)).masked_fill(~valid, 0)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_warmup_chunk_boundary_matches_packed_dense_attention_and_gradients():
    from torchtitan.models.glm5.dsa import DsaMetadata, Glm5SparseAttention

    torch.manual_seed(42)
    q = torch.randn(129, 2, 4, requires_grad=True)
    k = torch.randn(129, 2, 4, requires_grad=True)
    v = torch.randn(129, 2, 3, requires_grad=True)
    positions = torch.cat((torch.arange(80), torch.arange(49)))
    padding = torch.zeros(129, dtype=torch.bool)
    padding[[79, 128]] = True
    metadata = DsaMetadata.from_positions(positions, padding)
    from torchtitan.models.glm5.dsa import gather_attn

    attention = Glm5SparseAttention.Config(
        training_stage="warmup", max_context_length=80
    ).build()
    with patch("torchtitan.models.glm5.dsa.gather_attn", wraps=gather_attn) as kernel:
        actual = attention(
            q,
            k,
            v,
            attention_metadata=metadata,
            topk_indices_QS=torch.full((129, 1), -1, dtype=torch.int32),
            scale=0.5,
        )
    assert all(call.args[3].shape[-1] == 80 for call in kernel.call_args_list)
    documents = (positions == 0).cumsum(0)
    allowed = (
        (documents[:, None] == documents[None, :])
        & (positions[:, None] >= positions[None, :])
        & ~padding[:, None]
        & ~padding[None, :]
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=allowed,
        scale=0.5,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    torch.testing.assert_close(actual_grads, expected_grads)


@pytest.mark.parametrize("stage", ["frozen", "warmup", "sparse"])
def test_main_and_mtp_indexer_flops_exclusions_are_additive(stage):
    from torchtitan.models.glm5.dsa import Glm5DsaIndexer

    config = build_model_config(
        "glm5_2_debugmodel_mtp",
        parallelism=ParallelismConfig(),
        seq_len=128,
        indexer_stage=stage,
    )
    with torch.device("meta"):
        model = config.build()
    indexers = [
        module for module in model.modules() if isinstance(module, Glm5DsaIndexer)
    ]
    assert any(
        isinstance(module, Glm5DsaIndexer) for module in model.mtp_layers.modules()
    )
    batch = {"input": torch.ones(128, dtype=torch.int64)}

    def estimate(excluded):
        return config.build_flops_estimator(
            model, seq_len=128, excluded_submodules=excluded
        )(batch)

    full = estimate(())
    deltas = [full - estimate((module,)) for module in indexers]
    assert all(delta > 0 for delta in deltas)
    assert sum(deltas) == full - estimate(tuple(indexers))
