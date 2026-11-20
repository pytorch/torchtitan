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
    loss = output if model.config.indexer_training_stage == "warmup" else output.float().square().mean()
    loss.backward()
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
    if stage == "warmup":
        # Cross several KL chunks within one checkpoint/aux-loss name scope.
        for layer in model.layers.values():
            loss = layer.attention.inner_attention.inner_attention.indexer_loss
            if loss is not None:
                loss.chunk_size = 5
    expected = backward(model)
    assert any(
        ".indexer." in name and value.abs().sum() > 0
        for name, value in expected[1].items()
    )
    backbone_gradients = [name for name in expected[1] if ".indexer." not in name]
    assert bool(backbone_gradients) == (stage == "sparse")
    actual = backward(model, checkpoint=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["frozen", "sparse"])
def test_shared_indices_match_sorted_support_without_consumer_filtering(stage):
    from torchtitan.models.glm5 import dsa

    _, model = build_stage(stage)
    consumed = []

    def capture_indices(module, args, kwargs):
        consumed.append(kwargs["topk_indices_QS"])

    handles = [
        module.register_forward_pre_hook(capture_indices, with_kwargs=True)
        for module in model.modules()
        if isinstance(module, dsa.Glm5SparseAttention)
    ]
    try:
        actual = backward(model)
        assert len(consumed) == 4
        assert consumed[2] is consumed[3]
    finally:
        for handle in handles:
            handle.remove()

    original_forward = dsa.Glm5SparseAttention.forward

    def sort_each_layer(module, *args, **kwargs):
        indices = kwargs["topk_indices_QS"]
        capacity = kwargs["attention_metadata"].positions_T.numel()
        indices = indices.masked_fill(indices < 0, capacity).sort(-1).values
        kwargs["topk_indices_QS"] = indices.masked_fill(indices == capacity, -1)
        return original_forward(module, *args, **kwargs)

    # The old filter sorted valid keys. Removing that sort can change FP rounding.
    with patch.object(dsa.Glm5SparseAttention, "forward", sort_each_layer):
        expected = backward(model)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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


def test_kl_only_warmup_has_no_teacher_activation_backward_or_lm_head():
    from torchtitan.models.glm5.loss import Glm5WarmupLoss
    from torchtitan.components.loss import CrossEntropyLoss

    _, model = build_stage("warmup")
    teacher_outputs = []
    handles = [
        layer.feed_forward.register_forward_hook(
            lambda _module, _inputs, output: teacher_outputs.append(output.requires_grad)
        )
        for layer in model.layers.values()
    ]
    try:
        with patch.object(model.lm_head, "forward", side_effect=AssertionError("KL-only stage must skip LM head")):
            output, gradients, _ = backward(model)
    finally:
        for handle in handles:
            handle.remove()
    assert output.ndim == 0 and torch.isfinite(output)
    assert teacher_outputs and not any(teacher_outputs)
    assert all(".indexer." in name for name in gradients)
    loss = Glm5WarmupLoss.Config(evaluation_loss=CrossEntropyLoss.Config()).build()
    actual, metrics = loss(output, torch.arange(16), torch.tensor(999.0))
    assert actual is output and metrics == {}
    # Evaluation remains teacher CE, not KL and not a new inference interface.
    with torch.no_grad():
        logits = model(torch.arange(16), positions=torch.arange(16))
    labels = torch.arange(16)
    actual, _ = loss(logits, labels)
    expected, _ = CrossEntropyLoss.Config().build()(logits, labels)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("offset", [0, 4])
def test_kl_only_value_and_student_gradients_match_dense_oracle(offset):
    from torchtitan.models.glm5.dsa import DsaMetadata, Glm5IndexerLoss

    torch.manual_seed(61)
    q = torch.randn(4, 2, 6, requires_grad=True)
    pool = torch.randn(8, 6, requires_grad=True)
    features = (torch.randn(4, 2, 3, requires_grad=True),
                torch.randn(8, 3, requires_grad=True),
                torch.randn(4, 2, requires_grad=True))
    positions = torch.arange(8) % 4
    padding = torch.tensor([False, False, False, True] * 2)
    metadata = DsaMetadata.from_positions(positions, padding)
    metadata = DsaMetadata(metadata.positions_T, metadata.documents_T,
                           metadata.valid_T, metadata.query_indices_Q[offset:offset + 4])
    coefficient, denominator = 0.7, torch.tensor(6.0)
    module = Glm5IndexerLoss.Config(coeff=coefficient, num_heads=2,
                                    softmax_scale=0.25, chunk_size=3).build()
    actual = module.warmup_objective(q, pool, features, metadata, denominator)
    rows = metadata.query_indices_Q
    valid = ((metadata.documents_T[rows, None] == metadata.documents_T[None, :])
             & (metadata.positions_T[rows, None] >= metadata.positions_T[None, :])
             & metadata.valid_T[rows, None] & metadata.valid_T[None, :])
    teacher = (torch.einsum("qhd,kd->qhk", q.detach(), pool.detach()) * 0.25)
    target = teacher.masked_fill(~valid[:, None], -torch.inf).softmax(-1).nan_to_num().mean(1)
    target /= target.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    qi, ki, weights = features
    scores = (torch.einsum("qnh,kh->qnk", qi, ki).relu() * weights[:, :, None]).sum(1) / 6**0.5
    scores = scores.masked_fill(~valid, -torch.inf).masked_fill(~valid.any(-1, keepdim=True), 0)
    log_prob = scores.log_softmax(-1).masked_fill(~valid, 0)
    expected = (torch.special.xlogy(target, target) - target * log_prob).masked_fill(~valid, 0).sum() * coefficient / denominator
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, features, retain_graph=True),
                               torch.autograd.grad(expected, features))
    assert torch.autograd.grad(actual, (q, pool), allow_unused=True) == (None, None)
    torch.testing.assert_close(module.instance_acc, expected.detach() / coefficient)


def test_kl_only_mtp_default_remat_and_sparse_checkpoint_handoff():
    from torchtitan.models.glm5.dsa import DsaMetadata
    from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence

    config = build_model_config("glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(),
                                seq_len=16, indexer_stage="warmup")
    config.layers = config.layers[:4]
    config.index_sources = (0, 1, 2, 2)
    model = config.build()
    model.init_states()
    tokens, positions = torch.arange(16), torch.arange(16)
    shifted, valid = roll_mtp_sequence(tokens, shift=1, positions=positions,
                                      padding_mask=None, fill_value=0, return_valid_mask=True)
    kwargs = dict(positions=positions, mtp_input_valid_masks=(valid,),
                  mtp_attention_masks=(DsaMetadata.from_positions(positions, ~valid),),
                  aux_loss_denominators=torch.tensor([16.0, 15.0]))
    expected = model((tokens, shifted), **kwargs)
    expected.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert any("mtp_layers.0.attention.indexer." in name for name in gradients)
    model.zero_grad(set_to_none=True)
    DefaultActivationRematConfig().apply(model)
    actual = model((tokens, shifted), **kwargs)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    sparse_config = build_model_config("glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(),
                                       seq_len=16, indexer_stage="sparse")
    sparse_config.layers = sparse_config.layers[:4]
    sparse_config.index_sources = config.index_sources
    sparse_config.build().load_state_dict(model.state_dict(), strict=True)


def test_kl_only_warmup_rejects_pipeline_before_build():
    with pytest.raises(ValueError, match="warmup.*pipeline"):
        build_model_config("glm5_2_debugmodel", parallelism=ParallelismConfig(pipeline_parallel_degree=2),
                           seq_len=16, indexer_stage="warmup")


@pytest.mark.parametrize("query_offset", [0, 4])
def test_teacher_matches_head_averaged_dense_probabilities_without_gradients(query_offset):
    from torchtitan.models.glm5.dsa import Glm5IndexerLoss

    torch.manual_seed(42)
    q = torch.randn(4, 2, 8, requires_grad=True)
    k = torch.randn(8, 8, requires_grad=True)
    indices = (torch.arange(4) + query_offset).expand(4, -1).clone()
    valid = indices <= (torch.arange(4) + query_offset)[:, None]
    padding = torch.tensor([False, False, False, True])
    valid &= ~padding[:, None]
    indices.masked_fill_(~valid, -1)
    logits = torch.einsum("qhd,kd->qhk", q, k) * 0.25
    allowed = (
        (torch.arange(8)[None, :] >= query_offset)
        & (torch.arange(8)[None, :] <= (torch.arange(4) + query_offset)[:, None])
        & ~padding[:, None]
    )
    logits = logits.masked_fill(~allowed[:, None], -torch.inf)
    lse = logits.logsumexp(-1)
    expected = logits.softmax(-1).nan_to_num().mean(1)
    loss = Glm5IndexerLoss.Config(
        coeff=1.0, num_heads=2, softmax_scale=0.25, chunk_size=2
    ).build()
    # Global pool IDs and no document offsets are also the CP teacher contract.
    actual = loss._teacher(q, k, indices, lse, None, padding_mask_T=padding)
    expected = expected.gather(1, indices.clamp_min(0)).masked_fill(~valid, 0)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_warmup_chunk_boundary_matches_packed_dense_attention_and_gradients():
    from torchtitan.models.glm5.dsa import DsaMetadata, Glm5SparseAttention

    torch.manual_seed(42)
    q = torch.randn(129, 2, 4, requires_grad=True)
    k = torch.randn(129, 4, requires_grad=True)
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
        k.unsqueeze(0).expand(2, -1, -1),
        k.unsqueeze(0).expand(2, -1, -1),
        attn_mask=allowed,
        scale=0.5,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k))
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k))
    torch.testing.assert_close(actual_grads, expected_grads)


@pytest.mark.parametrize("stage", ["frozen", "warmup", "sparse"])
def test_flops_sequence_growth_counts_sparse_attention_and_unique_indexers(stage):
    from torchtitan.models.glm5.model import Glm5Attention

    config = build_model_config(
        "glm5_2_debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=128,
        indexer_stage=stage,
    )
    with torch.device("meta"):
        model = config.build()
    # Both lengths exceed top-k. Parameter projections have the same cost;
    # only full-document indexer scores grow outside the warmup stage.
    short, long = 64, 128
    tokens = 128
    batch = {"input": torch.ones(tokens, dtype=torch.int64)}
    expected_per_token = 0
    for layer in config.layers:
        attention = layer.attention
        assert isinstance(attention, Glm5Attention.Config)
        if stage == "warmup":
            expected_per_token += 2 * attention.n_heads * (
                2 * attention.kv_lora_rank
                + attention.qk_rope_head_dim
            )
        indexer = attention.indexer
        if indexer is not None:
            if stage != "warmup":
                expected_per_token += 2 * indexer.n_heads * (indexer.head_dim + 1)
            if stage == "warmup":
                expected_per_token += 6 * indexer.n_heads * (indexer.head_dim + 1)
                expected_per_token += 4 * attention.n_heads * (
                    attention.kv_lora_rank + attention.qk_rope_head_dim
                )
    estimates = [
        config.build_flops_estimator(model, seq_len=length)(batch)
        for length in (short, long)
    ]
    assert estimates[1] - estimates[0] == (
        expected_per_token * (long - short) * tokens
    )


@pytest.mark.parametrize("stage", ["frozen", "warmup", "sparse"])
def test_flops_match_explicit_dense_layer_operation_budget(stage):
    config = build_model_config(
        "glm5_2_debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=128,
        indexer_stage=stage,
    )
    config.layers = config.layers[:1]
    config.index_sources = (0,)
    with torch.device("meta"):
        model = config.build()
    attention = config.layers[0].attention
    indexer = attention.indexer
    assert indexer is not None
    # One dense layer: no expert weighting or shared parameters. Embedding is
    # a lookup, so exclude it. Count parameters directly, not via FLOPs helpers.
    backbone_params = sum(p.numel() for name, p in model.named_parameters()
                          if not name.startswith("tok_embeddings.") and ".indexer." not in name)
    indexer_params = sum(p.numel() for p in model.layers["0"].attention.indexer.parameters())
    keys = 128 if stage == "warmup" else attention.index_topk
    parameter_ops = (2 if stage == "warmup" else 6) * backbone_params
    if stage == "warmup":
        parameter_ops -= 2 * sum(p.numel() for module in (model.lm_head, model.norm) for p in module.parameters())
    parameter_ops += (2 if stage == "frozen" else 6) * indexer_params
    attention_ops = (2 if stage == "warmup" else 6) * attention.n_heads * (2 * attention.kv_lora_rank + attention.qk_rope_head_dim) * keys
    selection_ops = 0 if stage == "warmup" else 2 * indexer.n_heads * (indexer.head_dim + 1) * 128
    kl_ops = 0 if stage == "frozen" else (
        6 * indexer.n_heads * (indexer.head_dim + 1) * keys
        + (4 if stage == "warmup" else 2) * attention.n_heads * (attention.kv_lora_rank + attention.qk_rope_head_dim) * keys
    )
    expected = 128 * (parameter_ops + attention_ops + selection_ops + kl_ops)
    assert config.build_flops_estimator(model, seq_len=128)(
        {"input": torch.ones(128, dtype=torch.int64)}
    ) == expected


def test_warmup_never_runs_selector_but_still_trains_indexer():
    from torchtitan.models.glm5.dsa import DSAIndexerTopK

    _, model = build_stage("warmup")
    leaf = model.layers["0"].attention.inner_attention.inner_attention
    leaf.indexer_loss.chunk_size = 5
    with (
        patch.object(DSAIndexerTopK, "forward", side_effect=AssertionError("warmup must not rank keys")),
        patch.object(leaf, "_attend", wraps=leaf._attend) as attend,
        patch.object(model.layers["0"].attention.wq_a, "forward",
                     wraps=model.layers["0"].attention.wq_a.forward) as projection,
    ):
        backward(model)
    assert projection.call_count == 1
    assert [call.args[0].shape[0] for call in attend.call_args_list] == [5, 5, 5, 1]
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if ".indexer." in name)
    assert all(p.grad is None for name, p in model.named_parameters() if ".indexer." not in name)


@pytest.mark.parametrize("mutation", ["model", "indexer", "attention", "loss", "consumer"])
def test_config_overrides_cannot_silently_desynchronize_stages(mutation):
    config = build_model_config("glm5_2_debugmodel", parallelism=ParallelismConfig(), seq_len=128, indexer_stage="sparse")
    producer = config.layers[0].attention
    if mutation == "model":
        config.indexer_training_stage = "warmup"
    elif mutation == "indexer":
        producer.indexer.training_stage = "frozen"
    elif mutation == "attention":
        producer.inner_attention.inner_attention.training_stage = "frozen"
    elif mutation == "loss":
        producer.inner_attention.inner_attention.indexer_loss = None
    else:
        config.layers[3].attention.inner_attention.inner_attention.indexer_loss = producer.inner_attention.inner_attention.indexer_loss
    with pytest.raises(ValueError, match="disagree"), torch.device("meta"):
        config.build()
