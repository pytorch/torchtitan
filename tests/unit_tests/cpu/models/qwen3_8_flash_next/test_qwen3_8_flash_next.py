# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import math
from dataclasses import replace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.qwen3_8_flash_next import attention as qsa, build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import (
    QSAIndexerLoss,
    QSAIndexMetadata,
)
from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextBlock
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)


@pytest.fixture
def config():
    cfg = build_model_config(
        "debugmodel",
        seq_len=64,
        parallelism=ParallelismConfig(),
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    cfg.vocab_size = cfg.tok_embeddings.num_embeddings = cfg.lm_head.out_features = 64
    cfg.layers[1].ple.ple_embedding.vocab_size = 64
    cfg.layers[1].ple.ple_embedding.eos_token_id = 63
    return cfg


def test_released_architecture_and_checkpoint_schema():
    cfg = build_model_config(
        "125B-A6B",
        seq_len=16,
        parallelism=ParallelismConfig(),
    )
    with torch.device("meta"):
        model = cfg.build()
    hf = Qwen38FlashNextStateDictAdapter(cfg, None).to_hf(model.state_dict())
    # Captured independently from the decoder/vision safetensors headers at:
    # https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540
    expected_shapes = {
        "model.language_model.embed_tokens.weight": [248320, 2560],
        "lm_head.weight": [248320, 2560],
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [10240, 2560],
        "model.language_model.layers.3.self_attn.q_proj.weight": [12288, 2560],
        "model.language_model.layers.3.self_attn.k_proj.weight": [512, 2560],
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": [
            2500012,
            160,
        ],
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight": [
            2500012,
            160,
        ],
        "model.visual.patch_embed.proj.weight": [1152, 3, 2, 16, 16],
    }
    shapes = {name: list(tensor.shape) for name, tensor in hf.items()}
    for name, shape in expected_shapes.items():
        assert shapes[name] == shape, name
    assert len(shapes) == 1627
    assert sum(".ngram_embedding.shard_" in name for name in shapes) == 128
    # Digests of the full key list and of every key's shape. On a mismatch, diff
    # the adapter's keys and shapes against the released safetensors headers.
    assert (
        hashlib.sha256("\n".join(sorted(shapes)).encode()).hexdigest()
        == "56b8050fd51b5e8258b278d8de1db63840655ca1662f01487f5d910455572a11"
    ), "HF key names changed"
    canonical = json.dumps(shapes, sort_keys=True, separators=(",", ":"))
    assert (
        hashlib.sha256(canonical.encode()).hexdigest()
        == "c453af89687fed9659910ed9b990ab4544f7e7d796bbfb7b43dcb6d4ba96c8c4"
    ), "HF tensor shapes changed"


@pytest.mark.parametrize("layer_id", [0, 3])
def test_split_projection_preserves_head_order_and_parameter_gradients(
    config, layer_id
):
    torch.manual_seed(42)
    attention = config.layers[layer_id].attention
    linear_config = attention.in_proj if layer_id == 0 else attention.wq
    linear = linear_config.build()
    num_groups = linear_config.num_groups
    projection_rows = linear_config.projection.out_features // num_groups
    gate_rows = linear_config.gates.out_features // num_groups
    weight = torch.randn(
        num_groups, projection_rows + gate_rows, linear_config.in_features
    )
    with torch.no_grad():
        linear.projection.weight.copy_(weight[:, :projection_rows].flatten(0, 1))
        linear.gates.weight.copy_(weight[:, projection_rows:].flatten(0, 1))
    reference_weight = weight.flatten(0, 1).requires_grad_()
    x = torch.randn(31, linear_config.in_features, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = linear(x)
    expected = F.linear(reference_x, reference_weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    weight_grad = torch.cat(
        (
            linear.projection.weight.grad.unflatten(0, (num_groups, projection_rows)),
            linear.gates.weight.grad.unflatten(0, (num_groups, gate_rows)),
        ),
        1,
    ).flatten(0, 1)
    torch.testing.assert_close(weight_grad, reference_weight.grad, atol=0, rtol=0)


def test_ngram_hashes_reset_at_eos_and_packed_document_boundaries(config):
    embedding = config.layers[1].ple.ple_embedding.build()
    ids = torch.tensor([7, 8, 63, 9, 10, 11, 12, 13])
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    actual = embedding.hash_ids(ids, document_positions_T=positions)
    expected = []
    history = []
    mask = (1 << 64) - 1
    for token, position in zip(ids.tolist(), positions.tolist()):
        if position == 0:
            history = []
        context = [token, *(history[::-1][:2])]
        context += [63] * (3 - len(context))
        row = []
        for ngram in (2, 3):
            hashed = 0
            for i in range(ngram):
                hashed ^= (context[i] * int(embedding.layer_multipliers[i])) & mask
            if hashed >= 1 << 63:
                hashed -= 1 << 64
            for head in range((ngram - 2) * 2, (ngram - 1) * 2):
                row.append(
                    hashed % int(embedding.ngram_heads_vocab_sizes[head])
                    + int(embedding.ngram_heads_offsets[head])
                )
        expected.append(row)
        history = [] if token == 63 else history + [token]
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
    # EOS clears hash history even when the enclosing document keeps its positions.
    after_eos = embedding.hash_ids(ids[3:5], document_positions_T=torch.arange(2))
    torch.testing.assert_close(actual[3:5], after_eos, rtol=0, atol=0)
    # The oracle reuses the module's multipliers; pin them to the HF reference,
    # here and at the released vocabulary size.
    assert embedding.layer_multipliers.tolist() == [
        112381549946653559,
        107512210695146523,
        55314113489879299,
    ]
    released = replace(config.layers[1].ple.ple_embedding, vocab_size=248320).build()
    assert released.layer_multipliers.tolist() == [
        23703573157769,
        20109073645365,
        8052911324071,
    ]


def test_selective_remat_matches_no_remat_for_ple_and_gated_residuals(config):
    """Saved projection regions feed bare ops; selective remat must keep them."""
    layer = config.layers[1]

    class _Block(Module):
        def __init__(self):
            super().__init__()
            self.ple = layer.ple.build()
            self.attn_hyper_connection = layer.attn_hyper_connection.build()

        def forward(self, x_TC, ids_T, positions_T):
            x_TC = x_TC + self.ple(
                x_TC, input_ids_T=ids_T, document_positions_T=positions_T
            )
            mixed_TD, residual_TC, write_TS = self.attn_hyper_connection(x_TC)
            return Qwen38FlashNextBlock.add_residuals(mixed_TD, residual_TC, write_TS)

    class _Model(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleDict({"0": _Block()})

        def forward(self, *args):
            return self.layers["0"](*args)

    torch.manual_seed(42)
    reference = _Model()
    for module in reference.modules():
        if hasattr(module, "init_states"):
            module.init_states()
    x = torch.randn(22, 1024)
    ids = torch.randint(0, 63, (22,))
    positions = torch.arange(11).repeat(2)

    def gradients(activation_remat):
        model = _Model()
        model.load_state_dict(reference.state_dict())
        model.train()
        if activation_remat is not None:
            activation_remat.apply(model)
        x_TC = x.clone().requires_grad_()
        model(x_TC, ids, positions).square().sum().backward()
        return [x_TC.grad] + [p.grad for p in model.parameters()]

    expected = gradients(None)
    for activation_remat in (
        FullActivationRematConfig(),
        DefaultActivationRematConfig(),
    ):
        for actual, want in zip(gradients(activation_remat), expected, strict=True):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_ple_dilated_convolution_and_gradients_respect_documents(config):
    torch.manual_seed(42)
    ple = config.layers[1].ple.build()
    ple.init_states()
    with torch.no_grad():
        ple.conv1d.weight.uniform_(-0.2, 0.2)
    x = torch.randn(22, 1024, requires_grad=True)
    ids = torch.randint(0, 63, (22,))
    positions = torch.arange(11).repeat(2)
    padding = torch.zeros(22, dtype=torch.bool)
    padding[9:11] = True
    captured = []
    handle = ple.norm_conv.register_forward_hook(
        lambda module, args, output: captured.append(
            (args[0].detach(), output.detach())
        )
    )
    actual = ple(
        x, input_ids_T=ids, document_positions_T=positions, padding_mask_T=padding
    )
    handle.remove()
    # Batched Conv1d independently checks tap order and dilation within each document.
    with torch.no_grad():
        gated, normed = (tensor * ~padding[:, None] for tensor in captured[0])
        conv = (
            F.conv1d(
                F.pad(normed.unflatten(0, (2, 11)).transpose(1, 2), (9, 0)),
                ple.conv1d.weight,
                groups=1024,
                dilation=3,
            )
            .transpose(1, 2)
            .flatten(0, 1)
        )
    torch.testing.assert_close(actual, gated + F.silu(conv), rtol=2e-5, atol=2e-5)
    # Changing the first document must leave the second document's output
    # unchanged. Both runs have the same shapes and kernels, so this is exact
    # (comparing against separate per-document runs would also compare CPU
    # GEMMs across row counts, which round differently on some backends).
    perturbed_x = x.detach().clone()
    perturbed_x[:11] = torch.randn(11, 1024)
    perturbed_ids = ids.clone()
    perturbed_ids[:11] = torch.randint(0, 63, (11,))
    perturbed = ple(
        perturbed_x,
        input_ids_T=perturbed_ids,
        document_positions_T=positions,
        padding_mask_T=padding,
    )
    torch.testing.assert_close(perturbed[11:], actual[11:], rtol=0, atol=0)
    # Backward is isolated too: the second document sends no gradient to the
    # first document's inputs.
    (grad_x,) = torch.autograd.grad(actual[11:].square().sum(), x)
    assert not grad_x[:11].any()
    assert grad_x[11:].any()


@pytest.mark.parametrize("windowed", [False, True])
def test_qsa_selection_matches_document_local_block_oracle(config, windowed):
    torch.manual_seed(42)
    indexer_config = config.layers[3].attention.indexer
    if windowed:
        # 16-token chunks with 32-token documents score W = 12 of G = 15 blocks.
        indexer_config = replace(
            indexer_config,
            chunk_size=16,
            rope=replace(indexer_config.rope, max_context_length=32),
        )
    indexer = indexer_config.build()
    indexer.init_states()
    x = torch.randn(63, 256)
    positions = torch.cat((torch.arange(31), torch.arange(32)))
    padding = torch.zeros(63, dtype=torch.bool)
    padding[5] = padding[61:] = True
    metadata = QSAIndexMetadata.from_inputs(
        positions, padding_mask_T=padding, compress_ratio=4
    )
    selected, scores = indexer(x, positions=positions, metadata=metadata)
    projection = indexer.index_qk_proj(x)
    queries = indexer._rotate(
        indexer.q_layernorm(projection[:, :256].unflatten(-1, (4, 64))), positions
    )
    pruned_queries = 0
    for query in range(63):
        begin = 0 if query < 31 else 31
        visible = (
            [token for token in range(begin, query + 1) if not padding[token]]
            if not padding[query]
            else []
        )
        complete = len(visible) // 4
        blocks = torch.tensor(visible[: complete * 4], dtype=torch.long).reshape(
            complete, 4
        )
        pooled = projection[blocks, 256:].float().mean(1)
        keys = indexer._rotate(
            indexer.k_layernorm(pooled).unsqueeze(1), positions[blocks[:, 0]]
        ).squeeze(1)
        oracle_scores = (queries[query].float() @ keys.float().T).relu().sum(0) / 8
        winners = sorted(
            range(complete),
            key=lambda block: (-float(oracle_scores[block].detach()), block),
        )[:4]
        pruned_queries += complete > 4
        expected = [*blocks[winners].flatten().tolist(), *visible[complete * 4 :]]
        actual = selected[query][selected[query] >= 0].tolist()
        assert set(actual) == set(expected)
        torch.testing.assert_close(
            scores[query][scores[query].isfinite()].sort().values,
            oracle_scores[winners].sort().values,
            rtol=1e-5,
            atol=1e-5,
        )
        assert all(token <= query for token in actual)
    assert pruned_queries > 0


@pytest.mark.parametrize("all_padding", [False, True])
def test_qsa_output_and_backward_against_dense_selected_attention(config, all_padding):
    torch.manual_seed(42)
    indexer = config.layers[3].attention.indexer.build()
    indexer.init_states()
    positions = torch.arange(23)
    padding = torch.full((23,), all_padding)
    indices, scores = indexer(
        torch.randn(23, 256),
        positions=positions,
        metadata=QSAIndexMetadata.from_inputs(
            positions, padding_mask_T=padding, compress_ratio=indexer.compress_ratio
        ),
    )
    inner = config.layers[3].attention.inner_attention.build().eval()
    q, k, v = [
        torch.randn(23, heads, 64, dtype=torch.float64, requires_grad=True)
        for heads in (4, 2, 2)
    ]
    actual = inner(q, k, v, selected_indices_TJ=indices, scores_TB=scores, scale=1 / 8)
    mask = torch.zeros(23, 24, dtype=torch.bool).scatter(
        1, indices.long().masked_fill(indices < 0, 23), True
    )[:, :23]
    expected = F.scaled_dot_product_attention(
        q.permute(1, 0, 2)[None],
        k.permute(1, 0, 2)[None],
        v.permute(1, 0, 2)[None],
        attn_mask=mask,
        scale=1 / 8,
        enable_gqa=True,
    )[0].transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (q, k, v), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


def test_qsa_block_kl_and_gradients():
    scores = torch.tensor(
        [[0.0, math.log(2), math.log(4)], [-torch.inf] * 3], requires_grad=True
    )
    teacher = torch.tensor([[0.25, 0.5, 0.25], [0.0, 0.0, 0.0]], requires_grad=True)
    actual = QSAIndexerLoss.block_kl(scores, teacher)
    expected = sum(
        p * math.log(p / q) for p, q in zip((0.25, 0.5, 0.25), (1 / 7, 2 / 7, 4 / 7))
    )
    torch.testing.assert_close(actual, torch.tensor(expected))
    actual.backward()
    torch.testing.assert_close(
        scores.grad[0], torch.tensor([1 / 7 - 0.25, 2 / 7 - 0.5, 4 / 7 - 0.25])
    )
    assert not scores.grad[1].any() and teacher.grad is None


@pytest.mark.parametrize(
    "input_kind", ["complete", "short_documents", "padding", "frozen"]
)
def test_qsa_auxiliary_gradient_only_updates_the_indexer(config, input_kind):
    torch.manual_seed(42)
    if input_kind == "frozen":
        config.layers[3].attention.inner_attention.indexer_loss = None
    attention = config.layers[3].attention.build().train()
    attention.init_states()
    x = torch.randn(31, 256, requires_grad=True)
    positions = torch.arange(31)
    if input_kind == "short_documents":
        positions = positions.remainder(3)
    padding = torch.full((31,), input_kind == "padding")
    captured_scores = []
    handle = attention.indexer.register_forward_hook(
        lambda module, inputs, output: captured_scores.append(output[1])
    )
    attention.configure_remat_regions(("inner_attention",))
    forward = remat.checkpoint()(attention)
    metadata = QSAIndexMetadata.from_inputs(
        positions,
        padding_mask_T=padding,
        compress_ratio=attention.indexer.compress_ratio,
    )
    output = forward(x, metadata, positions, aux_loss_denominator=torch.tensor(31.0))
    handle.remove()
    # A zero LM carrier isolates the injected auxiliary objective.
    (output.sum() * 0).backward()
    assert x.grad is not None and not x.grad.any()
    for parameter in attention.indexer.parameters():
        if input_kind == "frozen":
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            if input_kind != "complete":
                assert not parameter.grad.any()
    if input_kind == "complete":
        assert attention.indexer.index_qk_proj.weight.grad.norm() > 0
    if input_kind == "frozen":
        assert not captured_scores[0].requires_grad
    for module in (attention.wq, attention.wk, attention.wv, attention.wo):
        for parameter in module.parameters():
            assert parameter.grad is None or not parameter.grad.any()


def test_gated_norm_matches_hf_rounding(config):
    torch.manual_seed(42)
    norm = config.layers[0].attention.norm.build().bfloat16()
    norm.init_states()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.2, 1.8, 128))
    x = torch.randn(5, 4, 128, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_gate = gate.detach().clone().requires_grad_()
    reference_weight = norm.weight.detach().clone().requires_grad_()
    normalized = F.rms_norm(reference_x.float(), (128,), eps=norm.eps)
    # HF rounds the unweighted RMSNorm before applying the learned scale.
    expected = (
        normalized.to(x.dtype) * reference_weight * reference_gate.float().sigmoid()
    ).to(x.dtype)
    actual = norm(input=x, gate=gate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        (actual * upstream).sum(), (x, gate, norm.weight)
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (reference_x, reference_gate, reference_weight)
    )
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_adapter_roundtrip_and_numeric_shard_order(config):
    # Part 10 must follow part 9, even when the checkpoint's keys arrive reversed.
    config.layers[1].ple.ple_embedding.split_ngram_parts = 16
    model = config.build()
    model.init_states()
    adapter = Qwen38FlashNextStateDictAdapter(config, None)
    state = model.state_dict()
    hf = adapter.to_hf(state)
    restored = adapter.from_hf(dict(reversed(list(hf.items()))))
    assert restored.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0, msg=key)
    config.build().load_state_dict(restored, strict=True)
    shard = next(key for key in hf if ".shard_0." in key)
    with_mtp = {**hf, "mtp.draft_head.weight": torch.empty(1)}
    without_mtp = adapter.from_hf(with_mtp)
    assert with_mtp.keys() == hf.keys() | {"mtp.draft_head.weight"}
    assert without_mtp.keys() == state.keys()
    incomplete = dict(hf)
    del incomplete[shard]
    with pytest.raises(ValueError, match="requires all"):
        adapter.from_hf(incomplete)


def test_qsa_teacher_sums_tp_head_shards_and_scales_indexer_gradient():
    """TP=2 sums both head shards before max pooling and halves the KL gradient.

    The teacher must match the full-head teacher, since pooling each shard
    first changes the target. The indexer scores are TP-replicated, so each
    rank contributes half of the KL gradient while the loss value is unchanged.
    """
    torch.manual_seed(42)
    T, R, B, H, H_kv, K = 37, 4, 3, 4, 2, 8
    indices = torch.full((T, B * R + R - 1), -1, dtype=torch.long)
    for i in range(T):
        complete = min((i + 1) // R, B)
        indices[i, : complete * R] = torch.arange(complete * R)
        tail = (i + 1) % R
        indices[i, B * R : B * R + tail] = torch.arange(i + 1 - tail, i + 1)
    indices[::11] = -1
    query, key, value = (torch.randn(T, n, K) for n in (H, H_kv, H_kv))
    scale = K**-0.5
    inner = (
        qsa.QSAInnerAttention.Config(
            compress_ratio=R, indexer_loss=qsa.QSAIndexerLoss.Config(coeff=1.0)
        )
        .build()
        .train()
    )
    with torch.no_grad():
        expected_teacher = inner._teacher(query, key, indices, B, scale=scale)

    shards = [(slice(0, H // 2), slice(0, 1)), (slice(H // 2, H), slice(1, 2))]
    remote_teacher = []

    def tp(all_reduce):
        return (
            mock.patch.object(qsa, "spmd_mesh_size", return_value=2),
            mock.patch.object(qsa, "spmd_mesh_group", return_value=object()),
            mock.patch.object(qsa, "all_reduce", side_effect=all_reduce),
            mock.patch.object(qsa, "wait_tensor", side_effect=lambda t: t),
        )

    remote_heads, remote_kv = shards[1]
    patches = tp(lambda t, op, group: remote_teacher.append(t) or t)
    with patches[0], patches[1], patches[2], patches[3], torch.no_grad():
        inner._teacher(
            query[:, remote_heads], key[:, remote_kv], indices, B, scale=scale
        )

    local_heads, local_kv = shards[0]
    scores = torch.randn(T, B).masked_fill(indices[:, : B * R : R] < 0, -torch.inf)
    scores.requires_grad_()
    captured = []
    original_inject = inner.indexer_loss.inject
    patches = tp(lambda t, op, group: t + remote_teacher[0])
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        mock.patch.object(
            inner.indexer_loss,
            "inject",
            side_effect=lambda raw_sum, **kwargs: captured.append(raw_sum)
            or original_inject(raw_sum, **kwargs),
        ),
    ):
        inner(
            query[:, local_heads],
            key[:, local_kv],
            value[:, local_kv],
            selected_indices_TJ=indices,
            scores_TB=scores,
            scale=scale,
            aux_loss_denominator=torch.tensor(float(T)),
        )
        with torch.no_grad():
            tp_teacher = inner._teacher(
                query[:, local_heads], key[:, local_kv], indices, B, scale=scale
            )

    torch.testing.assert_close(tp_teacher, expected_teacher, rtol=1e-6, atol=1e-7)
    oracle_scores = scores.detach().clone().requires_grad_()
    expected_raw_sum = qsa.QSAIndexerLoss.block_kl(oracle_scores, tp_teacher)
    torch.testing.assert_close(captured[0], expected_raw_sum, atol=0, rtol=0)
    (gradient,) = torch.autograd.grad(captured[0], scores)
    (expected_gradient,) = torch.autograd.grad(expected_raw_sum / 2, oracle_scores)
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
