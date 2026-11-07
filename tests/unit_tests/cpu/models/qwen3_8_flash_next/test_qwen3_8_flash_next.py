# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import (
    QSAIndexerLoss,
    QSAIndexMetadata,
)
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
    cfg.vision_encoder = None
    cfg.layers[1].ple.ple_embedding.vocab_size = 64
    cfg.layers[1].ple.ple_embedding.eos_token_id = 63
    return cfg


@pytest.mark.parametrize("expert_parallel_degree", [1, 2])
def test_released_architecture_and_checkpoint_schema(expert_parallel_degree):
    cfg = build_model_config(
        "125B-A6B",
        seq_len=16,
        parallelism=ParallelismConfig(expert_parallel_degree=expert_parallel_degree),
    )
    with torch.device("meta"):
        model = cfg.build()
    hf = Qwen38FlashNextStateDictAdapter(cfg, None).to_hf(model.state_dict())
    fixture = (
        Path(__file__).resolve().parents[4]
        / "assets/qwen3_8_flash_next/checkpoint_schema.json"
    )
    expected = json.loads(fixture.read_text())
    shapes = {name: list(tensor.shape) for name, tensor in hf.items()}
    assert len(shapes) == expected["num_tensors"]
    assert (
        sum(".ngram_embedding.shard_" in name for name in shapes)
        == expected["num_ngram_shards"]
    )
    assert (
        hashlib.sha256("\n".join(sorted(shapes)).encode()).hexdigest()
        == expected["names_sha256"]
    )
    # Both digests come from independently captured, pinned HF checkpoint headers.
    canonical = json.dumps(shapes, sort_keys=True, separators=(",", ":"))
    assert (
        hashlib.sha256(canonical.encode()).hexdigest()
        == expected["names_and_shapes_sha256"]
    )
    for name, shape in expected["representative_shapes"].items():
        assert shapes[name] == shape


def test_unsupported_attention_configs_are_explicit(config):
    with pytest.raises(ValueError, match="context parallelism"):
        build_model_config(
            "debugmodel", parallelism=ParallelismConfig(context_parallel_degree=2)
        )
    for flavor in ("debugmodel", "debugmodel_mm", "125B-A6B"):
        with pytest.raises(ValueError, match="must divide QSA KV heads"):
            build_model_config(
                flavor,
                seq_len=16,
                parallelism=ParallelismConfig(
                    tensor_parallel_degree=4, expert_parallel_degree=4
                ),
            )
    for indexer_ratio, attention_ratio in ((8, 4), (4, 8), (8, 8)):
        attention = copy.deepcopy(config.layers[3].attention)
        attention.indexer.compress_ratio = indexer_ratio
        attention.inner_attention.compress_ratio = attention_ratio
        with pytest.raises(ValueError, match="compress_ratio=4"):
            attention.build()


def test_gated_residual_forward_and_gradients(config):
    torch.manual_seed(42)
    mixer = config.layers[0].attn_hyper_connection.build()
    mixer.init_states()
    x = torch.randn(7, 1024, requires_grad=True)
    actual, residual, write = mixer(x)
    streams = x.unflatten(-1, (4, 256))
    normed = torch.stack(
        [
            F.rms_norm(streams[:, i].float(), (256,), eps=1e-6)
            * (1 + mixer.hc_norm.weight.unflatten(0, (4, 256))[i])
            for i in range(4)
        ],
        1,
    )
    down = F.linear(normed.flatten(1), mixer.input_mix_weight_down.weight)
    read = (
        F.linear(F.silu(down / 4), mixer.input_mix_weight_up.weight)
        .sigmoid()
        .unflatten(-1, (4, 256))
    )
    expected = (read * normed).mean(1)
    expected_write = (
        2
        * (F.linear(normed.flatten(1), mixer.block_inject_weight.weight) / 4).sigmoid()
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(write, expected_write, atol=1e-6, rtol=1e-6)
    assert residual is x
    parameters = (x, *mixer.parameters())
    actual_grads = torch.autograd.grad(
        actual.square().sum() + write.square().sum(), parameters, retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum() + expected_write.square().sum(), parameters
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


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
    assert embedding.layer_multipliers.dtype == torch.int64
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer.Config().build(
        tokenizer_path=str(Path(__file__).resolve().parents[4] / "assets/tokenizer")
    )
    recipe_config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    recipe_embedding = recipe_config.layers[1].ple.ple_embedding.build()
    recipe_ids = torch.tensor(tokenizer.encode("hello", add_eos=True) + [7, 8])
    recipe_hashes = recipe_embedding.hash_ids(
        recipe_ids, document_positions_T=torch.arange(recipe_ids.numel())
    )
    fresh_hashes = recipe_embedding.hash_ids(
        recipe_ids[-2:], document_positions_T=torch.arange(2)
    )
    torch.testing.assert_close(recipe_hashes[-2:], fresh_hashes, rtol=0, atol=0)


def test_fresh_ngram_initialization_matches_pinned_hf(config):
    """Pinned HF vectors are checked before loading any checkpoint buffers."""
    from torchtitan.models.qwen3_8_flash_next.embedding import ngram_multipliers

    embedding = config.layers[1].ple.ple_embedding.build()
    assert embedding.layer_multipliers.tolist() == [
        112381549946653559,
        107512210695146523,
        55314113489879299,
    ]
    ids = torch.tensor([7, 8, 63, 9, 10])
    expected = torch.tensor(
        [
            [20, 519, 650, 1008],
            [29, 306, 529, 795],
            [210, 355, 645, 994],
            [95, 273, 719, 1044],
            [58, 442, 694, 863],
        ]
    )
    torch.testing.assert_close(
        embedding.hash_ids(ids, document_positions_T=torch.arange(5)),
        expected,
        rtol=0,
        atol=0,
    )
    assert ngram_multipliers(248320, 3, ple_layer_index=0, seed=1234) == (
        23703573157769,
        20109073645365,
        8052911324071,
    )


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
    actual = ple(
        x, input_ids_T=ids, document_positions_T=positions, padding_mask_T=padding
    )
    expected = torch.cat(
        [
            ple(
                x[i : i + 11],
                input_ids_T=ids[i : i + 11],
                document_positions_T=positions[i : i + 11],
                padding_mask_T=padding[i : i + 11],
            )
            for i in (0, 11)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (x, *ple.parameters())
    gradients = torch.autograd.grad(
        actual.square().sum(), parameters, retain_graph=True
    )
    isolated_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for gradient, isolated in zip(gradients, isolated_gradients):
        torch.testing.assert_close(gradient, isolated, rtol=2e-5, atol=2e-5)
    # An independent Conv1d oracle checks tap order and the dilation of three.
    captured = []
    handle = ple.norm_conv.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    zero_conv = copy.deepcopy(ple)
    with torch.no_grad():
        zero_conv.conv1d.weight.zero_()
    gated = zero_conv(x[:11], input_ids_T=ids[:11], document_positions_T=positions[:11])
    ple(x[:11], input_ids_T=ids[:11], document_positions_T=positions[:11])
    handle.remove()
    conv = F.conv1d(
        F.pad(captured[-1].T[None], (9, 0)), ple.conv1d.weight, groups=1024, dilation=3
    )
    torch.testing.assert_close(
        ple(x[:11], input_ids_T=ids[:11], document_positions_T=positions[:11]),
        gated + F.silu(conv[0].T),
        rtol=2e-5,
        atol=2e-5,
    )


def test_qsa_selection_matches_document_local_block_oracle(config):
    torch.manual_seed(42)
    indexer = config.layers[3].attention.indexer.build()
    indexer.init_states()
    x = torch.randn(63, 256)
    positions = torch.cat((torch.arange(31), torch.arange(32)))
    padding = torch.zeros(63, dtype=torch.bool)
    padding[5] = padding[61:] = True
    selected, scores = indexer(
        x, positions=positions, document_positions_T=positions, padding_mask_T=padding
    )
    metadata = QSAIndexMetadata.from_inputs(
        positions, padding_mask_T=padding, compress_ratio=4
    )
    cached_selected, cached_scores = indexer(
        x,
        positions=positions,
        document_positions_T=positions,
        padding_mask_T=padding,
        metadata=metadata,
    )
    torch.testing.assert_close(cached_selected, selected, atol=0, rtol=0)
    torch.testing.assert_close(cached_scores, scores, atol=0, rtol=0)
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


def test_qsa_equal_score_ties_are_independent_of_future_candidates(config):
    indexer = config.layers[3].attention.indexer.build()
    indexer.init_states()
    with torch.no_grad():
        indexer.index_qk_proj.weight.zero_()
    x = torch.randn(40, 256)
    padding = torch.arange(40) >= 20
    variants = (
        (x[:20], torch.arange(20), None),
        (x, torch.arange(40), None),
        (x, torch.arange(20).repeat(2), None),
        (x, torch.arange(40), padding),
    )
    for hidden, positions, mask in variants:
        selected, scores = indexer(
            hidden,
            positions=positions,
            document_positions_T=positions,
            padding_mask_T=mask,
        )
        actual = selected[19][selected[19] >= 0].sort().values
        torch.testing.assert_close(actual, torch.arange(16).int(), atol=0, rtol=0)
        assert not scores[19].any()


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
        document_positions_T=positions,
        padding_mask_T=padding,
    )
    inner = config.layers[3].attention.inner_attention.build().eval()
    q, k, v = [
        torch.randn(23, heads, 64, dtype=torch.float64, requires_grad=True)
        for heads in (4, 2, 2)
    ]
    saved_storage_bytes = {}
    saved_index_dtypes = []

    def pack(tensor):
        storage = tensor.untyped_storage()
        saved_storage_bytes[storage.data_ptr()] = storage.nbytes()
        if tensor.dtype in (torch.int32, torch.int64):
            saved_index_dtypes.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = inner(
            q,
            k,
            v,
            selected_indices_TJ=indices,
            scores_TB=scores,
            scale=1 / 8,
            enable_gqa=True,
        )
    # Saving every query's gathered K/V exceeds this input-sized storage budget.
    input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (q, k, v))
    assert sum(saved_storage_bytes.values()) < 2 * input_bytes
    assert saved_index_dtypes and set(saved_index_dtypes) == {torch.int32}
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


def test_qsa_block_kl_and_gradient_injection(config):
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
    torch.manual_seed(42)
    attention = config.layers[3].attention.build()
    attention.init_states()
    x = torch.randn(31, 256, requires_grad=True)
    positions = torch.arange(31)
    output = attention(
        x,
        None,
        positions,
        document_positions_T=positions,
        aux_loss_denominator=torch.tensor(31.0),
    )
    output.square().mean().backward()
    assert attention.indexer.index_qk_proj.weight.grad.norm() > 0
    assert torch.isfinite(x.grad).all()
    assert attention.inner_attention.indexer_loss.instance_acc > 0
    attention.inner_attention.indexer_loss._reset_accumulators()
    assert attention.inner_attention.indexer_loss.instance_acc == 0


@pytest.mark.parametrize(
    "input_kind", ["complete", "short_documents", "padding", "frozen"]
)
@pytest.mark.parametrize("save_patterns", [None, (), ("inner_attention",)])
def test_qsa_auxiliary_gradient_only_updates_the_indexer(
    config, input_kind, save_patterns
):
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
    if save_patterns is not None:
        attention.configure_remat_regions(save_patterns)
    forward = remat.checkpoint()(attention) if save_patterns is not None else attention
    output = forward(
        x,
        None,
        positions,
        document_positions_T=positions,
        padding_mask_T=padding,
        aux_loss_denominator=torch.tensor(31.0),
    )
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
        assert module.weight.grad is None or not module.weight.grad.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deltanet_uses_sigmoid_output_gate(config, dtype):
    torch.manual_seed(42)
    norm = config.layers[0].attention.norm.build().to(dtype)
    norm.init_states()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.2, 1.8, 128))
    x = torch.randn(5, 4, 128, dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_gate = gate.detach().clone().requires_grad_()
    reference_weight = norm.weight.detach().clone().requires_grad_()
    reference_x_float = reference_x.float()
    normalized = reference_x_float * torch.rsqrt(
        reference_x_float.square().mean(-1, keepdim=True) + norm.eps
    )
    # HF rounds the unweighted RMSNorm before multiplying by a fractional scale.
    expected = (
        normalized.to(dtype) * reference_weight * reference_gate.float().sigmoid()
    ).to(dtype)
    actual = norm(input=x, gate=gate)
    tolerance = (
        dict(rtol=1e-6, atol=1e-6) if dtype == torch.float32 else dict(rtol=0, atol=0)
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        (actual * upstream).sum(), (x, gate, norm.weight)
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (reference_x, reference_gate, reference_weight)
    )
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


def test_router_bfloat16_ties_mixing_and_backward(config):
    """Round before expert selection and retain differentiable normalized gates."""
    router = config.layers[3].feed_forward.router.build().bfloat16().eval()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, :2] = torch.tensor([1.0, 1 / 256])
        router.gate.weight[1, 0] = 1
        router.gate.weight[2, :2] = torch.tensor([1.0, 1 / 256])
    router.gate.inplace_wgrad_accum = False
    x = torch.ones(3, 256, dtype=torch.bfloat16, requires_grad=True)
    reference_weight = router.gate.weight.detach().float().requires_grad_()
    reference_x = x.detach().float().requires_grad_()
    logits = F.linear(reference_x, reference_weight).to(x.dtype).float()
    reference_scores, reference_ids = logits.softmax(-1).topk(2, -1)
    reference_scores = (reference_scores / reference_scores.sum(-1, keepdim=True)).to(
        x.dtype
    )
    scores, indices, routing = router(x)
    expected_routing = torch.zeros_like(logits, dtype=torch.bool).scatter(
        -1, reference_ids, True
    )
    assert torch.equal(routing, expected_routing)
    assert not torch.equal(
        indices.sort(-1).values,
        F.linear(reference_x, reference_weight).topk(2, -1).indices.sort(-1).values,
    )
    actual_weights = torch.zeros_like(logits).scatter(-1, indices, scores.float())
    expected_weights = torch.zeros_like(logits).scatter(
        -1, reference_ids, reference_scores.float()
    )
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)
    objective = torch.arange(8, dtype=torch.float32)
    actual_gradients = torch.autograd.grad(
        (actual_weights * objective).sum(), (x, router.gate.weight)
    )
    expected_gradients = torch.autograd.grad(
        (expected_weights * objective).sum(), (reference_x, reference_weight)
    )
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_grad.float(), expected_grad.to(x.dtype).float(), atol=1e-3, rtol=1e-2
        )


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
    for layer_id, error in (
        (0, "non-PLE layer"),
        (len(config.layers), "unknown PLE layer"),
    ):
        malformed = {shard.replace(".layers.1.", f".layers.{layer_id}."): hf[shard]}
        with pytest.raises(ValueError, match=error):
            adapter.from_hf(malformed)
    with_mtp = {**hf, "mtp.draft_head.weight": torch.empty(1)}
    without_mtp = adapter.from_hf(with_mtp)
    assert with_mtp.keys() == hf.keys() | {"mtp.draft_head.weight"}
    assert without_mtp.keys() == state.keys()
    consolidated = shard.replace(".shard_0", "")
    with pytest.raises(ValueError, match="both consolidated and split"):
        adapter.from_hf(
            {
                **hf,
                consolidated: state[
                    "layers.1.ple.ple_embedding.ngram_embedding.weight"
                ],
            }
        )
    incomplete = dict(hf)
    del incomplete[shard]
    with pytest.raises(ValueError, match="requires all"):
        adapter.from_hf(incomplete)


@pytest.mark.parametrize("all_padding", [False, True])
def test_packed_padding_is_excluded_from_moe_auxiliary_gradients(config, all_padding):
    from torchtitan.models.common.aux_loss import AuxLoss

    config.layers = [copy.deepcopy(config.layers[3])]
    config.layers[0].attention.inner_attention.indexer_loss = None
    saved_counts, saved_acc = AuxLoss._group_counts.copy(), AuxLoss.group_acc.copy()
    handles = []
    try:
        torch.manual_seed(42)
        model = config.build().train()
        model.init_states()
        router = model.layers["0"].feed_forward.router
        captured = []
        handles = [
            router.register_forward_pre_hook(
                lambda module, inputs: captured.append(inputs[0])
            ),
            router.register_forward_hook(
                lambda module, inputs, output: captured.extend((output[0], output[2]))
            ),
        ]
        ids = torch.randint(0, 63, (12,))
        positions = torch.cat((torch.arange(5), torch.arange(7)))
        padding = torch.ones(12, dtype=torch.bool)
        if not all_padding:
            padding[:4] = padding[5:9] = False
        denominator = (~padding).sum().float().clamp_min(1)
        metadata = model._get_attention_metadata(
            positions=positions, padding_mask=padding
        )
        model(
            ids,
            positions=positions,
            document_positions=positions,
            ple_input_ids=ids,
            attention_metadata=metadata,
            padding_mask=padding,
            aux_loss_denominators=denominator[None],
        )
        # A zero carrier isolates the configured batch-wise auxiliary gradient.
        actual_gradients = torch.autograd.grad(
            captured[1].sum() * 0, (captured[0], router.gate.weight)
        )
        counts_E = (captured[2] & ~padding[:, None]).sum(0)
        torch.testing.assert_close(
            router.tokens_per_expert_E,
            counts_E.to(router.tokens_per_expert_E),
            atol=0,
            rtol=0,
        )
        assert torch.count_nonzero(actual_gradients[0][padding]) == 0
        if all_padding:
            assert router.aux_loss.instance_acc == 0
            assert not actual_gradients[1].any()
        else:
            assert router.aux_loss.instance_acc > 0
            assert actual_gradients[0][~padding].norm() > 0
            assert actual_gradients[1].norm() > 0
    finally:
        for handle in handles:
            handle.remove()
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(saved_counts)
        AuxLoss.group_acc.clear()
        AuxLoss.group_acc.update(saved_acc)


def test_optimizer_resets_auxiliary_losses_once_with_remat(config):
    from torchtitan.components.optim import AdamW, OptimizersContainer
    from torchtitan.distributed.activation_remat import FullActivationRematConfig
    from torchtitan.models.common.aux_loss import AuxLoss
    from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextModel

    config.layers = [copy.deepcopy(config.layers[3]), copy.deepcopy(config.layers[1])]
    config.layers[1].attention = copy.deepcopy(config.layers[0].attention)
    saved_counts, saved_acc = AuxLoss._group_counts.copy(), AuxLoss.group_acc.copy()
    try:
        for use_remat in (False, True):
            torch.manual_seed(42)
            model = config.build()
            model.init_states()
            if use_remat:
                FullActivationRematConfig().apply(model)
            optimizer = OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=".*", lr=0, fused=False)]
            ).build(model_parts=[model])
            Qwen38FlashNextModel._register_optimizer_hooks(optimizer, [model], None)
            losses = [
                module
                for module in model.modules()
                if isinstance(module, QSAIndexerLoss)
            ]
            ids, positions = torch.randint(0, 63, (23,)), torch.arange(23)
            for _ in range(2):
                output = model(
                    ids,
                    positions=positions,
                    document_positions=positions,
                    ple_input_ids=ids,
                    aux_loss_denominators=torch.tensor([46.0]),
                )
                accumulated = [loss.instance_acc.clone() for loss in losses]
                output.square().mean().backward()
                for loss, expected in zip(losses, accumulated):
                    torch.testing.assert_close(
                        loss.instance_acc, expected, atol=0, rtol=0
                    )
            expected = sum(loss.instance_acc.clone() for loss in losses)
            assert expected > 0
            optimizer.step()
            torch.testing.assert_close(
                AuxLoss.group_acc[("loss", "qsa_indexer_loss")],
                expected,
                atol=0,
                rtol=0,
            )
            assert all(loss.instance_acc == 0 for loss in losses)
            optimizer.step()
            assert AuxLoss.group_acc[("loss", "qsa_indexer_loss")] == 0
    finally:
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(saved_counts)
        AuxLoss.group_acc.clear()
        AuxLoss.group_acc.update(saved_acc)


def test_qsa_teacher_packets_preserve_tail_and_tp_gradient_scaling():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torchtitan.models.qwen3_8_flash_next import attention as qsa

    torch.manual_seed(42)
    T, R, B, H, K = 273, 4, 3, 2, 4
    J = B * R + R - 1
    indices = torch.full((T, J), -1, dtype=torch.long)
    for i in range(T):
        complete = min((i + 1) // R, B)
        if complete:
            indices[i, : complete * R] = torch.arange(complete * R)
        tail = (i + 1) % R
        if tail:
            indices[i, B * R : B * R + tail] = torch.arange(i + 1 - tail, i + 1)
    indices[::97] = -1
    scores = torch.randn(T, B)
    invalid_blocks = indices[:, : B * R].reshape(T, B, R)[:, :, 0] < 0
    scores = scores.masked_fill(invalid_blocks, -torch.inf).requires_grad_()
    oracle_scores = scores.detach().clone().requires_grad_()
    query = torch.randn(T, H, K, requires_grad=True)
    key = torch.randn(T, 1, K, requires_grad=True)
    value = torch.randn(T, 1, K, requires_grad=True)
    remote_query = torch.randn(T, H, K)
    remote_key = torch.randn(T, 1, K)
    scale = K**-0.5

    def teacher_chunk(q, k, start, stop):
        selected = indices[start:stop]
        valid = selected >= 0
        q_chunk = q[start:stop].unsqueeze(2)
        keys = k[selected.clamp_min(0)].transpose(1, 2).repeat_interleave(H, 1)
        logits = (q_chunk.float() * keys.float()).sum(-1) * scale
        logits = logits.masked_fill(~valid[:, None], torch.finfo(torch.float32).min)
        return logits.softmax(-1).sum(1) / (2 * H) * valid

    remote_chunks, expected_outputs = [], []
    expected_raw_sum = oracle_scores.new_zeros(())
    with torch.no_grad():
        local_teachers = [
            teacher_chunk(query, key, start, start + 16) for start in range(0, T, 16)
        ]
        remote_teachers = [
            teacher_chunk(remote_query, remote_key, start, start + 16)
            for start in range(0, T, 16)
        ]
        for start in range(0, T, 16):
            selected = indices[start : start + 16]
            with sdpa_kernel(SDPBackend.MATH):
                expected_outputs.append(
                    F.scaled_dot_product_attention(
                        query[start : start + 16].unsqueeze(2),
                        key[selected.clamp_min(0)].transpose(1, 2),
                        value[selected.clamp_min(0)].transpose(1, 2),
                        attn_mask=(selected >= 0)[:, None, None],
                        scale=scale,
                        enable_gqa=True,
                    ).squeeze(2)
                )
        for start in range(0, len(remote_teachers), 16):
            remote_chunks.append(
                torch.cat(
                    [chunk[:, : B * R] for chunk in remote_teachers[start : start + 16]]
                )
            )
    for chunk_index, start in enumerate(range(0, T, 16)):
        teacher = local_teachers[chunk_index] + remote_teachers[chunk_index]
        blocks = teacher[:, : B * R].unflatten(-1, (B, R)).amax(-1)
        expected_raw_sum = expected_raw_sum + qsa.QSAIndexerLoss.block_kl(
            oracle_scores[start : start + 16], blocks
        )

    inner = (
        qsa.QSAInnerAttention.Config(
            compress_ratio=R,
            indexer_loss=qsa.QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
        )
        .build()
        .train()
    )
    communication_shapes, captured = [], []
    original_inject = inner.indexer_loss.inject

    def collective(local, operation, group):
        assert operation == "sum"
        remote = remote_chunks[len(communication_shapes)]
        assert local.shape == remote.shape
        communication_shapes.append(list(local.shape))
        return local + remote

    def inject(raw_sum, **kwargs):
        captured.append(raw_sum)
        return original_inject(raw_sum, **kwargs)

    with patch.object(qsa, "spmd_mesh_group", return_value=object()), patch.object(
        qsa, "spmd_mesh_size", return_value=2
    ), patch.object(qsa, "all_reduce", side_effect=collective), patch.object(
        qsa, "wait_tensor", side_effect=lambda x: x
    ), patch.object(
        inner.indexer_loss, "inject", side_effect=inject
    ):
        output = inner(
            query,
            key,
            value,
            selected_indices_TJ=indices,
            scores_TB=scores,
            scale=scale,
            enable_gqa=True,
            aux_loss_denominator=torch.tensor(float(T)),
        )

    assert communication_shapes == [[256, 12], [17, 12]]
    torch.testing.assert_close(output, torch.cat(expected_outputs), atol=0, rtol=0)
    torch.testing.assert_close(captured[0], expected_raw_sum, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(output.sum() * 0, scores)[0]
    expected_grad = torch.autograd.grad(
        expected_raw_sum * ((1.0 / T) / 2), oracle_scores
    )[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
