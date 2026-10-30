# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import Sigmoid
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAAttention, QSAIndexerLoss
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)


@pytest.fixture
def config():
    cfg = build_model_config("debugmodel", seq_len=64, parallelism=ParallelismConfig())
    cfg.vocab_size = cfg.tok_embeddings.num_embeddings = cfg.lm_head.out_features = 64
    cfg.vision_encoder = None
    cfg.layers[1].ple.ple_embedding.vocab_size = 64
    cfg.layers[1].ple.ple_embedding.eos_token_id = 63
    return cfg


def test_released_architecture_and_checkpoint_schema():
    cfg = build_model_config("125B-A6B", seq_len=16, parallelism=ParallelismConfig())
    assert cfg.dim == 2560 and len(cfg.layers) == 48 and cfg.num_streams == 4
    assert [i for i, layer in enumerate(cfg.layers) if layer.ple] == [1]
    assert (
        sum(isinstance(layer.attention, QSAAttention.Config) for layer in cfg.layers)
        == 12
    )
    attn = cfg.layers[3].attention
    assert (attn.n_heads, attn.n_kv_heads, attn.head_dim, attn.rotary_dim) == (
        24,
        2,
        256,
        64,
    )
    assert (
        attn.indexer.num_heads,
        attn.indexer.head_dim,
        attn.indexer.token_budget,
    ) == (4, 128, 2048)
    ple = cfg.layers[1].ple.ple_embedding
    assert (ple.ngram_embedding.num_embeddings, ple.ngram_embedding.embedding_dim) == (
        320001536,
        160,
    )
    assert ple.split_ngram_parts == 128
    with torch.device("meta"):
        model = cfg.build()
    hf = Qwen38FlashNextStateDictAdapter(cfg, None).to_hf(model.state_dict())
    assert len(hf) == 1627
    assert "model.language_model.norm.weight" not in hf
    assert (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
        in hf
    )
    assert not any("input_layernorm" in key for key in hf)


@pytest.mark.parametrize("backend", ["flex", "varlen"])
def test_unsupported_backend_is_explicit(backend):
    with pytest.raises(ValueError, match="attn_backend"):
        build_model_config(
            "debugmodel", parallelism=ParallelismConfig(), attn_backend=backend
        )


def test_unsupported_cp_is_explicit():
    with pytest.raises(ValueError, match="context parallelism"):
        build_model_config(
            "debugmodel", parallelism=ParallelismConfig(context_parallel_degree=2)
        )


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
    assert embedding.layer_multipliers.dtype == torch.int64


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
    x = torch.randn(39, 256)
    positions = torch.cat((torch.arange(19), torch.arange(20)))
    padding = torch.zeros(39, dtype=torch.bool)
    padding[5] = padding[37:] = True
    selected, scores = indexer(
        x, positions=positions, document_positions_T=positions, padding_mask_T=padding
    )
    projection = indexer.index_qk_proj(x)
    queries = indexer._rotate(
        indexer.q_layernorm(projection[:, :256].unflatten(-1, (4, 64))), positions
    )
    for query in range(39):
        begin = 0 if query < 19 else 19
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
        winners = oracle_scores.topk(min(4, complete)).indices
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
    actual = inner(
        q,
        k,
        v,
        selected_indices_TJ=indices,
        scores_TB=scores,
        scale=1 / 8,
        enable_gqa=True,
    )
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


def test_deltanet_uses_sigmoid_output_gate(config):
    cfg = config.layers[0].attention
    assert isinstance(cfg, GDN.Config) and isinstance(
        cfg.norm.activation_fn, Sigmoid.Config
    )
    norm = cfg.norm.build()
    norm.init_states()
    x = torch.randn(5, 4, 128)
    gate = torch.randn_like(x)
    expected = F.rms_norm(x.float(), (128,), eps=1e-6) * gate.sigmoid()
    torch.testing.assert_close(norm(x, gate), expected)


def test_router_bfloat16_ties_mixing_and_backward(config):
    """Round before expert selection and retain differentiable normalized gates."""
    router = config.layers[3].feed_forward.router.build().bfloat16().eval()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, :2] = torch.tensor([1.0, 1 / 256])
        router.gate.weight[1, 0] = 1
        router.gate.weight[2, :2] = torch.tensor([1.0, 1 / 256])
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
    del hf[shard]
    with pytest.raises(ValueError, match="requires all"):
        adapter.from_hf(hf)


def test_original_ids_and_gradients_survive_a_pipeline_split(config):
    torch.manual_seed(42)
    config.layers = [copy.deepcopy(config.layers[3]), copy.deepcopy(config.layers[1])]
    config.layers[1].attention = copy.deepcopy(config.layers[0].attention)
    full = config.build().eval()
    full.init_states()
    first, last = copy.deepcopy(full), copy.deepcopy(full)
    del first.layers["1"]
    first.norm = first.lm_head = first.hyper_connection_mixer = None
    del last.layers["0"]
    last.tok_embeddings = None
    ids = torch.randint(0, 63, (25,))
    positions = torch.cat((torch.arange(11), torch.arange(14)))
    expected = full(ids, positions=positions)
    hidden = first(ids, positions=positions)
    actual = last(
        hidden, positions=positions, ple_input_ids=ids, document_positions=positions
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            # Splitting the graph can reorder FP32 gradient additions.
            torch.testing.assert_close(
                parameter.grad,
                dict(full.named_parameters())[name].grad,
                rtol=1e-5,
                atol=1e-7,
                msg=lambda message: f"{name}: {message}",
            )


def test_optimizer_resets_auxiliary_losses_once_with_remat(config):
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
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            Qwen38FlashNextModel._register_optimizer_hooks(optimizer, [model], None)
            losses = [
                module
                for module in model.modules()
                if isinstance(module, QSAIndexerLoss)
            ]
            ids, positions = torch.randint(0, 63, (23,)), torch.arange(23)
            for _ in range(2):
                output = model(
                    ids, positions=positions, aux_loss_denominators=torch.tensor([46.0])
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


def test_integration_recipe_and_pipeline_placement():
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.common.decoder import TransformerBlock
    from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextModel

    from tests.integration_tests.h100 import build_h100_tests_list

    test = next(
        test
        for test in build_h100_tests_list()
        if test.test_name == "qwen3_8_flash_next_all_parallelism"
    )
    cfg = test.configs[0]()
    assert test.ngpu == 8 and test.use_real_pg
    assert isinstance(cfg.model, Qwen38FlashNextModel.Config)
    assert all(isinstance(layer, TransformerBlock.Config) for layer in cfg.model.layers)
    parallelism = cfg.parallelism
    assert (
        parallelism.data_parallel_shard_degree,
        parallelism.tensor_parallel_degree,
        parallelism.expert_parallel_degree,
        parallelism.pipeline_parallel_degree,
    ) == (2, 2, 4, 2)
    assert parallelism.num_pp_microbatches == 2 and cfg.training.steps >= 10
    assert (
        cfg.debug.seed == 42
        and cfg.debug.deterministic
        and not cfg.debug.deterministic_warn_only
    )
    stages = _generate_llm_fqn_per_model_part(
        2,
        8,
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    assert "layers.1" not in stages[0] and "layers.1" in stages[1]


def test_numerical_recipe_and_golden_retain_model_features():
    from torchtitan_recipes.tests.models.qwen3_8_flash_next import (
        qwen38_flash_next_debugmodel,
        qwen38_flash_next_debugmodel_tp2,
    )

    from tests.integration_tests.b200 import build_b200_tests_list

    test = next(
        test
        for test in build_b200_tests_list()
        if test.test_name == "qwen3_8_flash_next_tp_ep_numerics"
    )
    assert list(test.configs) == [qwen38_flash_next_debugmodel_tp2]
    assert test.loss_compare_seed_config is qwen38_flash_next_debugmodel
    assert test.ngpu == 2 and test.use_real_pg and not test.disabled
    cfg = test.configs[0]()
    assert (
        cfg.parallelism.tensor_parallel_degree,
        cfg.parallelism.expert_parallel_degree,
        cfg.parallelism.pipeline_parallel_degree,
        cfg.parallelism.context_parallel_degree,
    ) == (2, 2, 1, 1)
    assert cfg.parallelism.enable_sequence_parallel and cfg.debug.spmd_typechecking
    assert cfg.debug.seed == 42 and cfg.debug.deterministic
    assert not cfg.debug.deterministic_warn_only
    assert cfg.training.steps >= 10 and cfg.training.disable_cuda_graphs
    assert cfg.activation_remat is None and not cfg.model.local_compile_regions
    assert cfg.model.num_streams == 4 and cfg.model.layers[1].ple is not None
    assert any(isinstance(layer.attention, GDN.Config) for layer in cfg.model.layers)
    sparse_layers = [
        layer
        for layer in cfg.model.layers
        if isinstance(layer.attention, QSAAttention.Config)
    ]
    assert sparse_layers and all(
        isinstance(layer.feed_forward, MoE.Config) for layer in cfg.model.layers
    )
    assert all(
        layer.attention.indexer.token_budget < cfg.training.max_context_length
        for layer in sparse_layers
    )
    assert test.golden_numerics_path is not None
    golden = Path(__file__).resolve().parents[4] / test.golden_numerics_path
    lines = golden.read_text().splitlines()
    assert "# step loss grad_norm" in lines
    rows = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert [int(row[0]) for row in rows] == list(range(1, 11))
    assert all(
        len(row) == 3 and all(math.isfinite(float(value)) for value in row[1:])
        for row in rows
    )
