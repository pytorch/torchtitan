#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same-weight CUDA HF parity for Flash-Next text, vision and multimodal gradients.

Install transformers at 494470d677d05d44153d235d795e0d01040307ab, then run:
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next

The fixed 39-token BF16 decoder uses production GDN and grouped experts.
Standalone vision runs in FP32. Independent CPU feature equations live in
tests/unit_tests/cpu/models/qwen3_8_flash_next/.
"""

import argparse
import copy
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)

REFERENCE_REVISION = "494470d677d05d44153d235d795e0d01040307ab"
CHECKPOINT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

# T = tokens, D = model dimension, C = residual streams * D, E = experts,
# K = selected experts, V = vocabulary, P = patches, F = patch features,
# N = image/video grids, A = rotary coordinates, B = HF microbatch.
BF16_EPS = torch.finfo(torch.bfloat16).eps
FORWARD_ATOL = BF16_EPS / 2
FORWARD_RTOL = BF16_EPS
GRADIENT_ATOL = 1e-4
GRADIENT_RTOL = 4 * BF16_EPS
FIXTURE_SEED = 44
VISION_SEED = 1729
INDEXER_SEEDS = {3: 10001, 7: 20008}
# Ordinary Normal(0, 0.02) projections with qualified multimodal cutoff margins.
MM_ROUTER_SEEDS = {
    0: 30003,
    1: 31002,
    2: 32004,
    3: 33011,
    4: 34001,
    5: 35007,
    6: 36023,
    7: 37036,
}


@dataclass
class _Fixture:
    native: Any
    reference: Any
    vision: Any
    reference_vision: Any
    adapter: Qwen38FlashNextStateDictAdapter
    vision_adapter: Qwen38FlashNextStateDictAdapter
    tokens_T: torch.Tensor  # noqa: N815
    positions_T: torch.Tensor  # noqa: N815
    patches_PF: torch.Tensor  # noqa: N815


@dataclass
class _Execution:
    logits_TV: torch.Tensor  # noqa: N815
    reference_logits_TV: torch.Tensor  # noqa: N815
    loss: torch.Tensor
    reference_loss: torch.Tensor
    fused_TC: torch.Tensor  # noqa: N815
    reference_fused_TC: torch.Tensor  # noqa: N815
    embedding_gradient_TD: torch.Tensor  # noqa: N815
    reference_embedding_gradient_TD: torch.Tensor  # noqa: N815
    native_states: dict[int, torch.Tensor]
    reference_states: dict[int, torch.Tensor]
    native_routes: dict[int, tuple[torch.Tensor, torch.Tensor]]
    reference_routes: dict[int, tuple[torch.Tensor, ...]]
    native_gate_logits: dict[int, torch.Tensor]
    native_qsa_indices: dict[int, torch.Tensor]
    reference_qsa_masks: dict[int, torch.Tensor]
    reference_qsa_margins: dict[int, dict[str, Any]]


def _reference_text_config() -> Any:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=256,
        vocab_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        heads_per_ngram=2,
        ngram_vocab_size_base=257,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=16,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        use_cache=False,
    )


def _build_fixture(device: torch.device) -> _Fixture:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpVisionConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpVisionModel,
    )

    torch.manual_seed(42)
    config = build_model_config(
        "debugmodel_mm", seq_len=64, parallelism=ParallelismConfig()
    )
    for layer in config.layers:
        assert isinstance(layer.feed_forward, MoE.Config)
        layer.feed_forward.router.aux_loss = None
    vision_config = config.vision_encoder
    assert vision_config is not None
    config.vision_encoder = None
    config.vocab_size = (
        config.tok_embeddings.num_embeddings
    ) = config.lm_head.out_features = 64
    ple = config.layers[1].ple
    assert ple is not None
    ple.ple_embedding.vocab_size = 64
    ple.ple_embedding.eos_token_id = 63
    hf_config = _reference_text_config()
    hf_config._attn_implementation = "sdpa"
    reference = Qwen4ExpForCausalLM(hf_config).to(device).eval()
    with torch.device("meta"):
        native = config.build().to(torch.bfloat16)
    native.to_empty(device=device).eval()
    adapter = Qwen38FlashNextStateDictAdapter(config, None)
    native.init_states(buffer_device=device)
    native.load_state_dict(adapter.from_hf(reference.state_dict()), strict=True)

    generator = torch.Generator(device="cpu").manual_seed(FIXTURE_SEED)
    tokens_T = torch.randint(0, 63, (39,), generator=generator).to(device)
    tokens_T[7] = 63
    positions_T = torch.arange(39, device=device)
    # Preserve the original fixture's RNG stream after removing component runs.
    # The intervening draws determine its PLE convolution weights and vision input.
    torch.randn((39, 1024), generator=generator)
    with torch.no_grad():
        conv_weight = reference.model.layers[1].ple.conv1d.weight
        conv_weight.copy_(
            torch.empty(conv_weight.shape, device="cpu")
            .uniform_(-0.02, 0.02, generator=generator)
            .to(conv_weight)
        )
        native.layers["1"].ple.conv1d.weight.copy_(conv_weight)
    torch.randn((39, 256), generator=generator)
    torch.randn((39, 256), generator=generator)
    patches_PF = torch.randn((24, 1536), generator=generator).to(device)

    # HF computes rotary frequencies in FP32 even when backbone parameters are BF16.
    rotary = copy.deepcopy(reference.model.rotary_emb)
    reference.to(torch.bfloat16)
    reference.model.rotary_emb = rotary
    with torch.no_grad():
        for layer_id in range(8):
            weight_ED = reference.model.layers[layer_id].mlp.gate.weight
            weight_ED.zero_()
            weight_ED[:, 0] = torch.arange(weight_ED.shape[0], device=device) / 4
            native.layers[str(layer_id)].feed_forward.router.gate.weight.copy_(
                weight_ED
            )
        for layer_id, seed in INDEXER_SEEDS.items():
            weight = reference.model.layers[
                layer_id
            ].self_attn.indexer.index_qk_proj.weight
            index_generator = torch.Generator(device="cpu").manual_seed(seed)
            weight.copy_(
                torch.empty(weight.shape, device="cpu")
                .normal_(0, 0.02, generator=index_generator)
                .to(weight)
            )
            native.layers[str(layer_id)].attention.indexer.index_qk_proj.weight.copy_(
                weight
            )

    vision_hf_config = Qwen4ExpVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        depth=2,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=1024,
    )
    vision_hf_config._attn_implementation = "sdpa"
    torch.manual_seed(VISION_SEED)
    reference_vision = Qwen4ExpVisionModel(vision_hf_config).to(device).eval()
    vision = vision_config.build().to(device).eval()
    vision_model_config = copy.deepcopy(config)
    vision_model_config.vision_encoder = vision_config
    vision_adapter = Qwen38FlashNextStateDictAdapter(vision_model_config, None)
    vision_state = {
        "model.visual." + key: value
        for key, value in reference_vision.state_dict().items()
    }
    vision_state["lm_head.weight"] = reference.lm_head.weight
    vision_state[
        "model.language_model.embed_tokens.weight"
    ] = reference.model.embed_tokens.weight
    mapped = vision_adapter.from_hf(vision_state)
    vision.load_state_dict(
        {
            key.removeprefix("vision_encoder."): value
            for key, value in mapped.items()
            if key.startswith("vision_encoder.")
        },
        strict=True,
    )
    return _Fixture(
        native,
        reference,
        vision,
        reference_vision,
        adapter,
        vision_adapter,
        tokens_T,
        positions_T,
        patches_PF,
    )


def _compare_tensor(results, name, actual, expected, *, atol, rtol) -> None:
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{name}: {message}"
    )
    reference = expected.detach().double()
    difference = actual.detach().double() - reference
    results[name] = {
        "max_abs": float(difference.abs().max()),
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": float(difference.norm() / reference.norm()),
        "atol": atol,
        "rtol": rtol,
    }


def _compare_gradient(
    results, name, actual, expected, *, atol, rtol, bfloat16_compute
) -> None:
    if not bfloat16_compute:
        _compare_tensor(results, name, actual, expected, atol=atol, rtol=rtol)
        return
    # BF16 packed and separate projections round gradient sums differently.
    # Preserve both norm bounds instead of relative elementwise error near zero.
    reference = expected.detach().double()
    difference = actual.detach().double() - reference
    error_rms = difference.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    assert error_rms <= atol + rtol * reference_rms, name
    assert difference.abs().max() <= atol + rtol * reference.abs().max(), name
    results[name] = {
        "max_abs": float(difference.abs().max()),
        "rms": float(error_rms),
        "relative_l2": float(difference.norm() / reference.norm()),
        "atol": atol,
        "rtol": rtol,
        "comparison": "RMS and infinity norm",
    }


def _compare_parameter_gradients(
    results, name, native, expected_gradients, *, atol, rtol, bfloat16_compute
) -> None:
    actual_norm_squared = 0.0
    expected_norm_squared = 0.0
    gradient_dot = 0.0
    count = 0
    for key, parameter in native.named_parameters():
        actual = parameter.grad
        expected = expected_gradients.pop(key, None)
        assert (actual is None) == (
            expected is None
        ), f"{name}/{key}: gradient presence differs"
        if actual is None:
            continue
        _compare_gradient(
            results,
            f"{name}/{key}",
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            bfloat16_compute=bfloat16_compute,
        )
        actual_norm = actual.double().norm()
        expected_norm = expected.double().norm()
        dot = torch.dot(actual.double().flatten(), expected.double().flatten())
        results[f"{name}/{key}"].update(
            norm_ratio=float(actual_norm / expected_norm) if expected_norm else None,
            cosine=float(dot / (actual_norm * expected_norm))
            if actual_norm and expected_norm
            else None,
        )
        actual_norm_squared += float(actual_norm.square())
        expected_norm_squared += float(expected_norm.square())
        gradient_dot += float(dot)
        count += 1
    assert (
        not expected_gradients
    ), f"{name}: unmapped gradients: {list(expected_gradients)}"
    actual_global_norm = actual_norm_squared**0.5
    reference_global_norm = expected_norm_squared**0.5
    torch.testing.assert_close(
        actual_global_norm, reference_global_norm, atol=atol, rtol=rtol
    )
    global_cosine = gradient_dot / (actual_global_norm * reference_global_norm)
    assert global_cosine >= 1 - rtol**2, name
    results[name + "/global"] = {
        "parameter_tensors": count,
        "actual_grad_norm": actual_global_norm,
        "reference_grad_norm": reference_global_norm,
        "norm_ratio": actual_global_norm / reference_global_norm,
        "cosine": global_cosine,
    }


@torch.no_grad()
def _qsa_pruning_margin(indexer, inputs) -> dict[str, Any]:
    """Qualify HF's unspecified top-k ties on the actual no-cache fixture input."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    # G = blocks, H = heads, K = head width, R = block size, W = rotary width,
    # U = visible tokens, C = query projection channels.
    hidden_BTD, (cos_BTW, sin_BTW), visible_B1TT, cache = inputs
    assert cache is None and hidden_BTD.shape[0] == 1
    assert visible_B1TT.dtype == torch.bool
    num_tokens = hidden_BTD.shape[1]
    width = indexer.index_head_dim
    query_BTC, raw_key_BTK = indexer.index_qk_proj(hidden_BTD).split(
        (indexer.index_n_heads * width, width), -1
    )
    # HF returns one tensor when k=None, although its inferred type includes tuples.
    query_BTHK = cast(
        torch.Tensor,
        apply_rotary_pos_emb(
            indexer.q_layernorm(
                query_BTC.unflatten(-1, (indexer.index_n_heads, width))
            ),
            cos=cos_BTW[:, -num_tokens:],
            sin=sin_BTW[:, -num_tokens:],
            unsqueeze_dim=2,
        ),
    )
    gaps = []
    for query in range(num_tokens):
        visible_U = visible_B1TT[0, 0, query].nonzero().flatten()
        num_blocks = visible_U.numel() // indexer.compress_ratio
        if num_blocks <= indexer.block_topk:
            continue
        blocks_GR = visible_U[: num_blocks * indexer.compress_ratio].unflatten(
            -1, (num_blocks, indexer.compress_ratio)
        )
        pooled_GK = raw_key_BTK[0, blocks_GR].float().mean(1).to(raw_key_BTK.dtype)
        key_GK = cast(
            torch.Tensor,
            apply_rotary_pos_emb(
                indexer.k_layernorm(pooled_GK).unsqueeze(1),
                cos=cos_BTW[0, blocks_GR[:, 0]],
                sin=sin_BTW[0, blocks_GR[:, 0]],
            ),
        ).squeeze(1)
        # Match the pinned per-query matmul, ReLU, head sum and scale order.
        scores_GH = (query_BTHK[0, query].float() @ key_GK.float().T).T
        scores_G = scores_GH.relu().sum(-1) / width**0.5
        ranked_G = scores_G.sort(descending=True).values
        gaps.append(ranked_G[indexer.block_topk - 1] - ranked_G[indexer.block_topk])
    return {
        "pruned_queries": len(gaps),
        "minimum_cutoff_gap": float(torch.stack(gaps).min()) if gaps else None,
    }


def _execute(
    native,
    reference,
    reference_decoder,
    tokens_T,
    labels_T,
    *,
    native_kwargs,
    reference_kwargs,
) -> _Execution:
    """Run unmodified model forwards/backwards and observe their inputs and choices."""
    native.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    native_states = {}
    reference_states = {}
    native_routes = {}
    reference_routes = {}
    native_gate_logits = {}
    native_qsa_indices = {}
    reference_qsa_masks = {}
    reference_qsa_margins = {}
    embedding_gradients = {}
    fused_inputs = {}
    with ExitStack() as hooks:
        for slot, embedding, first_block in (
            (0, native.tok_embeddings, native.layers["0"]),
            (1, reference_decoder.embed_tokens, reference_decoder.layers[0]),
        ):

            def retain_embedding_gradient(module, inputs, output, i=slot):
                # Capture the original embedding node before visual scatter mutates it.
                output.register_hook(
                    lambda gradient: embedding_gradients.__setitem__(
                        i, gradient.detach()
                    )
                )

            hooks.callback(
                embedding.register_forward_hook(retain_embedding_gradient).remove
            )
            hooks.callback(
                first_block.register_forward_pre_hook(
                    lambda module, inputs, i=slot: fused_inputs.__setitem__(
                        i, inputs[0].detach()
                    )
                ).remove
            )
        for layer_id in range(8):
            native_layer = native.layers[str(layer_id)]
            reference_layer = reference_decoder.layers[layer_id]
            hooks.callback(
                native_layer.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: native_states.__setitem__(
                        i, output.detach()
                    )
                ).remove
            )
            hooks.callback(
                reference_layer.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: reference_states.__setitem__(
                        i, output[0].detach()
                    )
                ).remove
            )
            hooks.callback(
                native_layer.feed_forward.router.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: native_routes.__setitem__(
                        i, tuple(value.detach() for value in output[:2])
                    )
                ).remove
            )
            hooks.callback(
                native_layer.feed_forward.router.gate.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: native_gate_logits.__setitem__(
                        i, output.detach()
                    )
                ).remove
            )
            hooks.callback(
                reference_layer.mlp.gate.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: reference_routes.__setitem__(
                        i, tuple(value.detach() for value in output)
                    )
                ).remove
            )
            if layer_id in INDEXER_SEEDS:
                hooks.callback(
                    native_layer.attention.indexer.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_qsa_indices.__setitem__(
                            i, output[0].detach()
                        )
                    ).remove
                )

                def capture_reference_qsa(module, inputs, output, i=layer_id):
                    reference_qsa_masks[i] = output[0, 0].detach()
                    reference_qsa_margins[i] = _qsa_pruning_margin(module, inputs)

                hooks.callback(
                    reference_layer.self_attn.indexer.register_forward_hook(
                        capture_reference_qsa
                    ).remove
                )
        logits_TV = native(tokens_T, **native_kwargs)
        reference_output = reference(
            input_ids=tokens_T[None], labels=labels_T[None], **reference_kwargs
        )
        loss = F.cross_entropy(logits_TV[:-1].float(), labels_T[1:], ignore_index=-100)
        loss.backward()
        reference_output.loss.backward()
    return _Execution(
        logits_TV,
        reference_output.logits[0],
        loss,
        reference_output.loss,
        fused_inputs[0],
        fused_inputs[1][0],
        embedding_gradients[0],
        embedding_gradients[1][0],
        native_states,
        reference_states,
        native_routes,
        reference_routes,
        native_gate_logits,
        native_qsa_indices,
        reference_qsa_masks,
        reference_qsa_margins,
    )


def _compare_execution(results, name, execution: _Execution, *, multimodal) -> None:
    _compare_tensor(
        results,
        f"{name}/logits",
        execution.logits_TV,
        execution.reference_logits_TV,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    _compare_tensor(
        results,
        f"{name}/ce_loss",
        execution.loss,
        execution.reference_loss,
        atol=2e-4,
        rtol=2e-4,
    )
    _compare_tensor(
        results,
        f"{name}/fused_input",
        execution.fused_TC,
        execution.reference_fused_TC,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    _compare_gradient(
        results,
        f"{name}/embedding_input_gradient",
        execution.embedding_gradient_TD,
        execution.reference_embedding_gradient_TD,
        atol=GRADIENT_ATOL,
        rtol=GRADIENT_RTOL,
        bfloat16_compute=True,
    )
    num_tokens = execution.logits_TV.shape[0]
    for layer_id in INDEXER_SEEDS:
        indices_TJ = execution.native_qsa_indices[layer_id]
        mask_TT = torch.zeros(
            num_tokens, num_tokens + 1, dtype=torch.bool, device=indices_TJ.device
        ).scatter(-1, indices_TJ.long().masked_fill(indices_TJ < 0, num_tokens), True)
        assert torch.equal(
            mask_TT[:, :num_tokens], execution.reference_qsa_masks[layer_id]
        ), f"{name}: layer {layer_id} QSA selected-token masks differ"
        margin = execution.reference_qsa_margins[layer_id]
        if margin["pruned_queries"]:
            assert (
                margin["minimum_cutoff_gap"] > 0
            ), f"{name}: layer {layer_id} QSA pruning cutoff ties"
        results[f"{name}/qsa_selection_{layer_id}"] = {
            "selected_token_mask": "exact",
            **margin,
        }
    for layer_id in range(8):
        if not multimodal:
            _compare_tensor(
                results,
                f"{name}/block_{layer_id}",
                execution.native_states[layer_id],
                execution.reference_states[layer_id],
                atol=FORWARD_ATOL,
                rtol=FORWARD_RTOL,
            )
        native_scores_TK, native_ids_TK = execution.native_routes[layer_id]
        (
            reference_logits_TE,
            reference_scores_TK,
            reference_ids_TK,
        ) = execution.reference_routes[layer_id]
        assert torch.equal(
            native_ids_TK.sort(-1).values, reference_ids_TK.sort(-1).values
        ), f"{name}: layer {layer_id} router IDs differ"
        _compare_tensor(
            results,
            f"{name}/router_{layer_id}",
            native_scores_TK.gather(-1, native_ids_TK.argsort(-1)),
            reference_scores_TK.gather(-1, reference_ids_TK.argsort(-1)),
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
        )
        if multimodal:
            native_logits_TE = execution.native_gate_logits[layer_id].float()
            reference_logits_TE = reference_logits_TE.float()
            native_sorted_TE = native_logits_TE.sort(-1, descending=True).values
            reference_sorted_TE = reference_logits_TE.sort(-1, descending=True).values
            gap_T = torch.minimum(
                native_sorted_TE[:, 1] - native_sorted_TE[:, 2],
                reference_sorted_TE[:, 1] - reference_sorted_TE[:, 2],
            )
            drift_T = (native_logits_TE - reference_logits_TE).abs().amax(-1)
            scale_T = torch.maximum(
                native_logits_TE.abs().amax(-1), reference_logits_TE.abs().amax(-1)
            )
            # Retain four measured drifts/dtype epsilons and the original 0.002 floor.
            limit_T = torch.maximum(4 * drift_T, 4 * BF16_EPS * scale_T).clamp_min(
                0.002
            )
            assert bool(
                (gap_T > limit_T).all()
            ), f"{name}: layer {layer_id} weak router margin"
            results[f"{name}/router_margin_{layer_id}"] = {
                "minimum_cutoff_gap": float(gap_T.min()),
                "maximum_logit_drift": float(drift_T.max()),
                "minimum_gap_over_limit": float((gap_T / limit_T).min()),
            }
        else:
            sorted_TE = reference_logits_TE.float().sort(-1, descending=True).values
            results[f"{name}/router_margin_{layer_id}"] = float(
                (sorted_TE[:, 1] - sorted_TE[:, 2]).min()
            )


def run(*, device: str = "cuda") -> dict[str, Any]:
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(
            "The HF comparison requires production CUDA GDN/grouped experts. "
            "Run independent CPU feature equations with pytest instead."
        )
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    results: dict[str, Any] = {}
    with sdpa_kernel(SDPBackend.MATH):
        fixture = _build_fixture(target_device)
        for training in (False, True):
            fixture.native.train(training)
            fixture.reference.train(training)
            execution = _execute(
                fixture.native,
                fixture.reference,
                fixture.reference.model,
                fixture.tokens_T,
                fixture.tokens_T,
                native_kwargs={
                    "positions": fixture.positions_T,
                    "document_positions": fixture.positions_T,
                    "ple_input_ids": fixture.tokens_T,
                },
                reference_kwargs={"use_cache": False},
            )
            mode = "train" if training else "eval"
            _compare_execution(results, mode, execution, multimodal=False)
            expected_gradients = fixture.adapter.from_hf(
                {
                    key: value.grad
                    for key, value in fixture.reference.named_parameters()
                    if value.grad is not None
                }
            )
            _compare_parameter_gradients(
                results,
                f"{mode}/parameter_gradient",
                fixture.native,
                expected_gradients,
                atol=GRADIENT_ATOL,
                rtol=GRADIENT_RTOL,
                bfloat16_compute=True,
            )

        grid_NA = torch.tensor([[1, 4, 4], [2, 2, 2]], device=target_device)
        for training in (False, True):
            fixture.vision.train(training)
            fixture.reference_vision.train(training)
            fixture.vision.zero_grad(set_to_none=True)
            fixture.reference_vision.zero_grad(set_to_none=True)
            patches_PF = fixture.patches_PF.clone().requires_grad_()
            reference_patches_PF = fixture.patches_PF.clone().requires_grad_()
            output_TD = fixture.vision(patches_PF, grid_thw=grid_NA)
            reference_TD = fixture.reference_vision(
                reference_patches_PF, grid_thw=grid_NA
            ).pooler_output
            output_TD.square().mean().backward()
            reference_TD.square().mean().backward()
            mode = "train" if training else "eval"
            _compare_tensor(
                results,
                f"vision/{mode}/output",
                output_TD,
                reference_TD,
                atol=2e-5,
                rtol=2e-5,
            )
            _compare_gradient(
                results,
                f"vision/{mode}/input_gradient",
                patches_PF.grad,
                reference_patches_PF.grad,
                atol=3e-6,
                rtol=3e-5,
                bfloat16_compute=False,
            )
            # The full-model adapter requires an LM-head entry for partial vision maps.
            vision_source = {
                "model.visual." + key: value.grad
                for key, value in fixture.reference_vision.named_parameters()
                if value.grad is not None
            }
            vision_source["lm_head.weight"] = fixture.reference.lm_head.weight.grad
            expected_gradients = fixture.vision_adapter.from_hf(vision_source)
            expected_gradients.pop("lm_head.weight")
            expected_gradients = {
                key.removeprefix("vision_encoder."): value
                for key, value in expected_gradients.items()
            }
            _compare_parameter_gradients(
                results,
                f"vision/{mode}/parameter_gradient",
                fixture.vision,
                expected_gradients,
                atol=3e-6,
                rtol=3e-5,
                bfloat16_compute=False,
            )

        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
        )

        fixture.native.vision_encoder = fixture.vision
        mm_config = Qwen4ExpConfig(
            text_config=fixture.reference.config,
            vision_config=fixture.reference_vision.config,
            image_token_id=61,
            video_token_id=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
        )
        torch.manual_seed(VISION_SEED)
        reference_mm = Qwen4ExpForConditionalGeneration(mm_config)
        reference_mm.model.language_model = fixture.reference.model
        reference_mm.model.visual = fixture.reference_vision
        reference_mm.lm_head = fixture.reference.lm_head
        with torch.no_grad():
            for layer_id, seed in MM_ROUTER_SEEDS.items():
                weight_ED = reference_mm.model.language_model.layers[
                    layer_id
                ].mlp.gate.weight
                generator = torch.Generator(device="cpu").manual_seed(seed)
                weight_ED.copy_(
                    torch.empty(weight_ED.shape, device="cpu")
                    .normal_(0, 0.02, generator=generator)
                    .to(weight_ED)
                )
                fixture.native.layers[
                    str(layer_id)
                ].feed_forward.router.gate.weight.copy_(weight_ED)
        mapped_state = fixture.vision_adapter.from_hf(reference_mm.state_dict())
        native_state = fixture.native.state_dict()
        assert native_state.keys() == mapped_state.keys()
        assert all(
            torch.equal(native_state[key], mapped_state[key]) for key in native_state
        )

        tokens_T = torch.tensor(
            [3, 60, 61, 61, 61, 61, 59, 4, 60, 62, 59, 5, 6, 7, 8, 9],
            device=target_device,
        )
        image_grid_NA = torch.tensor([[1, 4, 4]], device=target_device)
        video_grid_NA = torch.tensor([[1, 2, 2]], device=target_device)
        mm_types_T = torch.zeros_like(tokens_T)
        mm_types_T[tokens_T == 61] = 1
        mm_types_T[tokens_T == 62] = 2
        rope_ABT, _ = reference_mm.model.get_rope_index(
            tokens_T[None], mm_types_T[None], image_grid_NA, video_grid_NA
        )
        positions_TA = rope_ABT[:, 0].transpose(0, 1).contiguous()
        document_positions_T = torch.arange(tokens_T.numel(), device=target_device)
        assert not torch.equal(positions_TA[:, 0], document_positions_T)
        placeholder_T = (tokens_T == 61) | (tokens_T == 62)
        labels_T = tokens_T.masked_fill(placeholder_T, -100)
        generator = torch.Generator(device="cpu").manual_seed(45)
        image_PF = torch.randn((16, 1536), generator=generator).to(target_device)
        video_PF = torch.randn((4, 1536), generator=generator).to(target_device)
        for training in (False, True):
            fixture.native.train(training)
            reference_mm.train(training)
            native_image_PF = image_PF.clone().requires_grad_()
            reference_image_PF = image_PF.clone().requires_grad_()
            native_video_PF = video_PF.clone().requires_grad_()
            reference_video_PF = video_PF.clone().requires_grad_()
            execution = _execute(
                fixture.native,
                reference_mm,
                reference_mm.model.language_model,
                tokens_T,
                labels_T,
                native_kwargs={
                    "positions": positions_TA,
                    "document_positions": document_positions_T,
                    "ple_input_ids": tokens_T,
                    "pixel_values": native_image_PF,
                    "pixel_values_videos": native_video_PF,
                    "grid_thw": image_grid_NA,
                    "grid_thw_videos": video_grid_NA,
                    "special_tokens": {"image_id": 61, "video_id": 62},
                },
                reference_kwargs={
                    "position_ids": rope_ABT,
                    "pixel_values": reference_image_PF,
                    "pixel_values_videos": reference_video_PF,
                    "image_grid_thw": image_grid_NA,
                    "video_grid_thw": video_grid_NA,
                    "mm_token_type_ids": mm_types_T[None],
                    "use_cache": False,
                },
            )
            mode = "train" if training else "eval"
            name = f"multimodal/{mode}"
            _compare_execution(results, name, execution, multimodal=True)
            assert native_image_PF.grad is not None and native_video_PF.grad is not None
            assert native_image_PF.grad.norm() > 0 and native_video_PF.grad.norm() > 0
            for field, actual, expected in (
                ("image_patch_gradient", native_image_PF.grad, reference_image_PF.grad),
                ("video_patch_gradient", native_video_PF.grad, reference_video_PF.grad),
            ):
                _compare_gradient(
                    results,
                    f"{name}/{field}",
                    actual,
                    expected,
                    atol=GRADIENT_ATOL,
                    rtol=GRADIENT_RTOL,
                    bfloat16_compute=True,
                )
            assert (
                torch.count_nonzero(execution.embedding_gradient_TD[placeholder_T]) == 0
            )
            assert (
                torch.count_nonzero(
                    execution.reference_embedding_gradient_TD[placeholder_T]
                )
                == 0
            )
            expected_gradients = fixture.vision_adapter.from_hf(
                {
                    key: value.grad
                    for key, value in reference_mm.named_parameters()
                    if value.grad is not None
                }
            )
            _compare_parameter_gradients(
                results,
                f"{name}/parameter_gradient",
                fixture.native,
                expected_gradients,
                atol=GRADIENT_ATOL,
                rtol=GRADIENT_RTOL,
                bfloat16_compute=True,
            )
            results[f"{name}/loss_values"] = {
                "native_ce": float(execution.loss.detach()),
                "reference_ce": float(execution.reference_loss.detach()),
                "native_image_gradient_norm": float(
                    native_image_PF.grad.double().norm()
                ),
                "native_video_gradient_norm": float(
                    native_video_PF.grad.double().norm()
                ),
            }

    import transformers

    return {
        "reference_revision": REFERENCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "device": device,
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(target_device),
        "cuda": torch.version.cuda,
        "component_dtype": "float32 (standalone vision)",
        "backbone_dtype": "torch.bfloat16",
        "full_token_count": 39,
        "full_model_indexer_seeds": INDEXER_SEEDS,
        "input_fixture_seed": FIXTURE_SEED,
        "vision_weight_seed": VISION_SEED,
        "multimodal_router_seeds": MM_ROUTER_SEEDS,
        "expert_compute": "shared BF16 grouped GEMM",
        "gdn_path": "production GPU kernel",
        "seed": 42,
        "attention_backend": "SDPA MATH",
        "tolerance_rationale": (
            "BF16 half-epsilon absolute plus one epsilon relative forward bounds; "
            "unchanged RMS/infinity and global norm/cosine gradient bounds. "
            "All routing is observed from unmodified forwards."
        ),
        "tokens": fixture.tokens_T.tolist(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", default="outputs/qwen3_8_flash_next_reference_parity.json"
    )
    args = parser.parse_args()
    report = run(device=args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
