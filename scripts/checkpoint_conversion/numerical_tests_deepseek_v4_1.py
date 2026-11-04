# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare the port with the pinned author's inference implementation on CPU.

The author provides inference kernels, not a training reference. This harness
executes the unchanged Python model with independent PyTorch implementations
of those kernels. Dense math uses FP32 on both sides; activation quantization
and the quantized Engram table remain enabled. It checks text/image logits,
per-block outputs, vision gradients, DSpark logits/confidence and gradients,
and text backbone and sliding-window block gradients.
The image arithmetic comparison holds attention quantizer inputs fixed and
checks both quantizers independently on those inputs. Raw independently rounded
image logits are reported separately; rounding at FP8/FP4 midpoints can amplify
otherwise acceptable FP32 arithmetic differences.

B: draft blocks, T: context tokens, S: draft positions, D: hidden width,
V: vocabulary, F: concatenated backbone width.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.models.deepseek_v4_1.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)

REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
SOURCE_HASHES = {
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}


def quantize_fp8(
    x, block_size=32, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    blocks = x.float().unflatten(-1, (-1, block_size))
    scales = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    if scale_fmt is not None:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    values = (blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequantized = (values.float() * scales).flatten(-2).to(x.dtype)
    if inplace:
        x.copy_(x + (dequantized - x).detach())
        return x
    return values.flatten(-2), scales.squeeze(-1).to(scale_dtype)


def quantize_fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    blocks = x.float().unflatten(-1, (-1, block_size))
    nvfp4 = scale_dtype == torch.float8_e4m3fn
    minimum = 6 * (2**-9 if nvfp4 else 2**-126)
    scales = blocks.abs().amax(-1, keepdim=True).clamp_min(minimum) / 6
    scales = (
        scales.to(scale_dtype).float()
        if nvfp4
        else torch.exp2(torch.ceil(torch.log2(scales)))
    )
    # Prefer even codes when two E2M1 values are equally close.
    levels = blocks.new_tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])
    distances = (blocks.abs().unsqueeze(-1) / scales.unsqueeze(-1) - levels).abs()
    values = torch.copysign(levels[distances.argmin(-1)], blocks)
    dequantized = (values * scales).flatten(-2).to(x.dtype)
    if not inplace:
        raise ValueError(
            "The reference harness exercises dequantized activations only."
        )
    x.copy_(x + (dequantized - x).detach())
    return x


def sparse_attention(q, kv, sink, indices, scale):
    batch, tokens, _, _ = q.shape
    selected = kv[torch.arange(batch)[:, None, None], indices.clamp_min(0).long()]
    scores = torch.einsum("bthd,btkd->bthk", q.float(), selected.float()) * scale
    scores = scores.masked_fill(indices[:, :, None] < 0, -torch.inf)
    sink_scores = sink.float()[None, None, :, None].expand(batch, tokens, -1, 1)
    probabilities = torch.cat((scores, sink_scores), -1).softmax(-1)[..., :-1]
    return torch.einsum("bthk,btkd->bthd", probabilities, selected.float()).to(q.dtype)


def hc_coefficients(mixes, scales, bases, hc_mult, iterations, eps):
    pre_raw, post_raw, comb_raw = mixes.split([hc_mult, hc_mult, hc_mult**2], -1)
    pre = torch.sigmoid(pre_raw * scales[0] + bases[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_raw * scales[1] + bases[hc_mult : 2 * hc_mult])
    comb = (comb_raw * scales[2] + bases[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def unavailable_gemm(*args, **kwargs):
    raise AssertionError(
        "The FP32 parity harness must not call quantized weight GEMMs."
    )


def load_reference(reference_dir):
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((reference_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} is not from the pinned HF revision {REVISION}.")
    kernel: Any = ModuleType("kernel")
    kernel.act_quant = quantize_fp8
    kernel.fp4_act_quant = quantize_fp4
    kernel.sparse_attn = sparse_attention
    kernel.hc_split_sinkhorn = hc_coefficients
    kernel.fp4_gemm = kernel.fp8_gemm = unavailable_gemm
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(reference_dir))
    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_author", reference_dir / "model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IdentityTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.backend_tokenizer = self

    def __len__(self):
        return self.vocab_size

    def decode(self, ids, *, skip_special_tokens):
        return f"token_{ids[0]}"

    def id_to_token(self, token_id):
        return f"token_{token_id}"


def compare_draft_gradients(
    native, *, reference, adapter, features_TF, tokens_T, anchor
):
    """Compare a conditional five-position draft objective with frozen shared weights."""
    frozen = [
        (parameter, parameter.requires_grad)
        for module in (reference.embed, reference.head)
        for parameter in module.parameters()
    ]
    for parameter, _ in frozen:
        parameter.requires_grad_(False)
    anchors_B = torch.tensor([anchor])
    input_ids_B = tokens_T[anchors_B]
    reference_features_BTF = features_TF.detach().clone()[None]
    native_features_TF = features_TF.detach().clone().requires_grad_()
    with torch.autograd.graph.allow_mutation_on_saved_tensors():
        type(reference).forward_spec.__wrapped__(
            reference, input_ids_B, reference_features_BTF[:, :anchor], start_pos=0
        )
        output_ids_BS, reference_logits_BSV, reference_confidence_BS = type(
            reference
        ).forward_spec.__wrapped__(
            reference,
            input_ids_B,
            reference_features_BTF[:, anchor : anchor + 1],
            start_pos=anchor,
        )
        previous_BS = output_ids_BS[:, :-1].clone()
        draft_ids_BS = torch.full((1, 5), native.drafter.config.noise_token_id)
        draft_ids_BS[:, 0] = input_ids_B
        native_logits_BSV, native_confidence_BS = native.forward_draft(
            native_features_TF,
            native.tok_embeddings(draft_ids_BS.flatten()),
            previous_BS,
            anchors_B,
            main_positions_T=torch.arange(tokens_T.numel()),
            output_weight_VD=native.lm_head.weight,
        )
        targets_S = tokens_T[anchor + 1 : anchor + 6]
        native_loss = F.cross_entropy(
            native_logits_BSV.flatten(0, 1), targets_S
        ) + F.binary_cross_entropy_with_logits(
            native_confidence_BS, torch.full_like(native_confidence_BS, 0.5)
        )
        reference_loss = F.cross_entropy(
            reference_logits_BSV.flatten(0, 1), targets_S
        ) + F.binary_cross_entropy_with_logits(
            reference_confidence_BS, torch.full_like(reference_confidence_BS, 0.5)
        )
        torch.testing.assert_close(native_loss, reference_loss, rtol=2e-5, atol=2e-6)
        native_loss.backward()
        reference_loss.backward()
    assert native_features_TF.grad is None
    assert all(
        parameter.grad is None
        for name, parameter in native.named_parameters()
        if not name.startswith("drafter.")
    )
    gradients = adapter.to_hf(
        {
            name: parameter.grad
            for name, parameter in native.named_parameters()
            if name.startswith("drafter.") and parameter.grad is not None
        }
    )
    expected_names = {
        name
        for name, parameter in reference.named_parameters()
        if name.startswith("mtp.") and parameter.grad is not None
    }
    assert expected_names <= gradients.keys(), expected_names - gradients.keys()
    maximum_error, inactive_experts = 0.0, 0
    actual_norm_squared, reference_norm_squared = 0.0, 0.0
    for name, gradient in gradients.items():
        expected = reference.get_parameter(name).grad
        if expected is None:
            # GroupedLinear returns zero rows for experts skipped by the loop oracle.
            assert ".ffn.experts." in name
            expected = torch.zeros_like(gradient)
            inactive_experts += 1
        torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=2e-6)
        maximum_error = max(maximum_error, (gradient - expected).abs().max().item())
        actual_norm_squared += gradient.double().square().sum().item()
        reference_norm_squared += expected.double().square().sum().item()
    actual_norm = actual_norm_squared**0.5
    reference_norm = reference_norm_squared**0.5
    torch.testing.assert_close(actual_norm, reference_norm, rtol=2e-4, atol=2e-6)
    for parameter, requires_grad in frozen:
        parameter.requires_grad_(requires_grad)
    native.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return {
        "draft_objective": "five-position next-token cross entropy plus confidence BCE",
        "draft_cross_entropy_plus_confidence_loss": native_loss.item(),
        "reference_draft_cross_entropy_plus_confidence_loss": reference_loss.item(),
        "draft_parameter_grad_max_abs_error": maximum_error,
        "num_draft_parameter_gradients": len(gradients),
        "num_zero_routed_expert_gradients": inactive_experts,
        "draft_gradient_norm_ratio": actual_norm / reference_norm,
        "draft_shared_projection_gradients_stopped": True,
    }


def compare_image_quantizers(
    native, *, reference, author, tokens, token_types, patches, image
):
    """Check image arithmetic and quantizers separately from rounding boundaries.

    Execute inside no_grad. The native image forward retains quantization;
    the author quantizer receives the same source activation at each call.
    Each author's incoming activation must still pass an FP32 arithmetic check.
    """
    from torchtitan.models.deepseek_v4_1 import attention

    samples = []

    def capture(quantize, name):
        def forward(value, *args, **kwargs):
            original = value.detach().clone()
            output = quantize(value, *args, **kwargs)
            samples.append((name, original, output.detach().clone()))
            return output

        return forward

    with (
        patch.object(
            attention, "fp8_activation", capture(attention.fp8_activation, "fp8")
        ),
        patch.object(
            attention, "fp4_activation", capture(attention.fp4_activation, "fp4")
        ),
    ):
        actual = native(
            tokens,
            token_types=token_types,
            pixel_values=patches.detach(),
            grid_hw=((6, 6),),
        )
    _, raw_expected, _ = reference(
        tokens[None], images=[[image]], token_types=token_types[None]
    )
    raw_error = (actual - raw_expected[0]).abs().max().item()
    cursor, input_error = 0, 0.0

    def compare_quantizer(quantize, name):
        def forward(value, *args, **kwargs):
            nonlocal cursor, input_error
            captured_name, original, expected = samples[cursor]
            cursor += 1
            assert captured_name == name
            original = original.reshape(value.shape)
            torch.testing.assert_close(value, original, rtol=2e-5, atol=2e-6)
            input_error = max(input_error, (value - original).abs().max().item())
            independent = original.clone()
            quantize(independent, *args, **kwargs)
            torch.testing.assert_close(
                independent, expected.reshape(value.shape), rtol=0, atol=0
            )
            value.copy_(independent)
            return value

        return forward

    with (
        patch.object(author, "act_quant", compare_quantizer(author.act_quant, "fp8")),
        patch.object(
            author, "fp4_act_quant", compare_quantizer(author.fp4_act_quant, "fp4")
        ),
    ):
        _, expected, _ = reference(
            tokens[None], images=[[image]], token_types=token_types[None]
        )
    assert cursor == len(samples)
    torch.testing.assert_close(actual, expected[0], rtol=2e-4, atol=2e-5)
    return (
        actual,
        expected,
        {
            "image_attention_quantizer_inputs": "held identical for independent quantizer checks",
            "image_uncontrolled_logit_max_abs_error": raw_error,
            "image_quantizer_input_max_abs_error": input_error,
            "num_image_quantizer_comparisons": cursor,
        },
    )


def compare(reference_dir, *, seq_len):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    author = load_reference(reference_dir)
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=seq_len,
        indexer_loss_coeff=0.0,
        vision_enabled=True,
        num_draft_layers=3,
        vision_attention_backend="reference",
        attention_backend="reference",
    )
    assert config.drafter is not None
    for layer in (*config.layers, *config.drafter.layers):
        layer.feed_forward.routed_experts.compute_dtype = "float32"
    native = config.build()
    native.init_states(buffer_device=torch.device("cpu"))
    native.eval()
    adapter = DeepSeekV41StateDictAdapter(config, "")
    engram_rows = []
    for layer_id in config.engram_layer_ids:
        engram_config = config.layers[layer_id].engram
        assert engram_config is not None
        engram_rows.append(engram_config.embed.num_embeddings)
    args = author.ModelArgs(
        max_batch_size=1,
        max_seq_len=seq_len + 5,
        dtype="bf16",
        expert_dtype=None,
        temperature=0,
        vocab_size=config.vocab_size,
        dim=config.dim,
        moe_inter_dim=128,
        n_layers=len(config.layers),
        n_mtp_layers=3,
        n_heads=8,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=1.5,
        swiglu_limit=10,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=16,
        norm_eps=1e-20,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(*config.compress_ratios, 0, 0, 0),
        kv_source_layers=config.kv_source_layers,
        index_source_layers=config.index_source_layers,
        compress_rope_theta=160000,
        original_seq_len=65536,
        rope_factor=16,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        candidate_source_layer=8,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        engram_layer_ids=config.engram_layer_ids,
        engram_num_embeddings=tuple(engram_rows),
        engram_max_ngram_size=4,
        engram_vocab_size=128,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=config.vocab_size,
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=128,
        vision_patch_size=2,
        vision_downsample_ratio=3,
        dspark_block_size=5,
        dspark_noise_token_id=config.drafter.noise_token_id,
        dspark_target_layer_ids=config.drafter.target_layer_ids,
        dspark_markov_rank=32,
        dspark_n_routed_experts=config.drafter.layers[0].feed_forward.num_experts,
        dspark_n_activated_experts=config.drafter.layers[0].feed_forward.router.top_k,
    )
    reference = (
        author.Transformer(args, IdentityTokenizer(config.vocab_size)).float().eval()
    )
    for layer_id in config.engram_layer_ids:
        table = native.layers[str(layer_id)].engram.embed.weight
        with torch.no_grad():
            values, scales = quantize_fp8(table, 32, "ue8m0")
            table.copy_(
                (values.float().unflatten(-1, (-1, 32)) * scales[:, :, None]).flatten(
                    -2
                )
            )
        # Dequantized E4M3/E8M0 entries are exactly representable in BF16.
        # Promote the author's BF16 read to the harness's FP32 dense math.
        embedding = reference.layers[layer_id].engram.embed
        # Stored scales describe quantization; the logical table is the trained weight.
        embedding.scale.requires_grad_(False)
        original_forward = embedding.forward
        embedding.forward = lambda ids, read=original_forward: read(ids).float()
    weights = adapter.to_hf(native.state_dict())
    for layer_id in config.engram_layer_ids:
        weights[f"layers.{layer_id}.engram.embed.scale"] = torch.ones_like(
            reference.layers[layer_id].engram.embed.scale
        )
    # The Python reference registers aliases for the shared embedding and head;
    # the published safetensors checkpoint stores each shared weight only once.
    for index in range(3):
        weights[f"mtp.{index}.embed.weight"] = weights["embed.weight"]
        weights[f"mtp.{index}.head.weight"] = weights["head.weight"]
    reference.load_state_dict(weights, strict=True)
    original_head = reference.head.forward
    reference.head.forward = lambda hidden, full_logits=True: original_head(
        hidden, full_logits=True
    )
    native_blocks, reference_blocks = {}, {}
    native_parts, reference_parts = {}, {}
    native_shared, reference_shared = {}, {}
    handles = []
    for layer_id in range(len(config.layers)):
        handles.append(
            native.layers[str(layer_id)].register_forward_hook(
                lambda module, inputs, output, key=layer_id: native_blocks.update(
                    {key: output[0].detach().clone()}
                )
            )
        )
        handles.append(
            reference.layers[layer_id].register_forward_hook(
                lambda module, inputs, output, key=layer_id: reference_blocks.update(
                    {key: output[0].detach().squeeze(0).clone()}
                )
            )
        )
    for native_name, reference_name in (
        ("attention_norm", "attn_norm"),
        ("ffn_norm", "ffn_norm"),
        ("attention.wq_a", "attn.wq_a"),
        ("attention.q_norm", "attn.q_norm"),
        ("attention.wq_b", "attn.wq_b"),
        ("attention.wkv", "attn.wkv"),
        ("attention.kv_norm", "attn.kv_norm"),
        ("attention.wo_b", "attn.wo_b"),
        ("feed_forward", "ffn"),
    ):
        handles.append(
            native.layers["0"]
            .get_submodule(native_name)
            .register_forward_hook(
                lambda module, inputs, output, key=native_name: native_parts.update(
                    {key: output.detach().clone()}
                )
            )
        )
        handles.append(
            reference.layers[0]
            .get_submodule(reference_name)
            .register_forward_hook(
                lambda module, inputs, output, key=native_name: reference_parts.update(
                    {key: output.detach().reshape(native_parts[key].shape).clone()}
                )
            )
        )
    handles.append(
        native.layers["8"].attention.register_forward_hook(
            lambda module, inputs, output: native_shared.update(
                {
                    "kv": output[1].detach().clone(),
                    "index": output[2].detach().clone(),
                    "selected": output[3].detach().clone(),
                    "candidates": output[5].detach().clone(),
                }
            )
        )
    )
    handles.append(
        reference.layers[8].attn.register_forward_hook(
            lambda module, inputs, output: reference_shared.update(
                {
                    "kv": author.shared_attn.compress_kv[0, :seq_len].clone(),
                    "index": author.shared_attn.index_k[0, :seq_len].clone(),
                    "selected": torch.where(
                        author.shared_attn.topk_idxs[0] >= 0,
                        author.shared_attn.topk_idxs[0] - seq_len,
                        -1,
                    ).clone(),
                }
            )
        )
    )
    tokens = torch.randint(3, config.vocab_size, (seq_len,))
    with torch.no_grad():
        actual, native_context = native(tokens, return_draft_context=True)
        actual = actual.float()
        _, expected, reference_context = reference(tokens[None])
    for handle in handles:
        handle.remove()
    errors = {
        str(key): (native_blocks[key] - reference_blocks[key]).abs().max().item()
        for key in native_blocks
    }
    for layer_id in native_blocks:
        torch.testing.assert_close(
            native_blocks[layer_id], reference_blocks[layer_id], rtol=2e-4, atol=1e-5
        )
    component_errors = {}
    for name, value in native_parts.items():
        torch.testing.assert_close(value, reference_parts[name], rtol=1e-4, atol=1e-5)
        component_errors[name] = (value - reference_parts[name]).abs().max().item()
    shared_errors = {}
    for name in ("kv", "index", "selected"):
        torch.testing.assert_close(
            native_shared[name], reference_shared[name], rtol=0, atol=0
        )
        shared_errors[name] = (
            (native_shared[name] - reference_shared[name]).abs().max().item()
        )
    torch.testing.assert_close(actual, expected.squeeze(0), rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(
        native_context, reference_context[0], rtol=2e-4, atol=1e-5
    )
    patches = torch.randn(36, 12, requires_grad=True)
    reference_patches = patches.detach().clone().requires_grad_()
    encoded = native.vision(patches, grid_hw=((6, 6),))
    reference_encoded = reference.vision(reference_patches, 6, 6)
    aligned = native.aligner(encoded, grid_hw=((6, 6),))
    reference_aligned = reference.aligner(reference_encoded, 6, 6)
    torch.testing.assert_close(encoded, reference_encoded, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(aligned, reference_aligned, rtol=2e-4, atol=2e-6)
    (encoded.square().mean() + aligned.square().mean()).backward()
    (reference_encoded.square().mean() + reference_aligned.square().mean()).backward()
    torch.testing.assert_close(
        patches.grad, reference_patches.grad, rtol=2e-4, atol=2e-6
    )
    types = torch.tensor([-1, -1, 0, 1, 1, 2, 1, 1, 2, 3] + [-1] * (seq_len - 10))
    image = SimpleNamespace(
        start=2, types=types[2:10], patches=patches.detach(), n_vit_h=6, n_vit_w=6
    )
    with torch.no_grad():
        image_actual, image_expected, image_quantization = compare_image_quantizers(
            native,
            reference=reference,
            author=author,
            tokens=tokens,
            token_types=types,
            patches=patches,
            image=image,
        )
    anchor = min(10, seq_len - 6)
    anchors = torch.tensor([anchor])
    initial_tokens = tokens[anchors]
    with torch.no_grad():
        reference.forward_spec(
            initial_tokens, reference_context[:, :anchor], start_pos=0
        )
        output_ids, reference_draft, reference_confidence = reference.forward_spec(
            initial_tokens, reference_context[:, anchor : anchor + 1], start_pos=anchor
        )
        draft_ids = torch.full((1, 5), config.drafter.noise_token_id)
        draft_ids[:, 0] = initial_tokens
        native_draft, native_confidence = native.forward_draft(
            native_context,
            native.tok_embeddings(draft_ids.flatten()),
            output_ids[:, :-1],
            anchors,
            main_positions_T=torch.arange(seq_len),
            output_weight_VD=native.lm_head.weight,
        )
    torch.testing.assert_close(native_draft, reference_draft, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        native_confidence, reference_confidence, rtol=2e-4, atol=2e-5
    )
    # A standalone SWA block also tests arbitrary incoming mHC coefficients.
    block = native.layers["0"]
    h = torch.randn(seq_len, config.hc_mult, config.dim, requires_grad=True)
    ref_h = h.detach().clone()[None].requires_grad_()
    pre = torch.rand(seq_len, config.hc_mult)
    empty = h.new_empty(0, 64)
    empty_indices = torch.empty(seq_len, 0, dtype=torch.int32)
    result, result_pre, *_ = block(
        h,
        pre,
        empty,
        empty,
        empty_indices,
        h.new_empty(seq_len, 0),
        empty_indices,
        positions=torch.arange(seq_len),
    )
    ref_result, ref_pre = reference.layers[0](ref_h, 0, pre[None], None)
    torch.testing.assert_close(result, ref_result[0], rtol=1e-5, atol=1e-6)
    objective = result.square().mean() + result_pre.square().mean()
    ref_objective = ref_result.square().mean() + ref_pre.square().mean()
    objective.backward()
    ref_objective.backward()
    assert h.grad is not None and ref_h.grad is not None
    torch.testing.assert_close(h.grad, ref_h.grad[0], rtol=1e-4, atol=2e-6)
    gradients = adapter.to_hf(
        {
            name: parameter.grad
            for name, parameter in native.named_parameters()
            if parameter.grad is not None
        }
    )
    reference_gradient_names = {
        name
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    assert gradients.keys() == reference_gradient_names, (
        gradients.keys() ^ reference_gradient_names
    )
    gradient_error = 0.0
    for name, actual_grad in gradients.items():
        expected_grad = reference.get_parameter(name).grad
        assert expected_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-6)
        gradient_error = max(
            gradient_error, (actual_grad - expected_grad).abs().max().item()
        )
    native.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    draft_gradients = compare_draft_gradients(
        native,
        reference=reference,
        adapter=adapter,
        features_TF=native_context,
        tokens_T=tokens,
        anchor=anchor,
    )
    for layer_id in config.engram_layer_ids:
        table = reference.layers[layer_id].engram.embed.weight
        assert table.dtype == torch.float32
        table.requires_grad_(True)
    native_embeddings, reference_embeddings = [], []

    def retain_embedding_output(destination):
        def hook(module, inputs, output):
            output.retain_grad()
            destination.append(output)

        return hook

    native_hook = native.tok_embeddings.register_forward_hook(
        retain_embedding_output(native_embeddings)
    )
    reference_hook = reference.embed.register_forward_hook(
        retain_embedding_output(reference_embeddings)
    )
    native.train()
    reference.train()
    labels = tokens.roll(-1)
    labels[-1] = -100
    loss_tokens = (labels != -100).sum()
    # Execute the unchanged prefill body without its inference-mode decorator.
    # Autograd preserves tensors saved before the inference cache's writes.
    with torch.autograd.graph.allow_mutation_on_saved_tensors():
        whole_actual = native(tokens)
        _, whole_expected, _ = type(reference).forward.__wrapped__(
            reference, tokens[None]
        )
        torch.testing.assert_close(
            whole_actual, whole_expected[0], rtol=2e-4, atol=2e-5
        )
        whole_loss = (
            F.cross_entropy(whole_actual.float(), labels, reduction="sum") / loss_tokens
        )
        reference_whole_loss = (
            F.cross_entropy(whole_expected[0].float(), labels, reduction="sum")
            / loss_tokens
        )
        torch.testing.assert_close(
            whole_loss, reference_whole_loss, rtol=2e-5, atol=2e-6
        )
        whole_loss.backward()
        reference_whole_loss.backward()
    native_hook.remove()
    reference_hook.remove()
    embedding_gradient = native_embeddings[0].grad
    reference_embedding_gradient = reference_embeddings[0].grad
    assert embedding_gradient is not None and reference_embedding_gradient is not None
    assert embedding_gradient.norm() > 0
    torch.testing.assert_close(
        embedding_gradient, reference_embedding_gradient[0], rtol=5e-4, atol=1e-8
    )
    whole_gradients = adapter.to_hf(
        {
            name: parameter.grad
            for name, parameter in native.named_parameters()
            if parameter.grad is not None
            and not name.startswith(
                ("vision.", "aligner.", "image_embeddings.", "drafter.")
            )
        }
    )
    reference_whole_gradient_names = {
        name
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
        and not name.startswith(
            ("vision.", "aligner.", "image_start", "image_newline", "image_end", "mtp.")
        )
    }
    assert whole_gradients.keys() == reference_whole_gradient_names, (
        whole_gradients.keys() ^ reference_whole_gradient_names
    )
    whole_gradient_error = 0.0
    for name, actual_grad in whole_gradients.items():
        expected_grad = reference.get_parameter(name).grad
        assert expected_grad is not None, name
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=5e-4, atol=2e-6, msg=name
        )
        whole_gradient_error = max(
            whole_gradient_error, (actual_grad - expected_grad).abs().max().item()
        )
    shared_gradient = native.layers["5"].attention.compressor.wkv.weight.grad
    assert shared_gradient is not None and shared_gradient.norm() > 0
    native_gradient_norm = (
        torch.stack(
            [gradient.double().square().sum() for gradient in whole_gradients.values()]
        )
        .sum()
        .sqrt()
    )
    reference_gradient_norm = (
        torch.stack(
            [
                reference.get_parameter(name).grad.double().square().sum()
                for name in whole_gradients
            ]
        )
        .sum()
        .sqrt()
    )
    return {
        **draft_gradients,
        **image_quantization,
        "revision": REVISION,
        "source_sha256": SOURCE_HASHES,
        "dtype": "float32",
        "seq_len": seq_len,
        "quantization": "FP8 SWA/Engram and FP4 global KV/indexer",
        "block_max_abs_error": errors,
        "first_block_component_max_abs_error": component_errors,
        "shared_kv_index_selection_max_abs_error": shared_errors,
        "vision_max_abs_error": (encoded - reference_encoded).abs().max().item(),
        "aligner_max_abs_error": (aligned - reference_aligned).abs().max().item(),
        "image_logit_max_abs_error": (image_actual - image_expected[0])
        .abs()
        .max()
        .item(),
        "draft_logit_max_abs_error": (native_draft - reference_draft)
        .abs()
        .max()
        .item(),
        "confidence_max_abs_error": (native_confidence - reference_confidence)
        .abs()
        .max()
        .item(),
        "logit_max_abs_error": (actual - expected.squeeze(0)).abs().max().item(),
        "reference_to_native_probability_kl": F.kl_div(
            actual.double().log_softmax(-1),
            expected.squeeze(0).double().log_softmax(-1),
            reduction="batchmean",
            log_target=True,
        ).item(),
        "logit_cosine_similarity": F.cosine_similarity(
            actual.double().flatten(), expected.squeeze(0).double().flatten(), dim=0
        ).item(),
        "top1_agreement": (actual.argmax(-1) == expected.squeeze(0).argmax(-1))
        .double()
        .mean()
        .item(),
        "top5_set_agreement": (
            actual.topk(5, dim=-1).indices.sort(-1).values
            == expected.squeeze(0).topk(5, dim=-1).indices.sort(-1).values
        )
        .all(-1)
        .double()
        .mean()
        .item(),
        "input_grad_max_abs_error": (h.grad - ref_h.grad[0]).abs().max().item(),
        "parameter_grad_max_abs_error": gradient_error,
        "num_parameter_gradients": len(gradients),
        "whole_model_input_grad_max_abs_error": (
            embedding_gradient - reference_embedding_gradient[0]
        )
        .abs()
        .max()
        .item(),
        "whole_model_parameter_grad_max_abs_error": whole_gradient_error,
        "num_whole_model_parameter_gradients": len(whole_gradients),
        "whole_model_objective": "train-mode normalized next-token cross entropy; indexer auxiliary loss disabled",
        "whole_model_loss_token_count": loss_tokens.item(),
        "whole_model_cross_entropy_loss": whole_loss.item(),
        "reference_whole_model_cross_entropy_loss": reference_whole_loss.item(),
        "whole_model_gradient_norm_ratio": (
            native_gradient_norm / reference_gradient_norm
        ).item(),
    }


def compare_checkpoint_manifest(index_path):
    config = build_model_config(
        "deepseek_v4_1_flash", parallelism=ParallelismConfig(), seq_len=128
    )
    with torch.device("meta"):
        model = config.build()
    keys = set(DeepSeekV41StateDictAdapter(config).to_hf(model.state_dict()))
    manifest = json.loads(index_path.read_text())["weight_map"]
    expected = {key for key in manifest if not key.endswith(".scale")}
    if keys != expected:
        raise AssertionError(
            f"Checkpoint keys differ: missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    engram_rows = []
    for index in config.engram_layer_ids:
        engram = config.layers[index].engram
        assert engram is not None
        engram_rows.append(engram.embed.num_embeddings)
    return {
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "num_weight_keys": len(keys),
        "num_manifest_keys": len(manifest),
        "engram_rows": engram_rows,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path)
    args = parser.parse_args()
    if args.seq_len < 16:
        parser.error("--seq-len must be at least 16 for image and draft comparisons.")
    result = compare(args.reference_dir, seq_len=args.seq_len)
    if args.checkpoint_index is not None:
        result["released_checkpoint_manifest"] = compare_checkpoint_manifest(
            args.checkpoint_index
        )
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
