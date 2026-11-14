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
from typing import Any

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


def _compare_tensor(name, actual, expected, *, atol, rtol) -> float:
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{name}: {message}"
    )
    return float((actual.detach().double() - expected.detach().double()).abs().max())


def _compare_gradient(name, actual, expected, *, atol, rtol, bfloat16_compute):
    if not bfloat16_compute:
        _compare_tensor(name, actual, expected, atol=atol, rtol=rtol)
        return
    # BF16 packed and separate projections round gradient sums differently.
    # Keep the existing RMS and infinity-norm bounds, including near-zero values.
    reference = expected.detach().double()
    difference = actual.detach().double() - reference
    assert difference.square().mean().sqrt() <= (
        atol + rtol * reference.square().mean().sqrt()
    ), name
    assert difference.abs().max() <= atol + rtol * reference.abs().max(), name


def _compare_parameter_gradients(name, native, expected_gradients, *, bfloat16_compute):
    atol, rtol = (GRADIENT_ATOL, GRADIENT_RTOL) if bfloat16_compute else (3e-6, 3e-5)
    actual_norm_squared = expected_norm_squared = gradient_dot = 0.0
    count = 0
    for key, parameter in native.named_parameters():
        actual = parameter.grad
        expected = expected_gradients.pop(key, None)
        assert (actual is None) == (
            expected is None
        ), f"{name}/{key}: gradient presence"
        if actual is None:
            continue
        _compare_gradient(
            f"{name}/{key}",
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            bfloat16_compute=bfloat16_compute,
        )
        actual_norm_squared += float(actual.double().norm().square())
        expected_norm_squared += float(expected.double().norm().square())
        gradient_dot += float(
            torch.dot(actual.double().flatten(), expected.double().flatten())
        )
        count += 1
    assert (
        not expected_gradients
    ), f"{name}: unmapped gradients: {list(expected_gradients)}"
    actual_norm, expected_norm = (
        actual_norm_squared**0.5,
        expected_norm_squared**0.5,
    )
    torch.testing.assert_close(actual_norm, expected_norm, atol=atol, rtol=rtol)
    cosine = gradient_dot / (actual_norm * expected_norm)
    assert cosine >= 1 - rtol**2, name
    return {
        "parameter_tensors": count,
        "actual_grad_norm": actual_norm,
        "reference_grad_norm": expected_norm,
        "cosine": cosine,
    }


def _check_decoder(
    name,
    native,
    reference,
    reference_decoder,
    adapter,
    tokens_T,
    labels_T,
    *,
    native_kwargs,
    reference_kwargs,
    placeholder_T=None,
    patch_pairs=(),
):
    native.train().zero_grad(set_to_none=True)
    reference.train().zero_grad(set_to_none=True)
    native_routes, reference_routes = {}, {}
    native_indices, reference_masks = {}, {}
    embedding_gradients = {}
    with ExitStack() as hooks:
        for slot, embedding in enumerate(
            (native.tok_embeddings, reference_decoder.embed_tokens)
        ):

            def capture_embedding_gradient(module, inputs, output, i=slot):
                # Observe the original embedding node before visual scatter mutates it.
                output.register_hook(
                    lambda grad: embedding_gradients.__setitem__(i, grad.detach())
                )

            hooks.callback(
                embedding.register_forward_hook(capture_embedding_gradient).remove
            )
        for layer_id, reference_layer in enumerate(reference_decoder.layers):
            native_layer = native.layers[str(layer_id)]
            hooks.callback(
                native_layer.feed_forward.router.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: native_routes.__setitem__(
                        i, output[:2]
                    )
                ).remove
            )
            hooks.callback(
                reference_layer.mlp.gate.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: reference_routes.__setitem__(
                        i, output[1:]
                    )
                ).remove
            )
            if layer_id in INDEXER_SEEDS:
                hooks.callback(
                    native_layer.attention.indexer.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_indices.__setitem__(
                            i, output[0]
                        )
                    ).remove
                )
                hooks.callback(
                    reference_layer.self_attn.indexer.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: reference_masks.__setitem__(
                            i, output[0, 0]
                        )
                    ).remove
                )
        logits_TV = native(tokens_T, **native_kwargs)
        reference_output = reference(
            input_ids=tokens_T[None], labels=labels_T[None], **reference_kwargs
        )
        loss = F.cross_entropy(logits_TV[:-1].float(), labels_T[1:], ignore_index=-100)
        loss.backward()
        reference_output.loss.backward()

    max_error = _compare_tensor(
        f"{name}/logits",
        logits_TV,
        reference_output.logits[0],
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    _compare_tensor(f"{name}/loss", loss, reference_output.loss, atol=2e-4, rtol=2e-4)
    _compare_gradient(
        f"{name}/embedding_gradient",
        embedding_gradients[0],
        embedding_gradients[1][0],
        atol=GRADIENT_ATOL,
        rtol=GRADIENT_RTOL,
        bfloat16_compute=True,
    )
    for layer_id, (scores_TK, ids_TK) in native_routes.items():
        reference_scores_TK, reference_ids_TK = reference_routes[layer_id]
        assert torch.equal(
            ids_TK.sort(-1).values, reference_ids_TK.sort(-1).values
        ), f"{name}/router_{layer_id}"
        _compare_tensor(
            f"{name}/router_{layer_id}",
            scores_TK.gather(-1, ids_TK.argsort(-1)),
            reference_scores_TK.gather(-1, reference_ids_TK.argsort(-1)),
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
        )
    for layer_id, indices_TJ in native_indices.items():
        num_tokens = tokens_T.numel()
        mask_TT = torch.zeros(
            num_tokens, num_tokens + 1, dtype=torch.bool, device=tokens_T.device
        )
        mask_TT.scatter_(
            -1, indices_TJ.long().masked_fill(indices_TJ < 0, num_tokens), True
        )
        assert torch.equal(
            mask_TT[:, :num_tokens], reference_masks[layer_id]
        ), f"{name}/qsa_{layer_id}"
    if placeholder_T is not None:
        assert not torch.count_nonzero(embedding_gradients[0][placeholder_T])
        assert not torch.count_nonzero(embedding_gradients[1][0, placeholder_T])
    for field, actual, expected in patch_pairs:
        assert actual.grad is not None and actual.grad.norm() > 0, field
        _compare_gradient(
            f"{name}/{field}",
            actual.grad,
            expected.grad,
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
            bfloat16_compute=True,
        )
    expected_gradients = adapter.from_hf(
        {
            key: parameter.grad
            for key, parameter in reference.named_parameters()
            if parameter.grad is not None
        }
    )
    return {
        "native_ce": float(loss.detach()),
        "reference_ce": float(reference_output.loss.detach()),
        "max_logit_error": max_error,
        "gradients": _compare_parameter_gradients(
            name,
            native,
            expected_gradients,
            bfloat16_compute=True,
        ),
        "routing": "expert IDs and selected-token masks match exactly",
    }


def _check_vision(fixture, device):
    fixture.vision.train().zero_grad(set_to_none=True)
    fixture.reference_vision.train().zero_grad(set_to_none=True)
    patches_PF = fixture.patches_PF.clone().requires_grad_()
    reference_PF = fixture.patches_PF.clone().requires_grad_()
    grid_NA = torch.tensor([[1, 4, 4], [2, 2, 2]], device=device)
    output_TD = fixture.vision(patches_PF, grid_thw=grid_NA)
    reference_TD = fixture.reference_vision(
        reference_PF, grid_thw=grid_NA
    ).pooler_output
    output_TD.square().mean().backward()
    reference_TD.square().mean().backward()
    max_error = _compare_tensor(
        "vision/output", output_TD, reference_TD, atol=2e-5, rtol=2e-5
    )
    _compare_gradient(
        "vision/input_gradient",
        patches_PF.grad,
        reference_PF.grad,
        atol=3e-6,
        rtol=3e-5,
        bfloat16_compute=False,
    )
    source = {
        "model.visual." + key: parameter.grad
        for key, parameter in fixture.reference_vision.named_parameters()
        if parameter.grad is not None
    }
    # The adapter requires an LM-head entry when mapping a partial vision state.
    source["lm_head.weight"] = fixture.reference.lm_head.weight.grad
    expected = fixture.vision_adapter.from_hf(source)
    expected.pop("lm_head.weight")
    return {
        "max_output_error": max_error,
        "gradients": _compare_parameter_gradients(
            "vision",
            fixture.vision,
            {
                key.removeprefix("vision_encoder."): value
                for key, value in expected.items()
            },
            bfloat16_compute=False,
        ),
    }


def run(*, device: str = "cuda") -> dict[str, Any]:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForConditionalGeneration,
    )

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("HF parity requires the production CUDA GDN/grouped experts")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        fixture = _build_fixture(target_device)
        results = {
            "text": _check_decoder(
                "text",
                fixture.native,
                fixture.reference,
                fixture.reference.model,
                fixture.adapter,
                fixture.tokens_T,
                fixture.tokens_T,
                native_kwargs={
                    "positions": fixture.positions_T,
                    "document_positions": fixture.positions_T,
                    "ple_input_ids": fixture.tokens_T,
                },
                reference_kwargs={"use_cache": False},
            ),
            "vision": _check_vision(fixture, target_device),
        }
        fixture.native.vision_encoder = fixture.vision
        config = Qwen4ExpConfig(
            text_config=fixture.reference.config,
            vision_config=fixture.reference_vision.config,
            image_token_id=61,
            video_token_id=62,
            vision_start_token_id=60,
            vision_end_token_id=59,
        )
        torch.manual_seed(VISION_SEED)
        reference = Qwen4ExpForConditionalGeneration(config)
        reference.model.language_model = fixture.reference.model
        reference.model.visual = fixture.reference_vision
        reference.lm_head = fixture.reference.lm_head
        with torch.no_grad():
            for layer_id, seed in MM_ROUTER_SEEDS.items():
                weight_ED = reference.model.language_model.layers[
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
        mapped = fixture.vision_adapter.from_hf(reference.state_dict())
        native_state = fixture.native.state_dict()
        assert native_state.keys() == mapped.keys()
        assert all(torch.equal(native_state[key], mapped[key]) for key in native_state)
        tokens_T = torch.tensor(
            [3, 60, 61, 61, 61, 61, 59, 4, 60, 62, 59, 5, 6, 7, 8, 9],
            device=target_device,
        )
        image_grid_NA = torch.tensor([[1, 4, 4]], device=target_device)
        video_grid_NA = torch.tensor([[1, 2, 2]], device=target_device)
        mm_types_T = torch.zeros_like(tokens_T)
        mm_types_T[tokens_T == 61] = 1
        mm_types_T[tokens_T == 62] = 2
        positions_ABT, _ = reference.model.get_rope_index(
            tokens_T[None], mm_types_T[None], image_grid_NA, video_grid_NA
        )
        document_positions_T = torch.arange(tokens_T.numel(), device=target_device)
        positions_TA = positions_ABT[:, 0].transpose(0, 1).contiguous()
        assert not torch.equal(positions_TA[:, 0], document_positions_T)
        placeholder_T = (tokens_T == 61) | (tokens_T == 62)
        generator = torch.Generator(device="cpu").manual_seed(45)
        image_PF = torch.randn((16, 1536), generator=generator).to(target_device)
        video_PF = torch.randn((4, 1536), generator=generator).to(target_device)
        native_image_PF, reference_image_PF = (
            image_PF.clone().requires_grad_() for _ in range(2)
        )
        native_video_PF, reference_video_PF = (
            video_PF.clone().requires_grad_() for _ in range(2)
        )
        results["multimodal"] = _check_decoder(
            "multimodal",
            fixture.native,
            reference,
            reference.model.language_model,
            fixture.vision_adapter,
            tokens_T,
            tokens_T.masked_fill(placeholder_T, -100),
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
                "position_ids": positions_ABT,
                "pixel_values": reference_image_PF,
                "pixel_values_videos": reference_video_PF,
                "image_grid_thw": image_grid_NA,
                "video_grid_thw": video_grid_NA,
                "mm_token_type_ids": mm_types_T[None],
                "use_cache": False,
            },
            placeholder_T=placeholder_T,
            patch_pairs=(
                ("image", native_image_PF, reference_image_PF),
                ("video", native_video_PF, reference_video_PF),
            ),
        )
    return {
        "reference_revision": REFERENCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "device": torch.cuda.get_device_name(target_device),
        "seed": 42,
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
