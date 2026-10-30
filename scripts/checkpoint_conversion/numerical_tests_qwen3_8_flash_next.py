#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same-weight component, block, vision and full-logit Flash-Next parity.

Install transformers at 494470d677d05d44153d235d795e0d01040307ab, then run:
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next
The reduced fixture retains every backbone feature. Components and gradients
use FP32; the recurrent backbone uses BF16, as required by its shared kernel.
"""

import argparse
import copy
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)

REFERENCE_REVISION = "494470d677d05d44153d235d795e0d01040307ab"
CHECKPOINT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

# T = packed tokens, D = model dimension, C = residual streams * D,
# E = experts, K = selected experts, V = vocabulary.
BF16_EPS = torch.finfo(torch.bfloat16).eps


def reference_text_config() -> Any:
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


def run(*, device: str) -> dict[str, Any]:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpVisionConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpVisionModel,
    )

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    config = build_model_config(
        "debugmodel", seq_len=64, parallelism=ParallelismConfig()
    )
    vision_config = config.vision_encoder
    assert vision_config is not None
    config.vision_encoder = None
    config.vocab_size = (
        config.tok_embeddings.num_embeddings
    ) = config.lm_head.out_features = 64
    config.layers[1].ple.ple_embedding.vocab_size = 64
    config.layers[1].ple.ple_embedding.eos_token_id = 63
    hf_config = reference_text_config()
    hf_config._attn_implementation = "sdpa"
    hf = Qwen4ExpForCausalLM(hf_config).to(device).eval()
    native = config.build().to(device).eval()
    adapter = Qwen38FlashNextStateDictAdapter(config, None)
    native.load_state_dict(adapter.from_hf(hf.state_dict()), strict=True)
    positions = torch.arange(39, device=device)
    tokens = torch.randint(0, 63, (39,), device=device)
    tokens[7] = 63
    hidden = torch.randn(39, 1024, device=device, requires_grad=True)
    results: dict[str, Any] = {}

    def compare(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float = 2e-5,
        rtol: float = 2e-5,
    ) -> None:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
        difference = actual.detach().double() - expected.detach().double()
        results[name] = {
            "max_abs": float(difference.abs().max()),
            "rms": float(difference.square().mean().sqrt()),
            "relative_l2": float(difference.norm() / expected.detach().double().norm()),
            "atol": atol,
            "rtol": rtol,
        }

    def compare_gradients(
        name: str,
        actual_loss: torch.Tensor,
        expected_loss: torch.Tensor,
        input_TC: torch.Tensor,
        actual_module: torch.nn.Module,
        expected_module: torch.nn.Module,
    ) -> None:
        actual_parameters = dict(actual_module.named_parameters())
        expected_parameters = dict(expected_module.named_parameters())
        assert actual_parameters.keys() == expected_parameters.keys()
        actual_grads = torch.autograd.grad(
            actual_loss, (input_TC, *actual_parameters.values())
        )
        expected_grads = torch.autograd.grad(
            expected_loss, (input_TC, *expected_parameters.values())
        )
        for field, actual_grad, expected_grad in zip(
            ("input", *actual_parameters), actual_grads, expected_grads
        ):
            compare(
                f"{name}_gradient/{field}",
                actual_grad,
                expected_grad,
                atol=3e-6,
                rtol=3e-5,
            )

    # Norms and mixing gradients are checked before the recurrent stack.
    actual, _, write = native.layers["0"].attn_hyper_connection(hidden)
    expected, _, hf_write = hf.model.layers[0].attn_hyper_connection(hidden[None])
    compare("residual_read", actual, expected[0])
    compare("residual_write", write, hf_write[0])
    compare_gradients(
        "residual",
        actual.square().mean() + write.square().mean(),
        expected.square().mean() + hf_write.square().mean(),
        hidden,
        native.layers["0"].attn_hyper_connection,
        hf.model.layers[0].attn_hyper_connection,
    )
    ple = native.layers["1"].ple
    hf_ple = hf.model.layers[1].ple
    # Exercise convolution backwards with nonzero weights; released initialization is zero.
    with torch.no_grad():
        hf_ple.conv1d.weight.uniform_(-0.02, 0.02)
        ple.conv1d.weight.copy_(hf_ple.conv1d.weight)
    hidden = hidden.detach().requires_grad_()
    actual = ple(hidden, input_ids_T=tokens, document_positions_T=positions)
    expected = hf_ple(hidden[None], tokens[None], None)[0]
    compare("ple", actual, expected)
    compare_gradients(
        "ple", actual.square().mean(), expected.square().mean(), hidden, ple, hf_ple
    )
    x = torch.randn(39, 256, device=device)
    cos, sin = hf.model.rotary_emb(x[None], positions[None, None].expand(3, 1, -1))
    causal = torch.ones(39, 39, device=device, dtype=torch.bool).tril()[None, None]
    hf_mask = hf.model.layers[3].self_attn.indexer(x[None], (cos, sin), causal, None)[
        0, 0
    ]
    selected, _ = native.layers["3"].attention.indexer(
        x, positions=positions, document_positions_T=positions
    )
    mask = torch.zeros(39, 40, device=device, dtype=torch.bool).scatter(
        1, selected.long().masked_fill(selected < 0, 39), True
    )[:, :39]
    assert torch.equal(mask, hf_mask), "Indexer selection differs"
    results["indexer_selection"] = "exact"
    with sdpa_kernel(SDPBackend.MATH):
        compare(
            "qsa",
            native.layers["3"].attention(
                x, None, positions, document_positions_T=positions
            ),
            hf.model.layers[3].self_attn(x[None], (cos, sin), causal, None)[0][0],
        )
        # The router is tested independently from grouped expert computation.
        router_input = torch.randn(39, 256, device=device)
        hf_router_logits, hf_router_scores, hf_router_indices = hf.model.layers[
            3
        ].mlp.gate(router_input)
        native_logits = native.layers["3"].feed_forward.router.gate(router_input)
        hf_logits = torch.nn.functional.linear(
            router_input.float(), hf.model.layers[3].mlp.gate.weight.float()
        )
        compare("router_logits", native_logits, hf_logits)
        assert torch.equal(
            native_logits.topk(2, -1).indices.sort(-1).values,
            hf_router_indices.sort(-1).values,
        )
        native_router_scores, native_router_indices, _ = native.layers[
            "3"
        ].feed_forward.router(router_input)
        assert torch.equal(native_router_indices, hf_router_indices)
        compare("router_mixing_weights", native_router_scores, hf_router_scores)
        # The shared fused GDN kernel accepts only FP16/BF16 QKV. Check its
        # actual training dtype after the new components' FP32 comparisons.
        rotary = copy.deepcopy(hf.model.rotary_emb)
        native = native.bfloat16()
        hf = hf.bfloat16()
        hf.model.rotary_emb = rotary
        hidden = hidden.bfloat16()
        cos, sin = hf.model.rotary_emb(
            x.bfloat16()[None], positions[None, None].expand(3, 1, -1)
        )
        for layer_id in (0, 1, 3):
            hf_layer = hf.model.layers[layer_id]
            actual = native.layers[str(layer_id)](
                hidden.detach(),
                None,
                positions,
                ple_input_ids_T=tokens,
                document_positions_T=positions,
            )
            expected = hf_layer(
                hidden.detach()[None],
                position_embeddings=(cos, sin),
                attention_mask=causal if layer_id == 3 else None,
                ple_input_ids=tokens[None],
                past_key_values=None,
            )[0]
            compare(
                f"block_{layer_id}", actual, expected, atol=BF16_EPS / 2, rtol=BF16_EPS
            )
        with torch.no_grad():
            native_states = {}
            hf_states = {}
            native_routes = {}
            hf_routes = {}
            handles = []
            for layer_id in range(8):
                handles.append(
                    native.layers[str(layer_id)].register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_states.__setitem__(
                            i, output.detach()
                        )
                    )
                )
                handles.append(
                    hf.model.layers[layer_id].register_forward_hook(
                        lambda module, inputs, output, i=layer_id: hf_states.__setitem__(
                            i, output[0].detach()
                        )
                    )
                )
                handles.append(
                    native.layers[
                        str(layer_id)
                    ].feed_forward.router.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_routes.__setitem__(
                            i, tuple(value.detach() for value in output[:2])
                        )
                    )
                )
                handles.append(
                    hf.model.layers[layer_id].mlp.gate.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: hf_routes.__setitem__(
                            i, tuple(value.detach() for value in output)
                        )
                    )
                )
            logits = native(tokens, positions=positions)
            hf_logits = hf(input_ids=tokens[None], use_cache=False).logits[0]
            for handle in handles:
                handle.remove()
            for layer_id in range(8):
                compare(
                    f"stack_block_{layer_id}",
                    native_states[layer_id],
                    hf_states[layer_id],
                    atol=BF16_EPS / 2,
                    rtol=BF16_EPS,
                )
                native_scores_TK, native_ids_TK = native_routes[layer_id]
                hf_logits_TE, hf_scores_TK, hf_ids_TK = hf_routes[layer_id]
                assert torch.equal(
                    native_ids_TK.sort(-1).values, hf_ids_TK.sort(-1).values
                ), f"Layer {layer_id} router IDs differ"
                # Compare scores in the same expert order, including exact ties.
                native_order_TK = native_ids_TK.argsort(-1)
                hf_order_TK = hf_ids_TK.argsort(-1)
                compare(
                    f"stack_router_{layer_id}",
                    native_scores_TK.gather(-1, native_order_TK),
                    hf_scores_TK.gather(-1, hf_order_TK),
                    atol=BF16_EPS / 2,
                    rtol=BF16_EPS,
                )
                sorted_logits_TE = hf_logits_TE.float().sort(-1, descending=True).values
                results[f"stack_router_margin_{layer_id}"] = float(
                    (sorted_logits_TE[:, 1] - sorted_logits_TE[:, 2]).min()
                )
            compare("logits", logits, hf_logits, atol=BF16_EPS / 2, rtol=BF16_EPS)
            native_logprobs_TV = logits.double().log_softmax(-1)
            hf_logprobs_TV = hf_logits.double().log_softmax(-1)
            hf_probs_TV = hf_logprobs_TV.exp()
            native_top_TK = logits.topk(5, -1).indices
            hf_top_TK = hf_logits.topk(5, -1).indices
            results["logit_probability_metrics"] = {
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        logits.double().flatten(), hf_logits.double().flatten(), dim=0
                    )
                ),
                "mean_kl_reference_to_native": float(
                    (hf_probs_TV * (hf_logprobs_TV - native_logprobs_TV)).sum(-1).mean()
                ),
                "top1_agreement": float(
                    (native_top_TK[:, 0] == hf_top_TK[:, 0]).double().mean()
                ),
                "top5_set_overlap": float(
                    (native_top_TK.unsqueeze(-1) == hf_top_TK.unsqueeze(-2))
                    .any(-1)
                    .double()
                    .mean()
                ),
            }
    # Packed boundaries use the same weights and are compared with isolated documents.
    packed_positions = torch.cat((torch.arange(19), torch.arange(20))).to(device)
    metadata = native._get_attention_metadata(positions=packed_positions)
    with torch.no_grad():
        routes = {layer_id: [] for layer_id in range(8)}
        handles = [
            native.layers[str(layer_id)].feed_forward.router.register_forward_hook(
                lambda module, inputs, output, i=layer_id: routes[i].append(
                    output[1].detach()
                )
            )
            for layer_id in range(8)
        ]
        packed = native(tokens, positions=packed_positions, attention_metadata=metadata)
        isolated = torch.cat(
            [
                native(tokens[:19], positions=positions[:19]),
                native(tokens[19:], positions=positions[:20]),
            ]
        )
        for handle in handles:
            handle.remove()
        isolated_routes = {
            layer_id: torch.cat(values[1:]) for layer_id, values in routes.items()
        }
        changed_tokens = {
            layer_id: int(
                (values[0].sort(-1).values != isolated_routes[layer_id].sort(-1).values)
                .any(-1)
                .sum()
            )
            for layer_id, values in routes.items()
        }
        difference = packed.float() - isolated.float()
        results["packed_documents_unimposed"] = {
            "max_abs": float(difference.abs().max()),
            "rms": float(difference.square().mean().sqrt()),
            "changed_router_tokens_by_layer": changed_tokens,
        }
        # Packed GDN and GEMMs can round differently from isolated invocations.
        # Replay only the isolated expert IDs; recompute all router mixing weights
        # from the packed activations to verify boundaries through the full stack.
        with ExitStack() as stack:
            for layer_id, indices in isolated_routes.items():
                stack.enter_context(
                    patch.object(
                        native.layers[str(layer_id)].feed_forward.router,
                        "_select_experts",
                        lambda scores, expert_bias=None, ids=indices, **kwargs: ids,
                    )
                )
            packed = native(
                tokens, positions=packed_positions, attention_metadata=metadata
            )
        compare(
            "packed_documents_imposed_ids",
            packed,
            isolated,
            atol=BF16_EPS / 2,
            rtol=BF16_EPS,
        )
    hf_vision_cfg = Qwen4ExpVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        depth=2,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=1024,
    )
    hf_vision_cfg._attn_implementation = "sdpa"
    hf_vision = Qwen4ExpVisionModel(hf_vision_cfg).to(device).eval()
    vision = vision_config.build().to(device).eval()
    vision_model_config = copy.deepcopy(config)
    vision_model_config.vision_encoder = vision_config
    vision_adapter = Qwen38FlashNextStateDictAdapter(vision_model_config, None)
    vision_state = {
        "model.visual." + key: value for key, value in hf_vision.state_dict().items()
    }
    vision_state.update(
        {
            "lm_head.weight": hf.lm_head.weight,
            "model.language_model.embed_tokens.weight": hf.model.embed_tokens.weight,
        }
    )
    mapped = vision_adapter.from_hf(vision_state)
    vision.load_state_dict(
        {
            key.removeprefix("vision_encoder."): value
            for key, value in mapped.items()
            if key.startswith("vision_encoder.")
        },
        strict=True,
    )
    grid = torch.tensor([[1, 4, 4], [2, 2, 2]], device=device)
    patches = torch.randn(24, 3 * 2 * 16 * 16, device=device)
    with torch.no_grad():
        compare(
            "vision",
            vision(patches, grid_thw=grid),
            hf_vision(patches, grid_thw=grid).pooler_output,
            atol=2e-5,
            rtol=2e-5,
        )
    import transformers

    return {
        "reference_revision": REFERENCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "device": device,
        "torch": torch.__version__,
        "component_dtype": "float32",
        "torch_revision": torch.version.git_version,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda,
        "backbone_dtype": "bfloat16",
        "seed": 42,
        "attention_backend": "SDPA MATH",
        "tolerance_rationale": (
            "FP32 operation-order error; BF16 half-epsilon absolute plus one epsilon relative. "
            "Packed expert-ID replay is diagnostic only; unconstrained routing differences "
            "are reported separately."
        ),
        "tokens": tokens.tolist(),
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
