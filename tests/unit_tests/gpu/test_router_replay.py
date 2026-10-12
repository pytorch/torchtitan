# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Router replay through whole decoders: replaying a model's own routing changes nothing."""

import importlib
import unittest

import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.models.common.moe import TokenChoiceTopKRouter

_NUM_TOKENS = 64


def _build(family: str, flavor: str, attn_backend: str | None):
    kwargs = {"parallelism": ParallelismConfig()}
    if attn_backend is not None:
        kwargs["attn_backend"] = attn_backend
    config = importlib.import_module(f"torchtitan.models.{family}").build_model_config(
        flavor, **kwargs
    )
    config.local_compile_regions = []
    if hasattr(config, "vision_encoder"):
        config.vision_encoder = None
    if hasattr(config, "mtp_layers"):
        config.mtp_layers = []
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = config.build()
        model.init_states(buffer_device=torch.device("cuda"))
    # Attention Gym's linear-attention kernels need bf16 weights.
    return model.bfloat16().train(), config.vocab_size


def _forward(model, tokens, **kwargs):
    positions = torch.arange(len(tokens), device="cuda")
    output = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
        aux_loss_denominators=torch.tensor([len(tokens)], device="cuda"),
        **kwargs,
    )
    return output[0] if isinstance(output, tuple) else output


def _own_routing(model, tokens) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    """``[T, num_layers, top_k]`` ids the routers pick on their own, and the
    ``routed_expert_ids_TK`` each router received."""
    own, received = {}, []

    def record(router, args, kwargs, output):
        own[router.replay_test_layer_id] = output[1]
        received.append(kwargs.get("routed_expert_ids_TK"))

    hooks = []
    for layer_name, layer in model.layers.items():
        for module in layer.modules():
            if isinstance(module, TokenChoiceTopKRouter):
                module.replay_test_layer_id = int(layer_name)
                hooks.append(module.register_forward_hook(record, with_kwargs=True))
    with torch.no_grad():
        _forward(model, tokens)
    routed_expert_ids = torch.zeros(
        len(tokens), len(model.layers), own[next(iter(own))].shape[-1], device="cuda"
    ).to(torch.int16)
    for layer_id, ids in own.items():
        routed_expert_ids[:, layer_id] = ids.to(torch.int16)
    received.clear()
    with torch.no_grad():
        _forward(model, tokens, routed_expert_ids=routed_expert_ids)
    for hook in hooks:
        hook.remove()
    return routed_expert_ids, received


def _loss_and_grads(model, tokens, **kwargs):
    model.zero_grad()
    logits = _forward(model, tokens, **kwargs)
    logits.float().square().mean().backward()
    grads = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    return logits.detach(), grads


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRouterReplay(unittest.TestCase):
    def assert_same(self, expected, actual):
        (logits, grads), (other_logits, other_grads) = expected, actual
        self.assertTrue(torch.equal(logits, other_logits))
        self.assertEqual(grads.keys(), other_grads.keys())
        for name, grad in grads.items():
            self.assertTrue(torch.equal(grad, other_grads[name]), name)

    def test_replaying_own_routing_is_bitwise_identical_with_any_ac(self):
        model, vocab_size = _build("qwen3", "debugmodel_moe", "varlen")
        tokens = torch.randint(0, vocab_size, (_NUM_TOKENS,), device="cuda")
        routed_expert_ids, _ = _own_routing(model, tokens)
        expected = _loss_and_grads(model, tokens)

        self.assert_same(
            expected,
            _loss_and_grads(model, tokens, routed_expert_ids=routed_expert_ids),
        )
        for ac_config in (
            FullAC.Config(),
            SelectiveAC.Config(),
            RegionAC.Config(save_regions=["moe.router.routing_decision"]),
        ):
            with self.subTest(ac=ac_config):
                ac_model, _ = _build("qwen3", "debugmodel_moe", "varlen")
                ac_config.build().apply(ac_model)
                self.assert_same(
                    expected,
                    _loss_and_grads(
                        ac_model, tokens, routed_expert_ids=routed_expert_ids
                    ),
                )

    def test_every_moe_family_passes_the_ids_to_every_router(self):
        # (family, flavor, attention backend); Kimi K2.7's QK clipping needs flex.
        for family, flavor, attn_backend in [
            ("qwen3", "debugmodel_moe", "varlen"),
            ("qwen3_5", "debugmodel_moe", "varlen"),
            ("gpt_oss", "debugmodel", "varlen"),
            ("deepseek_v3", "debugmodel", "varlen"),
            ("deepseek_v4", "debugmodel", None),
            ("kimi_k2_7", "debugmodel", "flex"),
            ("kimi_k3", "debugmodel", "varlen"),
        ]:
            with self.subTest(family=family):
                model, vocab_size = _build(family, flavor, attn_backend)
                tokens = torch.randint(0, vocab_size, (_NUM_TOKENS,), device="cuda")
                routed_expert_ids, received = _own_routing(model, tokens)

                self.assertTrue(received)
                self.assertTrue(all(ids is not None for ids in received))
                with torch.no_grad():
                    self.assertTrue(
                        torch.equal(
                            _forward(model, tokens),
                            _forward(
                                model, tokens, routed_expert_ids=routed_expert_ids
                            ),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
