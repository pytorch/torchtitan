# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Router replay through a whole decoder: every MoE router uses the given expert ids."""

import importlib
import unittest

import torch

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.qwen3 import build_model_config

_NUM_TOKENS = 64


def _build(seed: int = 0):
    config = build_model_config("debugmodel_moe", attn_backend="varlen")
    config.local_compile_regions = []
    torch.manual_seed(seed)
    with torch.device("cuda"):
        model = config.build()
        model.init_states(buffer_device=torch.device("cuda"))
    model.train()
    return model


def _forward(model, tokens, **kwargs):
    positions = torch.arange(tokens.shape[0], device=tokens.device)
    return model(tokens, positions, model._get_attention_metadata(positions), **kwargs)


def _own_routing(model, tokens) -> torch.Tensor:
    """``[T, num_layers, top_k]`` uint8 ids each router picks on its own."""
    routing = {}
    hooks = [
        module.register_forward_hook(
            lambda router, args, output, layer_id=int(layer_name): routing.__setitem__(
                layer_id, output[1]
            )
        )
        for layer_name, layer in model.layers.items()
        for module in layer.modules()
        if isinstance(module, TokenChoiceTopKRouter)
    ]
    with torch.no_grad():
        _forward(model, tokens)
    for hook in hooks:
        hook.remove()
    return torch.stack([routing[i] for i in sorted(routing)], dim=1).to(torch.uint8)


def _random_routing(model) -> torch.Tensor:
    num_layers, top_k, num_experts = len(model.layers), 8, 64
    return (
        torch.stack(
            [
                torch.randperm(num_experts, device="cuda")[:top_k]
                for _ in range(_NUM_TOKENS * num_layers)
            ]
        )
        .view(_NUM_TOKENS, num_layers, top_k)
        .to(torch.uint8)
    )


def _loss_and_grads(model, tokens, **kwargs):
    model.zero_grad()
    logits = _forward(model, tokens, **kwargs)
    logits.float().square().mean().backward()
    return logits.detach(), {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRouterReplay(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokens = torch.randint(0, 2048, (_NUM_TOKENS,), device="cuda")

    def test_replaying_own_routing_is_bitwise_identical(self):
        model = _build()
        routed_expert_ids = _own_routing(model, self.tokens)

        logits, grads = _loss_and_grads(model, self.tokens)
        replay_logits, replay_grads = _loss_and_grads(
            model, self.tokens, routed_expert_ids=routed_expert_ids
        )

        self.assertTrue(torch.equal(logits, replay_logits))
        self.assertEqual(grads.keys(), replay_grads.keys())
        for name, grad in grads.items():
            self.assertTrue(torch.equal(grad, replay_grads[name]), name)
        for layer in model.layers.values():
            counts = layer.moe.router.routing_mismatch_counts
            self.assertEqual(counts[1:].tolist(), [0.0, 0.0])

    def test_every_router_uses_the_replayed_experts(self):
        model = _build()
        routed_expert_ids = _random_routing(model)

        used = {}
        hooks = [
            layer.moe.router.register_forward_hook(
                lambda router, args, output, layer_id=int(layer_name): used.__setitem__(
                    layer_id, output[1]
                )
            )
            for layer_name, layer in model.layers.items()
        ]
        with torch.no_grad():
            _forward(model, self.tokens, routed_expert_ids=routed_expert_ids)
        for hook in hooks:
            hook.remove()

        for layer_id in range(len(model.layers)):
            self.assertTrue(
                torch.equal(used[layer_id], routed_expert_ids[:, layer_id].long())
            )

    def test_activation_checkpointing_recomputes_with_the_replayed_experts(self):
        routed_expert_ids = _random_routing(_build())
        logits, grads = _loss_and_grads(
            _build(), self.tokens, routed_expert_ids=routed_expert_ids
        )

        for ac_config in (
            FullAC.Config(),
            SelectiveAC.Config(),
            RegionAC.Config(save_regions=[]),
            RegionAC.Config(save_regions=["moe.router.routing_decision"]),
        ):
            with self.subTest(ac=ac_config):
                model = _build()
                ac_config.build().apply(model)
                ac_logits, ac_grads = _loss_and_grads(
                    model, self.tokens, routed_expert_ids=routed_expert_ids
                )
                self.assertTrue(torch.equal(logits, ac_logits))
                self.assertEqual(grads.keys(), ac_grads.keys())
                for name, grad in grads.items():
                    self.assertTrue(torch.equal(grad, ac_grads[name]), name)

    def test_every_moe_family_threads_routed_expert_ids_to_every_router(self):
        # (family, flavor, attention backend); Kimi K2.7's QK clipping needs flex.
        families = [
            ("qwen3", "debugmodel_moe", "varlen"),
            ("qwen3_5", "debugmodel_moe", "varlen"),
            ("gpt_oss", "debugmodel", "varlen"),
            ("deepseek_v3", "debugmodel", "varlen"),
            ("deepseek_v4", "debugmodel", None),
            ("kimi_k2_7", "debugmodel", "flex"),
            ("kimi_k3", "debugmodel", "varlen"),
        ]
        for family, flavor, attn_backend in families:
            with self.subTest(family=family):
                kwargs = {} if attn_backend is None else {"attn_backend": attn_backend}
                config = importlib.import_module(
                    f"torchtitan.models.{family}"
                ).build_model_config(flavor, **kwargs)
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
                model = model.bfloat16().train()
                tokens = torch.randint(
                    0, config.vocab_size, (_NUM_TOKENS,), device="cuda"
                )
                positions = torch.arange(_NUM_TOKENS, device="cuda")

                def run(**replay_kwargs):
                    with torch.no_grad():
                        output = model(
                            tokens,
                            positions=positions,
                            attention_metadata=model._get_attention_metadata(positions),
                            aux_loss_denominators=torch.tensor(
                                [_NUM_TOKENS], device="cuda"
                            ),
                            **replay_kwargs,
                        )
                    return output[0] if isinstance(output, tuple) else output

                own, received = {}, {}

                def record(router, args, kwargs, output):
                    own[router.test_layer_id] = output[1]
                    received[router.test_layer_id] = kwargs.get("routed_expert_ids_TK")

                hooks = []
                for layer_name, layer in model.layers.named_children():
                    for module in layer.modules():
                        if isinstance(module, TokenChoiceTopKRouter):
                            module.test_layer_id = int(layer_name)
                            hooks.append(
                                module.register_forward_hook(record, with_kwargs=True)
                            )
                logits = run()
                routed_expert_ids = torch.zeros(
                    _NUM_TOKENS,
                    len(config.layers),
                    own[next(iter(own))].shape[-1],
                    dtype=torch.int16,
                    device="cuda",
                )
                for layer_id, ids in own.items():
                    routed_expert_ids[:, layer_id] = ids.to(torch.int16)
                replay_logits = run(routed_expert_ids=routed_expert_ids)
                for hook in hooks:
                    hook.remove()

                self.assertTrue(own)
                self.assertTrue(all(ids is not None for ids in received.values()))
                self.assertTrue(torch.equal(logits, replay_logits))


if __name__ == "__main__":
    unittest.main()
