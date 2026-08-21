# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Will vLLM's loader accept the names we export? (veRL weight sync.)

A veRL run syncs trainer weights into a rollout engine. vLLM's ``load_weights``
consumes HF CHECKPOINT names and remaps them to its own internal modules itself,
so what we owe it is checkpoint naming -- which ``hf_key_map.titan_to_official``
already produces. This pins the two conventions its loader keys on, both read off
``vllm/model_executor/models/kimi_linear.py``:

* routed experts are matched by the checkpoint substrings ``w1`` (gate), ``w2``
  (down), ``w3`` (up) -- ``fused_moe_make_expert_params_mapping(...,
  ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2", ckpt_up_proj_name="w3")``;
* the dense FFN must arrive UNFUSED as ``.gate_proj`` / ``.up_proj``, because
  vLLM's ``stacked_params_mapping`` fuses them into its own ``.gate_up_proj``:
  ``(".gate_up_proj", ".gate_proj", 0), (".gate_up_proj", ".up_proj", 1)``.
  Exporting a pre-fused name would simply never match.

Confirmed against vLLM's REAL K3 implementation, not just the predecessor. K3
support is vllm-project/vllm PR #50000 (branch ``kimi-k3``, open and conflicting
with main as of 2026-07-28), which ships the model under a separate top-level
package ``vllm.models.kimi_k3`` with ``nvidia/`` and ``amd/`` variants. Both use
the same conventions this test pins::

    vllm/models/kimi_k3/nvidia/model.py:1203  ckpt_gate_proj_name="w1"
    vllm/models/kimi_k3/amd/mtp.py:246        ckpt_gate_proj_name="w1"
                                              ckpt_down_proj_name="w2"

and its modules carry the same checkpoint-facing names our map emits, e.g.
``routed_expert_down_proj`` (amd/linear.py:230). The branch also registers
``KimiK3MTPModel``, so speculative decoding has its own weight surface to check
when we get there.

Note the registry entry points at ``vllm.models.kimi_k3`` rather than
``vllm.model_executor.models.*``, so a future check must look there.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.hf_key_map import titan_to_official
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

# vLLM's kimi_linear.py, verbatim.
VLLM_CKPT_EXPERT_NAMES = {"gate": "w1", "down": "w2", "up": "w3"}
VLLM_FUSES_INTO = ".gate_up_proj"
VLLM_FUSES_FROM = (".gate_proj", ".up_proj")

_KDA_1BASED_FULL = {4, 8, 12, 16, 20, 21}


def _exported_names():
    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    with torch.device("meta"):
        model = KimiK3Model.make_config(cfg).build()
    kda = {
        i
        for i in range(cfg.num_hidden_layers)
        if (i + 1) not in set(cfg.full_attn_layers)
    }
    out = {}
    for name, _ in model.named_parameters():
        if "inner_experts" in name:
            for e in range(cfg.num_experts):
                out[titan_to_official(name, kda_layers=kda, expert_idx=e)] = name
        else:
            out[titan_to_official(name, kda_layers=kda)] = name
    return out, cfg


class TestVLLMWeightContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exported, cls.cfg = _exported_names()

    def test_routed_experts_use_the_names_vllm_matches_on(self):
        expert_keys = [k for k in self.exported if ".experts." in k]
        self.assertTrue(expert_keys)
        for k in expert_keys:
            leaf = k.rsplit(".", 2)[-2]  # ...experts.{e}.{leaf}.weight
            self.assertIn(
                leaf,
                set(VLLM_CKPT_EXPERT_NAMES.values()),
                f"{k}: vLLM matches experts on w1/w2/w3, not {leaf!r}",
            )

    def test_every_expert_of_every_moe_layer_is_exported(self):
        """vLLM iterates expert ids; a gap would leave that expert unloaded."""
        per_layer: dict[str, set[tuple[int, str]]] = {}
        for k in self.exported:
            if ".experts." not in k:
                continue
            head, rest = k.split(".block_sparse_moe.experts.", 1)
            idx, leaf = rest.split(".")[0], rest.split(".")[1]
            per_layer.setdefault(head, set()).add((int(idx), leaf))
        self.assertTrue(per_layer)
        want = {
            (e, w)
            for e in range(self.cfg.num_experts)
            for w in VLLM_CKPT_EXPERT_NAMES.values()
        }
        for layer, got in per_layer.items():
            self.assertEqual(got, want, f"{layer} is missing expert slices")

    def test_dense_ffn_is_exported_unfused(self):
        """vLLM fuses gate_proj+up_proj itself; a pre-fused name never matches."""
        dense = [k for k in self.exported if ".mlp." in k]
        self.assertTrue(dense, "k3mini must have a dense layer")
        self.assertTrue(any(k.endswith(".gate_proj.weight") for k in dense))
        self.assertTrue(any(k.endswith(".up_proj.weight") for k in dense))
        for k in self.exported:
            self.assertNotIn(
                VLLM_FUSES_INTO,
                k,
                f"{k} is pre-fused; vLLM expects the unfused pair",
            )

    def test_shared_experts_are_unfused_too(self):
        shared = [k for k in self.exported if ".shared_experts." in k]
        self.assertTrue(shared)
        leaves = {k.rsplit(".", 2)[-2] for k in shared}
        self.assertEqual(leaves, {"gate_proj", "up_proj", "down_proj"})

    def test_no_titan_internal_names_leak(self):
        """A name that still carries our module structure would be dropped by
        vLLM's loader as unrecognized -- silently, since it only warns."""
        # Matched as PATH COMPONENTS, not substrings: the official name
        # "block_sparse_moe" contains "_moe", so a substring check flags a
        # correct export.
        leaks = {
            "_moe",
            "routed_experts",
            "inner_experts",
            "w1_EFD",
            "w2_EDF",
            "w3_EFD",
            "latent",
            "attn_gate_proj",
            "output_res_proj",
            "output_res_norm",
            "moe",
        }
        for k in self.exported:
            parts = set(k.split("."))
            self.assertEqual(
                parts & leaks, set(), f"{k} leaks internal name components"
            )

    def test_the_k3_class_is_not_in_this_vllm_yet(self):
        """Documents the scope limit rather than asserting a capability we have
        not got. If this starts failing, K3 landed and the mapping should be
        re-checked against the real class."""
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except Exception as e:
            # Not just ImportError. vLLM 0.26.0 pins torch 2.11 while the
            # torchtitan revision this fork tracks needs a nightly (2.14.dev,
            # for DataParallelMeshDims), so the two CANNOT share a venv --
            # importing vllm here dies with "operator torchvision::nms does not
            # exist". That is a real deployment constraint for veRL rather than a
            # test problem: trainer and rollout engine need separate
            # environments, which is how veRL runs them anyway.
            self.skipTest(f"vllm unusable in this venv: {type(e).__name__}: {e}")
        archs = set(ModelRegistry.get_supported_archs())
        self.assertIn("KimiLinearForCausalLM", archs)
        if "KimiK3ForConditionalGeneration" in archs:
            self.fail(
                "vLLM now registers KimiK3ForConditionalGeneration -- re-verify "
                "this contract against that class's load_weights"
            )


if __name__ == "__main__":
    unittest.main()
