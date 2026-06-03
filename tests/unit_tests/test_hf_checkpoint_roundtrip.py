# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bit-exact HuggingFace checkpoint round-trip tests.

torchtitan ships per-model ``StateDictAdapter`` implementations that map
between its native state dict layout and the HuggingFace safetensors
layout. Every transformation those adapters perform — key rename, RoPE
permutation, fused-vs-separate QKV repack, weight tying — is by design a
pure tensor reshuffle (no arithmetic, no dtype change). The end-to-end
checkpoint round-trip (native ↔ HF, then ↔ native again) must therefore
be invertible **bit-for-bit**, both as in-memory dicts and as
serialized safetensors files on disk.

This test catches conversion bugs that forward-pass numerical tests miss
when the bug is self-consistent — e.g., a swapped axis in ``_permute``
that makes the adapter agree with itself but disagree with HuggingFace's
own conversion script, or a key-rename typo that silently drops weights
on one direction of the conversion.

Per model+flavor under test:

1. Build the model on CPU with random init; capture native state dict.
2. ``to_hf`` it once → ``hf_first``.
3. Round-trip ``from_hf(hf_first)`` → ``tt_back``.
4. ``to_hf(tt_back)`` again → ``hf_second``.

Then two assertions:

* **Native-side bit equality**: ``torch.equal(orig, tt_back)`` for every
  key. Catches loss across ``to_hf ∘ from_hf``.
* **HF-side file byte equality**: serialize ``hf_first`` and
  ``hf_second`` to safetensors and compare the resulting files
  byte-for-byte. Catches loss across ``from_hf ∘ to_hf`` *and* any
  non-determinism the safetensors layer introduces.
"""

import filecmp
import importlib
import os
import re
import tempfile
import unittest

import safetensors.torch
import torch


# (model_name, flavor) pairs that exercise distinct adapter code paths.
# Every registered decoder-family adapter has a "debugmodel" flavor that
# constructs a tiny config suitable for CPU unit tests. New models are
# added here as their adapters are audited and any round-trip bugs they
# expose are fixed in their own diffs.
_MODEL_FLAVORS = [
    # llama3: vanilla GQA, separate QKV. Baseline correct adapter.
    ("llama3", "debugmodel"),
    # llama3: fused-QKV path, exercises fused_to_separate_qkv /
    # separate_to_fused_qkv.
    ("llama3", "debugmodel_fused_qkv"),
    # llama4: interleaved dense + MoE decoder. Exercises shape-suffix MoE keys
    # (w1_EFD / w2_EDF / w3_EFD), dense FF mappings on non-MoE layers, and
    # expert_bias_E reinit (see _NATIVE_EXCLUSIONS).
    ("llama4", "debugmodel"),
    # gpt_oss: gated MoE with gate_up_proj fused into a single mlp1 weight.
    # Different shape-suffix naming (mlp1_*_EGD/EG, mlp2_*_EDF/ED) but same
    # expert_bias_E semantics as llama4.
    ("gpt_oss", "debugmodel"),
    # deepseek_v3: MLA attention + MoE with per-expert HF keys. Unique among
    # MoE models here: HF DeepSeek has e_score_correction_bias as a real
    # buffer, so expert_bias_E genuinely round-trips (no exclusion needed).
    ("deepseek_v3", "debugmodel"),
    # qwen3: weight-tied lm_head + GQA. Exercises the from_hf input-copy
    # contract and the tying-restore path.
    ("qwen3", "debugmodel"),
    # qwen3: fused-QKV variant of the above.
    ("qwen3", "debugmodel_fused_qkv"),
]

# Native-side keys (or regex patterns) excluded from the bitwise comparison,
# keyed by (model_name, flavor). Every exclusion MUST have a comment that
# explains why the key cannot round-trip exactly through HF format — these
# are intentional design carve-outs, not unexplained skips.
_NATIVE_EXCLUSIONS: dict[tuple[str, str], list[str]] = {
    # llama4: `moe.expert_bias_E` is auxiliary-loss-free load-balancing bias
    # (DeepSeek paper, adapted for Qwen/Mixtral-style routers including
    # llama4). HF transformers' Llama4 router has no equivalent buffer
    # (Llama4Router is plain nn.Linear with bias=False), so there is no
    # HF key to map to. The adapter intentionally drops it on to_hf
    # (matches HF format) and reinitializes it to zeros on from_hf
    # (preserves the key set). The bias therefore does not survive an HF
    # round-trip by design; the optimizer pre-hook recomputes it during
    # training. Resuming from an HF checkpoint loses any accumulated bias.
    ("llama4", "debugmodel"): [r".*\.moe\.expert_bias_E$"],
    # gpt_oss: same expert_bias_E story as llama4 — HF transformers' gpt_oss
    # router has no equivalent buffer, so the round-trip resets the bias to
    # zeros by design.
    ("gpt_oss", "debugmodel"): [r".*\.moe\.expert_bias_E$"],
}


def _build_model_and_adapter(model_name: str, flavor: str):
    """Construct a CPU-initialized model and its state_dict adapter."""
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    spec = model_module.model_registry(flavor)
    model_config = spec.model
    with torch.device("cpu"):
        model = model_config.build()
    model.to_empty(device="cpu")
    model.init_weights(buffer_device=torch.device("cpu"))
    adapter = spec.state_dict_adapter(model_config, hf_assets_path=None)
    return model, adapter


class TestHFCheckpointRoundtrip(unittest.TestCase):
    """End-to-end HF checkpoint round-trip is bit-exact in both formats."""

    def _assert_roundtrip(self, model_name: str, flavor: str) -> None:
        model, adapter = _build_model_and_adapter(model_name, flavor)
        tt_orig = model.state_dict()

        # native → HF (first pass)
        hf_first = adapter.to_hf(tt_orig)
        # HF → native
        tt_back = adapter.from_hf(hf_first)
        # native → HF (second pass)
        hf_second = adapter.to_hf(tt_back)

        # Native keys that are intentionally excluded from the bitwise tensor
        # comparison (e.g. buffers that have no HF equivalent and get
        # reinitialized to zeros on from_hf). The key set itself is still
        # required to match — exclusions only apply to tensor-value comparison.
        exclusion_patterns = [
            re.compile(p) for p in _NATIVE_EXCLUSIONS.get((model_name, flavor), [])
        ]

        def _is_excluded(fqn: str) -> bool:
            return any(p.search(fqn) for p in exclusion_patterns)

        # ---- Native-side: tt_back must equal tt_orig key-for-key, bit-for-bit.
        missing = tt_orig.keys() - tt_back.keys()
        extra = tt_back.keys() - tt_orig.keys()
        self.assertFalse(
            missing or extra,
            f"{model_name}/{flavor}: native key set diverged "
            f"missing={sorted(missing)} extra={sorted(extra)}",
        )
        for fqn, orig_tensor in tt_orig.items():
            back_tensor = tt_back[fqn]
            self.assertEqual(
                orig_tensor.shape,
                back_tensor.shape,
                f"{model_name}/{flavor}: native shape diverged for {fqn}: "
                f"{tuple(orig_tensor.shape)} vs {tuple(back_tensor.shape)}",
            )
            self.assertEqual(
                orig_tensor.dtype,
                back_tensor.dtype,
                f"{model_name}/{flavor}: native dtype diverged for {fqn}: "
                f"{orig_tensor.dtype} vs {back_tensor.dtype}",
            )
            if _is_excluded(fqn):
                continue
            if not torch.equal(orig_tensor, back_tensor):
                max_abs_diff = (
                    (orig_tensor.to(torch.float64) - back_tensor.to(torch.float64))
                    .abs()
                    .max()
                    .item()
                )
                self.fail(
                    f"{model_name}/{flavor}: native bitwise mismatch at "
                    f"{fqn} (max|Δ|={max_abs_diff:.3e}, "
                    f"shape={tuple(orig_tensor.shape)})"
                )

        # ---- HF-side: serialize both passes, compare safetensors bytes.
        with tempfile.TemporaryDirectory() as td:
            path_a = os.path.join(td, "first.safetensors")
            path_b = os.path.join(td, "second.safetensors")
            safetensors.torch.save_file(hf_first, path_a)
            safetensors.torch.save_file(hf_second, path_b)
            if not filecmp.cmp(path_a, path_b, shallow=False):
                size_a = os.path.getsize(path_a)
                size_b = os.path.getsize(path_b)
                self.fail(
                    f"{model_name}/{flavor}: safetensors files diverged "
                    f"(first={size_a}B, second={size_b}B). The "
                    f"to_hf ∘ from_hf composition is not the identity on "
                    f"the HF side."
                )

    def test_roundtrip_is_bit_exact(self) -> None:
        for model_name, flavor in _MODEL_FLAVORS:
            with self.subTest(model_name=model_name, flavor=flavor):
                self._assert_roundtrip(model_name, flavor)


if __name__ == "__main__":
    unittest.main()
