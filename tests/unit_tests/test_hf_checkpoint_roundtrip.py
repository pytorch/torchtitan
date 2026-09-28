# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bit-exact HuggingFace checkpoint adapter tests.

torchtitan ships per-model ``StateDictAdapter`` implementations that map
between its native state dict layout and the HuggingFace safetensors
layout. Every transformation those adapters perform — key rename, RoPE
permutation, fused-vs-separate QKV repack, weight tying — is by design a
pure tensor reshuffle (no arithmetic, no dtype change). The adapter-level
checkpoint round-trip (native -> HF -> native) must therefore reconstruct
the native state dict **bit-for-bit**.

This test catches conversion bugs that drop keys, change tensor metadata,
or lose values across the adapter boundary. It also writes the HF-format
dict to safetensors after ``from_hf`` consumes it, catching unserializable
adapter output and ``from_hf`` input mutations that leave the HF dict
invalid for checkpointing.

Per model+flavor under test:

1. Build the model on CPU with random init; capture native state dict.
2. ``to_hf`` it once → ``hf_first``.
3. Round-trip ``from_hf(hf_first)`` → ``tt_back``.

Then two assertions:

* **Native-side bit equality**: ``torch.equal(orig, tt_back)`` for every
  key. Catches loss across ``to_hf`` followed by ``from_hf``.
* **HF-side serialization**: serialize ``hf_first`` to safetensors.
  Catches adapter output that cannot be saved as an HF checkpoint.
"""

import importlib
import os
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
]


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

        # native -> HF
        hf_first = adapter.to_hf(tt_orig)
        # HF -> native
        tt_back = adapter.from_hf(hf_first)

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

        # ---- HF-side: the produced HF dict must be safetensors-serializable.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.safetensors")
            safetensors.torch.save_file(hf_first, path)

    def test_roundtrip_is_bit_exact(self) -> None:
        for model_name, flavor in _MODEL_FLAVORS:
            with self.subTest(model_name=model_name, flavor=flavor):
                self._assert_roundtrip(model_name, flavor)


if __name__ == "__main__":
    unittest.main()
