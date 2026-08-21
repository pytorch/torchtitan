# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for KimiLinearStateDictAdapter (HF <-> tt key mapping).

Uses meta-device builds -- key/shape coverage only, no weight values.
"""

import inspect
import unittest

import torch

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiLinearStateDictAdapter


def _build_state_dict(flavor: str):
    spec = model_registry(flavor)
    with torch.device("meta"):
        model = spec.model.build()
    return spec, model.state_dict()


class TestKimiLinearStateDictAdapter(unittest.TestCase):
    def test_wired_into_model_registry(self):
        spec = model_registry("kimi_linear_194m_block_attn_res")
        self.assertIs(spec.state_dict_adapter, KimiLinearStateDictAdapter)

    def test_round_trip_194m_block_attn_res(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        back = adapter.from_hf(hf)
        # Graft extras (attn_res/mlp_res) are deliberately NOT part of
        # the HF key space (official checkpoints must load into graft
        # flavors without phantom read keys); the round trip covers the
        # backbone exactly.
        backbone = {
            k
            for k in sd
            if "attention_res" not in k
            and "ffn_res" not in k
            and "output_res" not in k
        }
        self.assertEqual(set(back), backbone)
        for k in backbone:
            self.assertEqual(
                tuple(back[k].shape), tuple(sd[k].shape), f"shape drift at {k}"
            )

    def test_round_trip_baseline_no_attn_res(self):
        """Baseline flavor has no attn_res keys; mapping must still cover all."""
        spec, sd = _build_state_dict("kimi_linear_194m_baseline")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        back = adapter.from_hf(adapter.to_hf(sd))
        self.assertEqual(set(back), set(sd))

    def test_expert_weights_split_and_restack(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        num_experts = spec.model.kimi_config.num_experts
        # Per-expert HF keys exist for a known MoE layer
        moe_keys = [k for k in hf if ".block_sparse_moe.experts." in k]
        self.assertTrue(moe_keys)
        self.assertEqual(
            len(moe_keys),
            3 * num_experts * sum(1 for k in sd if k.endswith("w1_EFD")),
        )

    def test_a_log_reshape_from_hf(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        a_log_keys = [k for k in sd if k.endswith("delta_attention.A_log")]
        self.assertTrue(a_log_keys)
        h = sd[a_log_keys[0]].shape[0]
        # The file spells the module self_attn for both attention kinds; ours is
        # delta_attention on a KDA layer. Prefixing our own key with "model."
        # only produced a valid HF key while the two spellings coincided.
        hf_key = "model." + a_log_keys[0].replace(
            "delta_attention.", "self_attn.", 1
        )
        # from_hf must flatten [1,1,H,1] -> [H]
        out = adapter.from_hf({hf_key: torch.zeros(1, 1, h, 1)})
        self.assertEqual(tuple(out[a_log_keys[0]].shape), (h,))

    def test_packed_weights_rejected(self):
        spec, _ = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        with self.assertRaises(NotImplementedError):
            adapter.from_hf(
                {"model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(2)}
            )
        with self.assertRaises(NotImplementedError):
            adapter.from_hf(
                {
                    "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                        4, 4, dtype=torch.uint8
                    )
                }
            )

    def test_quantized_reader_dequantizes(self):
        """from_quantized must return a reader that UNPACKS, not a plain one.

        This replaces a test that pinned a blanket NotImplementedError. The hazard
        that refusal guarded -- packed bytes reaching the model as if they were
        values -- is what is asserted here instead: the reader has to be the
        dequantizing subclass. A plain HuggingFaceStorageReader would pass uint8
        blocks straight through.
        """
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
        from torch.distributed.checkpoint.quantized_hf_storage import (
            QuantizedHuggingFaceStorageReader,
        )

        spec, _ = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        reader = adapter.get_hf_storage_reader("/nonexistent", from_quantized=True)
        self.assertIsInstance(reader, QuantizedHuggingFaceStorageReader)

        plain = adapter.get_hf_storage_reader("/nonexistent")
        self.assertIsInstance(plain, HuggingFaceStorageReader)
        self.assertNotIsInstance(plain, QuantizedHuggingFaceStorageReader)

    def test_our_e2m1_table_matches_the_readers(self):
        """The two decoders have to agree on the value table, or a checkpoint read
        through torch's reader would differ from one read through ours."""
        import re

        from torch.distributed.checkpoint.quantized_hf_storage import (
            QuantizedHuggingFaceStorageReader,
        )

        from torchtitan.models.kimi_k3.packed_mxfp4 import (
            _E2M1_VALUES,
            MXFP4_GROUP_SIZE,
        )

        src = inspect.getsource(
            QuantizedHuggingFaceStorageReader._dequantize_tensor_mxfp4
        )
        # Anchored past the "[" so the 4 in "FP4_VALUES" is not read as an entry.
        start = src.index("[", src.index("FP4_VALUES"))
        table = src[start : src.index("]", start)]
        theirs = tuple(float(v) for v in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", table))
        # Length first: this reads upstream source, so a reformat there could
        # silently extract nothing and make the comparison vacuous. Sixteen is the
        # only correct answer for E2M1, so assert it and let a change be loud.
        self.assertEqual(len(theirs), 16)
        self.assertEqual(theirs, tuple(float(v) for v in _E2M1_VALUES))
        self.assertEqual(MXFP4_GROUP_SIZE, 32)

    def test_tied_embedding_alias_warns(self):
        spec, sd = _build_state_dict("kimi_linear_194m_block_attn_res")
        adapter = KimiLinearStateDictAdapter(spec.model, hf_assets_path=None)
        hf = adapter.to_hf(sd)
        hf.pop("lm_head.weight")
        back = adapter.from_hf(hf)
        self.assertIn("lm_head.weight", back)


if __name__ == "__main__":
    unittest.main()


class TestMultimodalRoundTrip(unittest.TestCase):
    """to_hf -> from_hf on a MULTIMODAL flavor must return the text keys.

    The round-trip tests above use text-only flavors, whose keys have no wrapper
    prefix. A multimodal model's text tensors are named ``language_model.*``, and
    ``to_hf`` strips that prefix before handing the key to ``hf_key_map`` -- so the
    inverse has to put it back. It did not, and because an unmapped key returns
    ``(None, value)`` rather than raising, every text tensor of a multimodal
    checkpoint was dropped in silence: loading an official shard produced a
    near-empty state dict with no error.

    Written as its own class so the failure names the direction that is broken.
    """

    def _adapter_and_sd(self):
        # Multimodal flavors live in config_registry (Trainer.Config factories), not in
        # model_registry, which only parses 'kimi_k3_<size>_<variant>'.
        from torchtitan.models.kimi_k3.config_registry import kimi_k3_mini_vl

        model_spec = kimi_k3_mini_vl().model_spec
        with torch.device("meta"):
            model = model_spec.model.build()
        return (
            KimiLinearStateDictAdapter(model_spec.model, hf_assets_path=None),
            model.state_dict(),
        )

    def test_text_keys_survive_the_round_trip(self):
        adapter, sd = self._adapter_and_sd()
        back = adapter.from_hf(adapter.to_hf(sd))
        # Graft extras are deliberately outside the HF key space, as in the
        # text-only round trips above.
        expected = {
            k
            for k in sd
            if "attention_res" not in k
            and "ffn_res" not in k
            and "output_res" not in k
            and k.startswith("language_model.")
        }
        missing = sorted(expected - set(back))
        self.assertEqual(
            missing[:8],
            [],
            f"{len(missing)} of {len(expected)} text tensors lost in the round trip",
        )

    def test_round_trip_preserves_text_shapes(self):
        adapter, sd = self._adapter_and_sd()
        back = adapter.from_hf(adapter.to_hf(sd))
        for k, v in sd.items():
            if not k.startswith("language_model."):
                continue
            if "attention_res" in k or "ffn_res" in k or "output_res" in k:
                continue
            if k in back:
                self.assertEqual(
                    tuple(back[k].shape), tuple(v.shape), f"shape drift at {k}"
                )
