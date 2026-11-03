# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 numerical-script MRoPE positions follow the collator contract.

``MultiModalCollator._build_mrope_positions`` reads ``patch_order`` and wants
a 1-D token row. The numerical script used to pass a stand-in with no
``patch_order`` and the processor's ``(1, S)`` ``input_ids``.
"""

import ast
import types
import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "checkpoint_conversion"
    / "numerical_tests_qwen3_5.py"
)


def _is_raw_hf_input_ids(node: ast.AST) -> bool:
    """True for a bare ``hf_in["input_ids"]`` expression."""
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != "hf_in":
        return False
    slice_node = node.slice
    return isinstance(slice_node, ast.Constant) and slice_node.value == "input_ids"


def _load_numerical_script():
    try:
        from scripts.checkpoint_conversion import numerical_tests_qwen3_5 as module
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        raise unittest.SkipTest(
            f"could not import the Qwen3.5 numerical script ({detail})"
        ) from exc
    return module


def _load_collator():
    try:
        from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        raise unittest.SkipTest(
            f"could not import MultiModalCollator ({detail})"
        ) from exc
    return MultiModalCollator


class TestQwen35NumericalMropeSource(unittest.TestCase):
    """Source checks that run without importing torchtitan."""

    def test_script_matches_collator_mrope_contract(self):
        source = _SCRIPT.read_text()
        tree = ast.parse(source)
        has_block_order = any(
            isinstance(node, ast.keyword)
            and node.arg == "patch_order"
            and isinstance(node.value, ast.Constant)
            and node.value.value == "block"
            for node in ast.walk(tree)
        )
        self.assertTrue(
            has_block_order,
            'numerical script must pass patch_order="block" into the MRoPE builder',
        )
        self.assertIn('patch_order="block"', source)

        saw_mrope_call = False
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "_build_mrope_positions"
            ):
                continue
            saw_mrope_call = True
            arguments = list(call.args) + [kw.value for kw in call.keywords]
            for arg in arguments:
                self.assertFalse(
                    _is_raw_hf_input_ids(arg),
                    "raw hf_in['input_ids'] is passed into _build_mrope_positions",
                )
        self.assertTrue(
            saw_mrope_call,
            "numerical script must still call _build_mrope_positions",
        )


class TestQwen35NumericalMropePositions(unittest.TestCase):
    """Positions handed to Qwen3.5 are the rank MRoPE indexes."""

    def test_text_row_positions_match_on_all_axes(self):
        import torch

        module = _load_numerical_script()
        helper = getattr(module, "_mrope_positions_for_forward", None)
        if helper is None:
            self.fail("numerical_tests_qwen3_5 is missing _mrope_positions_for_forward")

        # No image tokens, so every axis is the plain text position.
        token_row = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
        builder = types.SimpleNamespace(spatial_merge_size=2, patch_order="block")
        grid = torch.zeros(1, 3, dtype=torch.long)
        expected = torch.arange(token_row.shape[0])
        for tokens in (token_row, token_row.unsqueeze(0)):
            positions = helper(
                builder,
                tokens,
                grid,
                image_token_id=99,
                video_token_id=98,
            )
            # MRoPE._reshape_cache takes (num_tokens, 3). Rank 3 skips that path.
            self.assertEqual(positions.ndim, 2)
            self.assertEqual(tuple(positions.shape), (token_row.shape[0], 3))
            self.assertTrue(torch.equal(positions[:, 0], expected))
            self.assertTrue(torch.equal(positions[:, 1], expected))
            self.assertTrue(torch.equal(positions[:, 2], expected))

    def test_builder_without_patch_order_raises(self):
        import torch

        collator = _load_collator()
        builder = types.SimpleNamespace(spatial_merge_size=2)
        with self.assertRaises(AttributeError):
            collator._build_mrope_positions(
                builder,
                torch.tensor([1, 2, 3, 4]),
                None,
                None,
                None,
                image_token_id=99,
                video_token_id=98,
            )


if __name__ == "__main__":
    unittest.main()
