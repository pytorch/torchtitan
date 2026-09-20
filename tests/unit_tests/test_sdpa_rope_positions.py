import unittest
from unittest.mock import patch

import torch

from torchtitan.models.common.attention import GQAttention


class _StopAfterRoPE(Exception):
    pass


class TestGQAttentionRoPEPositions(unittest.TestCase):
    def _run_until_rope(self, attn_backend: str, rope_backend: str):
        attention = GQAttention(
            GQAttention.Config(
                n_heads=2,
                n_kv_heads=1,
                attn_backend=attn_backend,
                rope_backend=rope_backend,
            ),
            dim=8,
        )
        positions = torch.tensor([[4, 5, 6]])
        captured_positions = []

        def stop_after_rope(*args, **kwargs):
            received_positions = (
                kwargs["positions"] if "positions" in kwargs else args[3]
            )
            captured_positions.append(received_positions)
            raise _StopAfterRoPE

        rope_function = (
            "apply_rotary_emb_cos_sin"
            if rope_backend == "cos_sin"
            else "apply_rotary_emb_complex"
        )
        with patch(
            f"torchtitan.models.common.attention.{rope_function}",
            side_effect=stop_after_rope,
        ):
            with self.assertRaises(_StopAfterRoPE):
                attention(
                    torch.randn(1, 3, 8),
                    torch.randn(8, 2),
                    attention_masks=None,
                    positions=positions,
                )

        return captured_positions[0], positions

    def test_sdpa_drops_positions_for_both_rope_formats(self):
        for rope_backend in ("complex", "cos_sin"):
            with self.subTest(rope_backend=rope_backend):
                received_positions, _ = self._run_until_rope("sdpa", rope_backend)
                self.assertIsNone(received_positions)

    def test_non_sdpa_keeps_positions_for_both_rope_formats(self):
        for attn_backend in ("flex", "varlen"):
            for rope_backend in ("complex", "cos_sin"):
                with self.subTest(
                    attn_backend=attn_backend, rope_backend=rope_backend
                ):
                    received_positions, positions = self._run_until_rope(
                        attn_backend, rope_backend
                    )
                    self.assertIs(received_positions, positions)
