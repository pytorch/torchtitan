# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

import torch
from torchtitan.config import (
    apply_overrides,
    clear_overrides,
    Configurable,
    derive,
    override,
    OverrideConfig,
)
from torchtitan.models.common.attention import (
    GQAttention,
    QKVLinear,
    ScaledDotProductAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import (
    _maybe_check_max_pos,
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
)
from torchtitan.models.qwen3_5.rope import MRoPE


class TestApplyRotaryEmbCosSin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 16
        self.n_heads = 4
        self.head_dim = 64
        self.xq = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.rope_cache = torch.randn(
            self.seqlen, self.head_dim * 2, dtype=torch.float32
        ).view(1, self.seqlen, 1, self.head_dim * 2)
        self.rope = CosSinRoPE(
            CosSinRoPE.Config(dim=self.head_dim, max_seq_len=self.seqlen)
        )

    def test_output_dtype_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.dtype, self.xq.dtype)
        self.assertEqual(xk_out.dtype, self.xk.dtype)

    def test_output_shape_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.shape, self.xq.shape)
        self.assertEqual(xk_out.shape, self.xk.shape)

    def test_computes_in_fp32(self):
        """Output must match a reference computed entirely in float32.

        Ensures inductor cannot fuse away the fp32 upcast when compiling
        adjacent ops (e.g. q_norm/k_norm) with the RoPE computation.
        """
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )

        cos = self.rope_cache[..., : self.head_dim]
        sin = self.rope_cache[..., self.head_dim :]

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        xq_ref = (
            (self.xq.float() * cos) + (rotate_half(self.xq.float()) * sin)
        ).bfloat16()
        xk_ref = (
            (self.xk.float() * cos) + (rotate_half(self.xk.float()) * sin)
        ).bfloat16()

        self.assertEqual((xq_out - xq_ref).abs().max().item(), 0.0)
        self.assertEqual((xk_out - xk_ref).abs().max().item(), 0.0)


class TestMaybeCheckMaxPos(unittest.TestCase):
    """Tests for the _maybe_check_max_pos bounds check."""

    def test_positions_within_bounds(self):
        positions = torch.tensor([[0, 1, 2, 3]])
        _maybe_check_max_pos(positions, max_valid_pos=3)

    def test_positions_at_boundary(self):
        positions = torch.tensor([[0, 5, 10, 15]])
        _maybe_check_max_pos(positions, max_valid_pos=15)

    def test_positions_out_of_bounds_raises(self):
        positions = torch.tensor([[0, 1, 2, 16]])
        with self.assertRaises(RuntimeError):
            _maybe_check_max_pos(positions, max_valid_pos=15)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


class TestRoPEPositionBoundsComplex(unittest.TestCase):
    """RoPE complex-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = ComplexRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, ComplexRoPE)

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = CosSinRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, CosSinRoPE)

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestMRoPECache(unittest.TestCase):
    def test_forward_accepts_three_axis_positions(self):
        torch.manual_seed(42)
        bsz, seqlen, n_heads = 2, 3, 4
        head_dim = 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_seq_len=8,
            mrope_section=[2, 1, 1],
        ).build()
        # (batch, seq, 3): per-token [temporal, height, width] positions.
        position_ids = torch.tensor(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],  # batch 0
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],  # batch 1
            ]
        )
        xq = torch.randn(bsz, seqlen, n_heads, head_dim)
        xk = torch.randn(bsz, seqlen, n_heads, head_dim)

        xq_out, xk_out = rope(xq, xk, position_ids)

        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)


class TestPerLayerRoPECache(unittest.TestCase):
    def test_gqa_attention_uses_layer_rope_cache(self):
        torch.manual_seed(42)
        dim = 8
        head_dim = 4
        attention = GQAttention.Config(
            n_heads=2,
            n_kv_heads=2,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=ScaledDotProductAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_seq_len=16),
        ).build()

        x = torch.randn(2, 4, dim)
        out = attention(x, None)

        self.assertIsNotNone(attention.rope)
        self.assertEqual(out.shape, x.shape)

    def test_decoder_builds_distinct_rope_modules_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        layer_ropes = [layer.attention.rope for layer in model.layers.values()]

        self.assertTrue(all(isinstance(rope, RoPE) for rope in layer_ropes))
        self.assertEqual(len({id(rope) for rope in layer_ropes}), len(layer_ropes))

    def test_decoder_builds_distinct_rope_configs_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        cfg = llama3_configs["debugmodel"]("flex")
        layer_rope_cfgs = [layer.attention.rope for layer in cfg.layers]

        self.assertEqual(
            len({id(rope_cfg) for rope_cfg in layer_rope_cfgs}),
            len(layer_rope_cfgs),
        )


def _attention_config(rope_cfg, *, dim=8, head_dim=4) -> GQAttention.Config:
    """A minimal ``GQAttention.Config`` around ``rope_cfg``."""
    return GQAttention.Config(
        n_heads=2,
        n_kv_heads=2,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(in_features=dim, out_features=dim),
            wkv=Linear.Config(in_features=dim, out_features=dim),
        ),
        wo=Linear.Config(in_features=dim, out_features=dim),
        inner_attention=ScaledDotProductAttention.Config(),
        rope=rope_cfg,
    )


class TestRoPEConfigBuildReuse(unittest.TestCase):
    """``RoPE.Config.build`` memoizes, so one config object means one module.

    Reuse is opt-in through the config tree: layers handed the same config
    object share a RoPE (and its cache); layers handed separate configs do not.
    """

    def test_same_config_object_builds_one_module(self):
        cfg = ComplexRoPE.Config(dim=8, max_seq_len=16)

        first, second = cfg.build(), cfg.build()

        self.assertIs(first, second)
        self.assertEqual(first.cache.data_ptr(), second.cache.data_ptr())

    def test_separate_configs_build_distinct_modules(self):
        first = ComplexRoPE.Config(dim=8, max_seq_len=16).build()
        second = ComplexRoPE.Config(dim=8, max_seq_len=16).build()

        self.assertIsNot(first, second)
        self.assertNotEqual(first.cache.data_ptr(), second.cache.data_ptr())
        torch.testing.assert_close(first.cache, second.cache)

    def test_replace_builds_a_fresh_module(self):
        """The memo must not survive ``replace``, or a retargeted config would
        return a module built for the old field values."""
        cfg = ComplexRoPE.Config(dim=8, max_seq_len=16)
        original = cfg.build()

        retargeted = dataclasses.replace(cfg, max_seq_len=32).build()

        self.assertIsNot(retargeted, original)
        self.assertEqual(retargeted.cache.shape[0], 32)
        self.assertEqual(original.cache.shape[0], 16)

    def test_building_does_not_change_config_identity_semantics(self):
        """Building must not leak into equality or serialization, which config
        dumps and comparisons rely on."""
        cfg = ComplexRoPE.Config(dim=8, max_seq_len=16)
        other = ComplexRoPE.Config(dim=8, max_seq_len=16)
        before = cfg.to_dict()

        cfg.build()

        self.assertEqual(cfg, other)
        self.assertEqual(cfg.to_dict(), before)

    def test_layers_sharing_a_rope_config_share_one_cache(self):
        rope_cfg = ComplexRoPE.Config(dim=4, max_seq_len=16)
        shared_a = _attention_config(rope_cfg).build()
        shared_b = _attention_config(rope_cfg).build()

        self.assertIs(shared_a.rope, shared_b.rope)
        self.assertEqual(shared_a.rope.cache.data_ptr(), shared_b.rope.cache.data_ptr())

    def test_sharing_survives_init_states(self):
        """``init_states`` recomputes caches per module; a shared RoPE must
        still end up with one cache holding the reference values."""
        rope_cfg = ComplexRoPE.Config(dim=4, max_seq_len=16)
        shared_a = _attention_config(rope_cfg).build()
        shared_b = _attention_config(rope_cfg).build()
        unshared = _attention_config(ComplexRoPE.Config(dim=4, max_seq_len=16)).build()

        for attention in (shared_a, shared_b, unshared):
            attention.init_states(buffer_device=torch.device("cpu"))

        self.assertIs(shared_a.rope, shared_b.rope)
        torch.testing.assert_close(shared_a.rope.cache, unshared.rope.cache)

    def test_shared_rope_does_not_change_attention_output(self):
        torch.manual_seed(42)
        rope_cfg = ComplexRoPE.Config(dim=4, max_seq_len=16)
        shared = _attention_config(rope_cfg).build()
        unshared = _attention_config(ComplexRoPE.Config(dim=4, max_seq_len=16)).build()
        unshared.load_state_dict(shared.state_dict())

        x = torch.randn(2, 4, 8)

        torch.testing.assert_close(shared(x, None), unshared(x, None))


class _ReplacementRoPE(ComplexRoPE):
    """Stand-in for an override replacement such as ``HelionComplexRoPE``."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass


class _AttentionLayers(Configurable):
    """Minimal config root holding several attention configs."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        layers: list = dataclasses.field(default_factory=list)

    def __init__(self, config: Config):
        self.config = config


class TestRoPEOverrideReuse(unittest.TestCase):
    """An override must not split a shared rope config into per-layer copies."""

    def setUp(self):
        clear_overrides()

    def tearDown(self):
        clear_overrides()

    def test_override_keeps_one_module_for_a_shared_rope_config(self):
        @override(target=ComplexRoPE.Config, exact=True)
        def to_replacement_rope(cfg: ComplexRoPE.Config) -> _ReplacementRoPE.Config:
            return derive(cfg, _ReplacementRoPE.Config)

        rope_cfg = ComplexRoPE.Config(dim=4, max_seq_len=16)
        root = _AttentionLayers.Config(
            layers=[
                _attention_config(rope_cfg),
                _attention_config(rope_cfg),
            ]
        )

        replacements = apply_overrides(
            OverrideConfig(imports=[f"{__name__}.to_replacement_rope"]), root
        )

        # One node claimed through two slots -> one replacement config ...
        self.assertEqual(len(replacements), 2)
        self.assertIs(root.layers[0].rope, root.layers[1].rope)
        # ... so the layers still share one module and one cache.
        first, second = root.layers[0].build(), root.layers[1].build()
        self.assertIsInstance(first.rope, _ReplacementRoPE)
        self.assertIs(first.rope, second.rope)


class TestUpdateFromConfigSeqLenValidation(unittest.TestCase):
    """update_from_config must reject seq_len > rope.max_seq_len."""

    def _make_trainer_config(self, seq_len):
        from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
        from torchtitan.trainer import Trainer

        return Trainer.Config(
            training=dataclasses.replace(TrainingConfig(), seq_len=seq_len),
            parallelism=ParallelismConfig(),
            debug=DebugConfig(),
        )

    def _make_config(self):
        """Build a minimal Llama3 debug config."""
        from torchtitan.models.llama3 import llama3_configs

        return llama3_configs["debugmodel"]("flex")

    def test_rejects_oversized_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_seq_len
        with self.assertRaises(ValueError):
            cfg.update_from_config(config=self._make_trainer_config(rope_max + 1))

    def test_accepts_valid_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_seq_len
        cfg.update_from_config(config=self._make_trainer_config(rope_max))
        self.assertEqual(cfg.max_seq_len, rope_max)

    def test_vllm_max_model_len_as_seq_len(self):
        """vLLM wrapper translates max_model_len to TrainingConfig.seq_len.

        When seq_len equals rope.max_seq_len, the RoPE cache stays at
        the model's intrinsic maximum.
        """
        cfg = self._make_config()
        original_max = cfg.max_seq_len
        cfg.update_from_config(config=self._make_trainer_config(original_max))
        self.assertEqual(cfg.max_seq_len, original_max)


if __name__ == "__main__":
    unittest.main()
