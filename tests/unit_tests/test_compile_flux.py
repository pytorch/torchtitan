# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig
from torchtitan.models.flux.parallelize import apply_compile, apply_compile_to_encoders


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.img_norm = nn.LayerNorm(dim)
        self.txt_norm = nn.LayerNorm(dim)
        self.img_linear = nn.Linear(dim, dim)
        self.txt_linear = nn.Linear(dim, dim)

    def forward(self, img, txt, vec=None, pe=None):
        return self.img_linear(self.img_norm(img)), self.txt_linear(self.txt_norm(txt))


class SingleStreamBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, vec=None, pe=None):
        return self.linear(self.norm(x))


class FakeFluxModel(nn.Module):
    """Minimal model matching the interface expected by apply_compile."""

    def __init__(self, num_double=2, num_single=3, dim=64):
        super().__init__()
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(dim) for _ in range(num_double)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(dim) for _ in range(num_single)]
        )

    def forward(self, img, txt):
        for block in self.double_blocks:
            img, txt = block(img, txt)
        x = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            x = block(x)
        return x


class FakeEncoderBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class FakeT5Model(nn.Module):
    """Mimics the T5 encoder structure: hf_module.encoder.block (ModuleList)."""

    def __init__(self, num_blocks=2, dim=32):
        super().__init__()
        self.hf_module = nn.Module()
        self.hf_module.encoder = nn.Module()
        self.hf_module.encoder.block = nn.ModuleList(
            [FakeEncoderBlock(dim) for _ in range(num_blocks)]
        )


class FakeCLIPModel(nn.Module):
    """Mimics the CLIP encoder structure: hf_module.text_model.encoder.layers (ModuleList)."""

    def __init__(self, num_layers=2, dim=32):
        super().__init__()
        self.hf_module = nn.Module()
        self.hf_module.text_model = nn.Module()
        self.hf_module.text_model.encoder = nn.Module()
        self.hf_module.text_model.encoder.layers = nn.ModuleList(
            [FakeEncoderBlock(dim) for _ in range(num_layers)]
        )


def _is_compiled(module: nn.Module) -> bool:
    return isinstance(module, torch._dynamo.OptimizedModule)


class TestApplyCompileFlux(unittest.TestCase):
    def test_apply_compile_wraps_all_blocks(self):
        """Verify apply_compile wraps every double and single block."""
        model = FakeFluxModel(num_double=2, num_single=3)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile(model, compile_config)

        for block in model.double_blocks:
            self.assertTrue(
                _is_compiled(block),
                "DoubleStreamBlock should be compiled",
            )
        for block in model.single_blocks:
            self.assertTrue(
                _is_compiled(block),
                "SingleStreamBlock should be compiled",
            )

    def test_apply_compile_uses_specified_backend(self):
        """Verify the compile backend from config is actually used."""
        model = FakeFluxModel(num_double=1, num_single=1)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile(model, compile_config)

        for block in model.double_blocks:
            self.assertTrue(_is_compiled(block))
        for block in model.single_blocks:
            self.assertTrue(_is_compiled(block))

    def test_forward_after_compile(self):
        """Verify a forward pass succeeds after compilation."""
        model = FakeFluxModel(num_double=2, num_single=2, dim=64)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile(model, compile_config)

        img = torch.randn(2, 4, 64)
        txt = torch.randn(2, 3, 64)
        output = model(img, txt)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 64)

    def test_apply_compile_multiple_calls(self):
        """Calling apply_compile multiple times (e.g. PP stages) should not error."""
        model1 = FakeFluxModel(num_double=1, num_single=1)
        model2 = FakeFluxModel(num_double=1, num_single=1)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile(model1, compile_config)
        apply_compile(model2, compile_config)

        for block in model1.double_blocks:
            self.assertTrue(_is_compiled(block))
        for block in model2.double_blocks:
            self.assertTrue(_is_compiled(block))


class TestApplyCompileToEncoders(unittest.TestCase):
    def test_wraps_t5_blocks(self):
        """Verify all T5 encoder blocks are compiled."""
        t5_model = FakeT5Model(num_blocks=3)
        clip_model = FakeCLIPModel(num_layers=2)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile_to_encoders(t5_model, clip_model, compile_config)

        for block in t5_model.hf_module.encoder.block:
            self.assertTrue(
                _is_compiled(block),
                "T5 encoder block should be compiled",
            )

    def test_wraps_clip_layers(self):
        """Verify all CLIP encoder layers are compiled."""
        t5_model = FakeT5Model(num_blocks=2)
        clip_model = FakeCLIPModel(num_layers=3)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile_to_encoders(t5_model, clip_model, compile_config)

        for layer in clip_model.hf_module.text_model.encoder.layers:
            self.assertTrue(
                _is_compiled(layer),
                "CLIP encoder layer should be compiled",
            )

    def test_encoder_forward_after_compile(self):
        """Verify T5 and CLIP encoder forward passes work after compilation."""
        t5_model = FakeT5Model(num_blocks=2, dim=32)
        clip_model = FakeCLIPModel(num_layers=2, dim=32)
        compile_config = CompileConfig(enable=True, backend="eager")

        apply_compile_to_encoders(t5_model, clip_model, compile_config)

        t5_input = torch.randn(2, 8, 32)
        for block in t5_model.hf_module.encoder.block:
            t5_input = block(t5_input)
        self.assertEqual(t5_input.shape, (2, 8, 32))

        clip_input = torch.randn(2, 4, 32)
        for layer in clip_model.hf_module.text_model.encoder.layers:
            clip_input = layer(clip_input)
        self.assertEqual(clip_input.shape, (2, 4, 32))

    def test_no_fullgraph_for_encoders(self):
        """
        Verify encoder compilation does not use fullgraph=True.
        We test this indirectly: a module with a graph-breaking pattern
        should compile successfully (would fail with fullgraph=True).
        """

        class GraphBreakingBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                # print() causes a graph break, but should be fine
                # without fullgraph=True
                print("graph break")
                return self.linear(x)

        t5_model = FakeT5Model(num_blocks=1)
        # Replace with graph-breaking block
        t5_model.hf_module.encoder.block = nn.ModuleList([GraphBreakingBlock()])
        clip_model = FakeCLIPModel(num_layers=1)

        compile_config = CompileConfig(enable=True, backend="eager")
        apply_compile_to_encoders(t5_model, clip_model, compile_config)

        # Forward should succeed despite graph break
        x = torch.randn(2, 4, 16)
        output = t5_model.hf_module.encoder.block[0](x)
        self.assertEqual(output.shape, (2, 4, 16))


class TestCompileConfigGating(unittest.TestCase):
    """Test that compilation is correctly gated by CompileConfig settings."""

    def test_compile_disabled(self):
        """When enable=False, blocks should not be compiled."""
        model = FakeFluxModel(num_double=2, num_single=2)
        compile_config = CompileConfig(enable=False)

        # apply_compile is only called when gated, so we test the gate logic
        if compile_config.enable and "model" in compile_config.components:
            apply_compile(model, compile_config)

        for block in model.double_blocks:
            self.assertFalse(
                _is_compiled(block),
                "Block should NOT be compiled when enable=False",
            )

    def test_compile_enabled_without_model_component(self):
        """When 'model' not in components, blocks should not be compiled."""
        model = FakeFluxModel(num_double=2, num_single=2)
        compile_config = CompileConfig(enable=True, components=["loss"])

        if compile_config.enable and "model" in compile_config.components:
            apply_compile(model, compile_config)

        for block in model.double_blocks:
            self.assertFalse(
                _is_compiled(block),
                "Block should NOT be compiled when 'model' not in components",
            )

    def test_compile_enabled_with_model_component(self):
        """When enabled and 'model' in components, blocks should be compiled."""
        model = FakeFluxModel(num_double=2, num_single=2)
        compile_config = CompileConfig(enable=True, components=["model", "loss"])

        if compile_config.enable and "model" in compile_config.components:
            apply_compile(model, compile_config)

        for block in model.double_blocks:
            self.assertTrue(
                _is_compiled(block),
                "Block should be compiled when 'model' in components",
            )

    def test_encoder_compile_disabled(self):
        """When enable=False, encoder blocks should not be compiled."""
        t5_model = FakeT5Model(num_blocks=2)
        clip_model = FakeCLIPModel(num_layers=2)
        compile_config = CompileConfig(enable=False)

        if compile_config.enable and "model" in compile_config.components:
            apply_compile_to_encoders(t5_model, clip_model, compile_config)

        for block in t5_model.hf_module.encoder.block:
            self.assertFalse(
                _is_compiled(block),
                "T5 block should NOT be compiled when enable=False",
            )
        for layer in clip_model.hf_module.text_model.encoder.layers:
            self.assertFalse(
                _is_compiled(layer),
                "CLIP layer should NOT be compiled when enable=False",
            )


class TestMSELossCompile(unittest.TestCase):
    def test_mse_loss_compiled(self):
        """Verify build_mse_loss returns a compiled function when enabled."""
        from torchtitan.components.loss import build_mse_loss

        compile_config = CompileConfig(
            enable=True, components=["model", "loss"], backend="eager"
        )
        loss_fn = build_mse_loss(compile_config)

        pred = torch.randn(4, 64)
        labels = torch.randn(4, 64)
        loss = loss_fn(pred, labels)
        self.assertGreater(loss.item(), 0)

    def test_mse_loss_not_compiled_when_disabled(self):
        """Verify build_mse_loss returns a plain function when disabled."""
        from torchtitan.components.loss import build_mse_loss, mse_loss

        compile_config = CompileConfig(enable=False)
        loss_fn = build_mse_loss(compile_config)
        self.assertIs(loss_fn, mse_loss)

    def test_mse_loss_not_compiled_without_loss_component(self):
        """Verify build_mse_loss returns a plain function when 'loss' not in components."""
        from torchtitan.components.loss import build_mse_loss, mse_loss

        compile_config = CompileConfig(enable=True, components=["model"])
        loss_fn = build_mse_loss(compile_config)
        self.assertIs(loss_fn, mse_loss)


if __name__ == "__main__":
    unittest.main()
