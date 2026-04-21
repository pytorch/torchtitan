# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import types

import torch
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.custom_codegen import (
    _CodegenGraphModule,
    custom_codegen_pass,
)


class TestCustomCodegenPass(TestCase):
    """Tests for custom_codegen_pass - dump to disk, record_function, hot-reload.

    All tests run with enrich_profiler_metadata=True to match production behavior.
    """

    def setUp(self):
        import torch.fx.experimental._config as fx_config

        self._original_enrich = fx_config.enrich_profiler_metadata
        fx_config.enrich_profiler_metadata = True
        self._codegen_dir = tempfile.mkdtemp(prefix="test_codegen_")

    def tearDown(self):
        import torch.fx.experimental._config as fx_config

        fx_config.enrich_profiler_metadata = self._original_enrich
        shutil.rmtree(self._codegen_dir, ignore_errors=True)

    def test_custom_codegen_dual_path_generation(self):
        """Test that custom_codegen_pass generates two-path code with if/else."""
        codegen_dir = os.path.join(self._codegen_dir, "test_dual_path")

        class SimpleModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y * 2

        module = SimpleModule()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        exported = torch.export.export(module, (x, y))
        gm = exported.module()

        custom_gm = custom_codegen_pass(gm, codegen_dir=codegen_dir)
        self.assertIsInstance(custom_gm, _CodegenGraphModule)
        self.assertIsNotNone(custom_gm._code_path)
        self.assertTrue(os.path.exists(custom_gm._code_path))

        with open(custom_gm._code_path, "r") as f:
            code = f.read()

        # Verify dual-path structure
        self.assertIn("_autograd_profiler._is_profiler_enabled", code)
        self.assertIn("_RecordFunctionFast", code)
        self.assertIn("if _autograd_profiler._is_profiler_enabled:", code)
        self.assertIn("else:", code)

        # Verify _forward_impl does NOT contain _RecordFunctionFast
        impl_start = code.find("def _forward_impl(")
        profiled_start = code.find("def _forward_profiled(")
        self.assertGreaterEqual(impl_start, 0, "_forward_impl not found")
        self.assertGreaterEqual(profiled_start, 0, "_forward_profiled not found")
        impl_section = code[impl_start:profiled_start]
        self.assertNotIn(
            "_RecordFunctionFast",
            impl_section,
            "_forward_impl should NOT contain _RecordFunctionFast",
        )

        # Verify _forward_profiled DOES contain _RecordFunctionFast
        profiled_section = code[profiled_start:]
        self.assertIn(
            "_RecordFunctionFast",
            profiled_section,
            "_forward_profiled should contain _RecordFunctionFast",
        )

        # Verify execution works (without profiler)
        result = custom_gm(x, y)
        expected = gm(x, y)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_custom_codegen_serializable(self):
        """Test that custom_codegen_pass result is serializable."""
        codegen_dir = os.path.join(self._codegen_dir, "test_serializable")

        class SimpleModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y * 2

        module = SimpleModule()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        exported = torch.export.export(module, (x, y))
        gm = exported.module()

        custom_gm = custom_codegen_pass(gm, codegen_dir=codegen_dir)
        expected = custom_gm(x, y)
        self.assertIsInstance(custom_gm, _CodegenGraphModule)
        state = custom_gm.__getstate__()
        shutil.rmtree(codegen_dir, ignore_errors=True)
        custom_gm.__setstate__(state)
        actual = custom_gm(x, y)
        self.assertTrue(torch.allclose(actual, expected))

    def test_custom_codegen_with_profiler_and_hot_reload(self):
        """Test profiler integration and hot-reload in one test."""
        codegen_dir = os.path.join(self._codegen_dir, "test_hot_reload")

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        module = SimpleModule()
        x = torch.randn(4, 4)

        exported = torch.export.export(module, (x,))
        gm = exported.module()

        custom_gm = custom_codegen_pass(gm, codegen_dir=codegen_dir)
        self.assertIsInstance(custom_gm, _CodegenGraphModule)
        self.assertIsNotNone(custom_gm._code_path)
        self.assertTrue(os.path.exists(custom_gm._code_path))

        with open(custom_gm._code_path, "r") as f:
            code = f.read()
        self.assertIn("_RecordFunctionFast", code)
        self.assertIn("_autograd_profiler._is_profiler_enabled", code)

        # Test execution without profiler
        expected = gm(x)
        result = custom_gm(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

        # Test execution with profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            result_profiled = custom_gm(x)
        self.assertTrue(torch.allclose(result_profiled, expected, atol=1e-5))
        self.assertGreater(len([e.key for e in prof.key_averages()]), 0)

        # Test hot-reload: modify code and verify new results
        original_hash = custom_gm._code_hash

        # Replace the multiplier from 2 to 4 (doubles the result)
        modified_code = code.replace(", 2);  x = None", ", 4);  x = None")
        self.assertNotEqual(modified_code, code, "Code should be modified")

        with open(custom_gm._code_path, "w") as f:
            f.write(modified_code)

        # Force check that modification is detected
        self.assertTrue(
            custom_gm._check_file_modified(), "File modification should be detected"
        )

        # Verify hot-reload detects modification and loads new code
        modified_result = custom_gm(x)

        # Verify the hash changed (file was re-read)
        self.assertNotEqual(
            custom_gm._code_hash,
            original_hash,
            "Code hash should change after reload",
        )

        self.assertTrue(torch.allclose(modified_result, expected * 2, atol=1e-5))

    def test_invoke_subgraph(self):
        """Test that custom_codegen_pass processes subgraph modules into a single file.

        This happens when invoke_subgraph HOP is used. Verify child
        GraphModules are included as module-level functions in a single file,
        loaded and executed at runtime.
        """
        codegen_dir = os.path.join(self._codegen_dir, "test_invoke_subgraph")
        os.makedirs(codegen_dir, exist_ok=True)

        # Create main module
        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        class SubgraphModule(torch.nn.Module):
            def forward(self, x, y):
                return x * y + 1

        module = MainModule()
        x = torch.randn(4, 10)
        exported = torch.export.export(module, (x,))
        gm = exported.module()
        # Attach subgraph GraphModules as children (this is what invoke_subgraph does)
        gm.repeated_subgraph0 = torch.fx.symbolic_trace(SubgraphModule())

        processed_gm = custom_codegen_pass(gm, codegen_dir=codegen_dir)

        self.assertIsInstance(processed_gm, _CodegenGraphModule)
        self.assertTrue(hasattr(processed_gm, "repeated_subgraph0"))

        # Verify single file output (no separate subgraph directories)
        py_files = [f for f in os.listdir(codegen_dir) if f.endswith(".py")]
        self.assertEqual(len(py_files), 1, f"Expected single .py file, got {py_files}")

        # Verify single file contains both main module and subgraph functions
        with open(os.path.join(codegen_dir, py_files[0]), "r") as f:
            content = f.read()

        # Main module profiler instrumentation
        self.assertIn("_forward_profiled", content)
        self.assertIn("_RecordFunctionFast", content)

        # Verify main module: _forward_impl does NOT contain _RecordFunctionFast
        main_impl_start = content.find("def _forward_impl(")
        main_profiled_start = content.find("def _forward_profiled(")
        self.assertGreaterEqual(main_impl_start, 0, "main _forward_impl not found")
        self.assertGreaterEqual(
            main_profiled_start, 0, "main _forward_profiled not found"
        )
        main_impl = content[main_impl_start:main_profiled_start]
        self.assertNotIn(
            "_RecordFunctionFast",
            main_impl,
            "main _forward_impl should NOT contain _RecordFunctionFast",
        )
        self.assertIn(
            "_RecordFunctionFast",
            content[main_profiled_start:],
            "main _forward_profiled should contain _RecordFunctionFast",
        )

        # Subgraph functions (loaded and bound at runtime)
        self.assertIn("# ===== Subgraph: repeated_subgraph0 =====", content)
        self.assertIn("_repeated_subgraph0_forward_impl", content)
        self.assertIn("_repeated_subgraph0_forward_profiled", content)
        self.assertIn("_repeated_subgraph0_forward", content)

        # Verify subgraph: _forward_impl does NOT contain _RecordFunctionFast
        sub_impl_start = content.find("def _repeated_subgraph0_forward_impl(")
        sub_profiled_start = content.find("def _repeated_subgraph0_forward_profiled(")
        self.assertGreaterEqual(sub_impl_start, 0, "subgraph _forward_impl not found")
        self.assertGreaterEqual(
            sub_profiled_start, 0, "subgraph _forward_profiled not found"
        )
        sub_impl = content[sub_impl_start:sub_profiled_start]
        self.assertNotIn(
            "_RecordFunctionFast",
            sub_impl,
            "subgraph _forward_impl should NOT contain _RecordFunctionFast",
        )
        self.assertIn(
            "_RecordFunctionFast",
            content[sub_profiled_start:],
            "subgraph _forward_profiled should contain _RecordFunctionFast",
        )

        # Verify subgraph forward was bound (the subgraph can be called)
        subgraph = processed_gm.repeated_subgraph0
        self.assertTrue(hasattr(subgraph, "forward"))
        self.assertIsInstance(subgraph.forward, types.MethodType)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
