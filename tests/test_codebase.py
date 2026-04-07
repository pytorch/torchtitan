"""
Tests to verify the stripped-down torchtitan codebase works correctly.
Covers: imports, config, models, tokenizer, protocols, and components.
"""

import os
import sys
import tempfile
import unittest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImports(unittest.TestCase):
    """Verify all core modules import without errors."""

    def _assert_importable(self, module_path: str, names: list[str]):
        """Assert that names can be imported from module_path."""
        import importlib

        mod = importlib.import_module(module_path)
        for name in names:
            self.assertTrue(hasattr(mod, name), f"{module_path}.{name} not found")

    def test_torchtitan_package(self):
        import torchtitan

        self.assertTrue(hasattr(torchtitan, "__version__"))

    def test_config_imports(self):
        self._assert_importable(
            "torchtitan.config",
            [
                "JobConfig",
                "ConfigManager",
                "TORCH_DTYPE_MAP",
                "Job",
                "Model",
                "Training",
                "Parallelism",
                "Checkpoint",
                "ActivationCheckpoint",
                "Comm",
                "Profiling",
                "Validation",
                "Debug",
                "Optimizer",
                "LRScheduler",
                "Metrics",
            ],
        )

    def test_protocols_imports(self):
        self._assert_importable(
            "torchtitan.protocols",
            [
                "BaseModelArgs",
                "ModelProtocol",
                "StateDictAdapter",
                "BaseStateDictAdapter",
            ],
        )
        self._assert_importable(
            "torchtitan.protocols.train_spec",
            ["TrainSpec", "get_train_spec"],
        )

    def test_components_imports(self):
        self._assert_importable(
            "torchtitan.components.checkpoint", ["CheckpointManager"]
        )
        self._assert_importable("torchtitan.components.dataloader", ["BaseDataLoader"])
        self._assert_importable(
            "torchtitan.components.loss", ["build_cross_entropy_loss"]
        )
        self._assert_importable(
            "torchtitan.components.lr_scheduler", ["build_lr_schedulers"]
        )
        self._assert_importable(
            "torchtitan.components.metrics", ["MetricsProcessor", "WandBLogger"]
        )
        self._assert_importable(
            "torchtitan.components.optimizer",
            ["OptimizersContainer", "build_optimizers"],
        )
        self._assert_importable(
            "torchtitan.components.tokenizer",
            ["HuggingFaceTokenizer", "build_hf_tokenizer"],
        )
        self._assert_importable("torchtitan.components.validate", ["build_validator"])

    def test_distributed_imports(self):
        self._assert_importable(
            "torchtitan.distributed", ["ParallelDims", "NoParallel"]
        )
        self._assert_importable(
            "torchtitan.distributed.activation_checkpoint", ["apply_ac"]
        )
        self._assert_importable(
            "torchtitan.distributed.pipeline_parallel", ["pipeline_llm"]
        )
        self._assert_importable(
            "torchtitan.distributed.expert_parallel", ["ExpertParallel"]
        )

    def test_model_imports(self):
        self._assert_importable("torchtitan.models", ["_supported_models"])
        self._assert_importable("torchtitan.models.attention", ["FlexAttentionWrapper"])
        self._assert_importable("torchtitan.models.moe", ["MoEArgs", "build_moe"])
        self._assert_importable(
            "torchtitan.models.parallelize",
            ["apply_fsdp", "apply_moe_ep_tp", "apply_compile"],
        )

    def test_tools_imports(self):
        self._assert_importable("torchtitan.tools.logging", ["logger"])
        self._assert_importable(
            "torchtitan.tools.profiling",
            ["maybe_enable_profiling", "maybe_enable_memory_snapshot"],
        )
        self._assert_importable(
            "torchtitan.tools.utils", ["device_type", "get_peak_flops"]
        )

    def test_train_import(self):
        self._assert_importable("torchtitan.train", ["Trainer", "main"])


class TestSupportedModels(unittest.TestCase):
    """Verify only the expected models remain."""

    def test_supported_models_list(self):
        from torchtitan.models import _supported_models

        self.assertEqual(_supported_models, frozenset(["llama3", "deepseek_v3"]))

    def test_llama3_train_spec(self):
        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("llama3")
        self.assertEqual(spec.model_cls.__name__, "Transformer")
        self.assertIn("debugmodel", spec.model_args)
        self.assertIn("8B", spec.model_args)
        self.assertIn("70B", spec.model_args)
        self.assertIn("405B", spec.model_args)
        self.assertIsNotNone(spec.parallelize_fn)
        self.assertIsNotNone(spec.pipelining_fn)
        self.assertIsNotNone(spec.build_optimizers_fn)
        self.assertIsNotNone(spec.build_dataloader_fn)
        self.assertIsNotNone(spec.build_loss_fn)
        self.assertIsNotNone(spec.state_dict_adapter)

    def test_deepseek_v3_train_spec(self):
        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("deepseek_v3")
        self.assertEqual(spec.model_cls.__name__, "DeepSeekV3Model")
        self.assertIn("debugmodel", spec.model_args)
        self.assertIn("16B", spec.model_args)
        self.assertIn("671B", spec.model_args)
        self.assertIsNotNone(spec.state_dict_adapter)

    def test_unknown_model_raises(self):
        from torchtitan.protocols.train_spec import get_train_spec

        with self.assertRaises(ValueError):
            get_train_spec("nonexistent_model")

    def test_removed_models_raise(self):
        from torchtitan.protocols.train_spec import get_train_spec

        for name in ["flux", "gpt_oss", "qwen3", "llama3_ft", "llama4"]:
            with self.assertRaises((ValueError, ModuleNotFoundError)):
                get_train_spec(name)


class TestConfig(unittest.TestCase):
    """Verify config parsing and removed fields."""

    def test_job_config_defaults(self):
        from torchtitan.config import JobConfig

        config = JobConfig()
        self.assertEqual(config.model.name, "llama3")
        self.assertEqual(config.model.flavor, "debugmodel")
        self.assertTrue(hasattr(config, "training"))
        self.assertTrue(hasattr(config, "parallelism"))
        self.assertTrue(hasattr(config, "checkpoint"))
        self.assertTrue(hasattr(config, "validation"))

    def test_removed_config_fields(self):
        from torchtitan.config import JobConfig

        config = JobConfig()
        # These should all be gone
        self.assertFalse(hasattr(config, "experimental"))
        self.assertFalse(hasattr(config, "quantize"))
        self.assertFalse(hasattr(config, "fault_tolerance"))
        self.assertFalse(hasattr(config, "memory_estimation"))
        self.assertFalse(hasattr(config.model, "tokenizer_path"))
        self.assertFalse(hasattr(config.model, "converters"))
        self.assertFalse(hasattr(config.model, "print_after_conversion"))
        self.assertFalse(hasattr(config.checkpoint, "initial_load_in_hf_quantized"))
        self.assertFalse(hasattr(config.checkpoint, "enable_ft_dataloader_checkpoints"))
        self.assertFalse(hasattr(config.metrics, "enable_tensorboard"))

    def test_config_from_toml(self):
        from torchtitan.config import ConfigManager

        cm = ConfigManager()
        config = cm.parse_args(
            [
                "--job.config_file",
                "torchtitan/models/llama3/train_configs/debug_model.toml",
            ]
        )
        self.assertEqual(config.model.name, "llama3")
        self.assertEqual(config.model.flavor, "debugmodel")
        self.assertEqual(config.training.steps, 10)
        self.assertEqual(config.training.local_batch_size, 8)

    def test_cli_override(self):
        from torchtitan.config import ConfigManager

        cm = ConfigManager()
        config = cm.parse_args(
            [
                "--job.config_file",
                "torchtitan/models/llama3/train_configs/debug_model.toml",
                "--training.steps=99",
                "--training.local_batch_size=4",
            ]
        )
        self.assertEqual(config.training.steps, 99)
        self.assertEqual(config.training.local_batch_size, 4)

    def test_deepseek_config(self):
        from torchtitan.config import ConfigManager

        cm = ConfigManager()
        config = cm.parse_args(
            [
                "--job.config_file",
                "torchtitan/models/deepseek_v3/train_configs/debug_model.toml",
            ]
        )
        self.assertEqual(config.model.name, "deepseek_v3")
        self.assertEqual(config.model.flavor, "debugmodel")


class TestTokenizer(unittest.TestCase):
    """Verify tokenizer loads correctly from tokenizer.json."""

    def _create_test_tokenizer_dir(self):
        """Create a minimal tokenizer.json for testing using the tokenizers library."""
        tmpdir = tempfile.mkdtemp()
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        # Build a simple WordLevel tokenizer programmatically
        vocab = {f"tok{i}": i for i in range(100)}
        vocab["[UNK]"] = 100
        vocab["hello"] = 101
        vocab["world"] = 102
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.save(os.path.join(tmpdir, "tokenizer.json"))
        return tmpdir

    def test_load_tokenizer(self):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tmpdir = self._create_test_tokenizer_dir()
        try:
            tok = HuggingFaceTokenizer(tmpdir)
            self.assertIsNotNone(tok.tokenizer)
            self.assertGreater(tok.get_vocab_size(), 0)
            self.assertEqual(tok.vocab_size, tok.get_vocab_size())
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_encode_decode(self):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tmpdir = self._create_test_tokenizer_dir()
        try:
            tok = HuggingFaceTokenizer(tmpdir)
            ids = tok.encode("hello")
            self.assertIsInstance(ids, list)
            self.assertTrue(all(isinstance(i, int) for i in ids))
            decoded = tok.decode(ids)
            self.assertIsInstance(decoded, str)
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_missing_tokenizer_raises(self):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        with self.assertRaises(FileNotFoundError):
            HuggingFaceTokenizer("/nonexistent/path")

    def test_base_tokenizer_interface(self):
        from torchtitan.components.tokenizer import BaseTokenizer

        self.assertTrue(hasattr(BaseTokenizer, "encode"))
        self.assertTrue(hasattr(BaseTokenizer, "decode"))
        self.assertTrue(hasattr(BaseTokenizer, "get_vocab_size"))


class TestModelConstruction(unittest.TestCase):
    """Verify models can be instantiated on meta device."""

    def test_llama3_debug_model(self):
        import torch

        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("llama3")
        args = spec.model_args["debugmodel"]
        with torch.device("meta"):
            model = spec.model_cls(args)
        self.assertIsNotNone(model)

    def test_deepseek_v3_debug_model(self):
        import torch

        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("deepseek_v3")
        args = spec.model_args["debugmodel"]
        with torch.device("meta"):
            model = spec.model_cls(args)
        self.assertIsNotNone(model)

    def test_llama3_model_param_count(self):
        import torch

        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("llama3")
        args = spec.model_args["debugmodel"]
        with torch.device("meta"):
            model = spec.model_cls(args)
        nparams, nflops = args.get_nparams_and_flops(model, seq_len=2048)
        self.assertGreater(nparams, 0)
        self.assertGreater(nflops, 0)


class TestMoE(unittest.TestCase):
    """Verify MoE building blocks work."""

    def test_moe_args(self):
        from torchtitan.models.moe import MoEArgs

        args = MoEArgs(num_experts=8, top_k=2)
        self.assertEqual(args.num_experts, 8)
        self.assertEqual(args.top_k, 2)

    def test_build_moe_on_meta(self):
        import torch

        from torchtitan.models.moe import MoEArgs, build_moe

        args = MoEArgs(num_experts=4, top_k=2)
        with torch.device("meta"):
            moe = build_moe(args, dim=64, hidden_dim=128)
        self.assertIsNotNone(moe)


class TestTrainSpec(unittest.TestCase):
    """Verify the TrainSpec protocol and registration."""

    def test_train_spec_fields(self):
        from torchtitan.protocols.train_spec import get_train_spec

        spec = get_train_spec("llama3")
        self.assertIsNotNone(spec.model_cls)
        self.assertIsNotNone(spec.model_args)
        self.assertIsNotNone(spec.parallelize_fn)
        self.assertIsNotNone(spec.build_optimizers_fn)
        self.assertIsNotNone(spec.build_lr_schedulers_fn)
        self.assertIsNotNone(spec.build_dataloader_fn)
        self.assertIsNotNone(spec.build_loss_fn)

    def test_register_custom_train_spec(self):
        from torchtitan.protocols.train_spec import (
            TrainSpec,
            _extra_train_specs,
            get_train_spec,
            register_train_spec,
        )

        # Register a dummy spec
        dummy_spec = TrainSpec(
            model_cls=type("DummyModel", (), {}),
            model_args={},
            parallelize_fn=lambda *a: None,
            pipelining_fn=None,
            build_optimizers_fn=lambda *a: None,
            build_lr_schedulers_fn=lambda *a: None,
            build_dataloader_fn=lambda *a: None,
            build_tokenizer_fn=None,
            build_loss_fn=lambda *a: None,
        )
        register_train_spec("test_dummy", dummy_spec)
        retrieved = get_train_spec("test_dummy")
        self.assertEqual(retrieved.model_cls.__name__, "DummyModel")
        # Cleanup
        del _extra_train_specs["test_dummy"]


class TestNoRemovedModules(unittest.TestCase):
    """Verify removed modules/files no longer exist."""

    def test_no_experiments_dir(self):
        self.assertFalse(os.path.exists("torchtitan/experiments"))

    def test_no_quantization_dir(self):
        self.assertFalse(os.path.exists("torchtitan/components/quantization"))

    def test_no_ft_dir(self):
        self.assertFalse(os.path.exists("torchtitan/components/ft"))

    def test_no_deepep_dir(self):
        self.assertFalse(os.path.exists("torchtitan/distributed/deepep"))

    def test_no_model_converter(self):
        self.assertFalse(os.path.exists("torchtitan/protocols/model_converter.py"))

    def test_no_moe_deepep(self):
        self.assertFalse(os.path.exists("torchtitan/models/moe/moe_deepep.py"))

    def test_no_removed_model_dirs(self):
        for model in ["flux", "gpt_oss", "qwen3", "llama3_ft", "llama4"]:
            self.assertFalse(
                os.path.exists(f"torchtitan/models/{model}"),
                f"Model dir {model} should have been removed",
            )

    def test_no_docs_dir(self):
        self.assertFalse(os.path.exists("docs"))

    def test_no_scripts_dir(self):
        self.assertFalse(os.path.exists("scripts"))


class TestFlattenedModelStructure(unittest.TestCase):
    """Verify model directories are properly flattened."""

    def test_llama3_flat(self):
        base = "torchtitan/models/llama3"
        self.assertTrue(os.path.exists(f"{base}/args.py"))
        self.assertTrue(os.path.exists(f"{base}/model_def.py"))
        self.assertTrue(os.path.exists(f"{base}/parallelize.py"))
        self.assertTrue(os.path.exists(f"{base}/state_dict_adapter.py"))
        self.assertFalse(os.path.exists(f"{base}/model"))
        self.assertFalse(os.path.exists(f"{base}/infra"))

    def test_deepseek_v3_flat(self):
        base = "torchtitan/models/deepseek_v3"
        self.assertTrue(os.path.exists(f"{base}/args.py"))
        self.assertTrue(os.path.exists(f"{base}/model_def.py"))
        self.assertTrue(os.path.exists(f"{base}/parallelize.py"))
        self.assertTrue(os.path.exists(f"{base}/state_dict_adapter.py"))
        self.assertFalse(os.path.exists(f"{base}/model"))
        self.assertFalse(os.path.exists(f"{base}/infra"))


if __name__ == "__main__":
    unittest.main()
