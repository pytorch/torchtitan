# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


class TestPrecompileMain(unittest.TestCase):
    def test_attention_metadata_reaches_tracing_with_and_without_mla(self):
        from graph_trainer import make_fx_tracer, precompile

        from torchtitan.experiments.graph_trainer import precompile_main
        from torchtitan.models.common.attention import (
            FlexInnerAttention,
            MLAAttention,
            VarlenInnerAttention,
        )
        from torchtitan.models.common.decoder import Decoder
        from torchtitan.models.common.linear import Linear

        for kernel in (FlexInnerAttention, VarlenInnerAttention):
            for use_mla in (False, True):
                with self.subTest(kernel=kernel.__name__, use_mla=use_mla):
                    backend = kernel.Config()
                    if use_mla:
                        backend = MLAAttention.Config(
                            wkv_b=Linear.Config(in_features=4, out_features=8),
                            packed_kv_head_dim=4,
                            inner_attention=backend,
                        )
                    model_config = MagicMock(spec=Decoder.Config)
                    model_config.layers = [
                        SimpleNamespace(
                            attention=SimpleNamespace(inner_attention=backend)
                        )
                    ]
                    model_config.vocab_size = 16
                    model_config.traverse.return_value = iter(())
                    model_config.first_moe = None
                    model = MagicMock()
                    metadata = {backend._owner: object()}
                    model._get_attention_metadata.return_value = metadata
                    config = SimpleNamespace(
                        loss=MagicMock(),
                        training=SimpleNamespace(
                            num_tokens_per_microbatch_per_dp_rank=16,
                            max_context_length=8,
                        ),
                    )
                    parallelism_context = MagicMock(
                        dp_shard=1,
                        dp_replicate=1,
                        cp=1,
                        cp_enabled=False,
                        tp_enabled=False,
                    )
                    trace = MagicMock(
                        side_effect=RuntimeError("stop before compilation")
                    )
                    with (
                        patch.object(
                            make_fx_tracer, "minimal_fx_tracer", return_value=trace
                        ),
                        patch.object(
                            precompile, "get_spmd_precompile_meshes", return_value=[]
                        ),
                        patch.object(
                            precompile_main, "maybe_register_blockmask_pytree_node"
                        ),
                        self.assertRaisesRegex(RuntimeError, "stop before compilation"),
                    ):
                        precompile_main._precompile_aot_fx_trace(
                            config,
                            model,
                            model_config,
                            SimpleNamespace(),
                            parallelism_context,
                            torch.device("cpu"),
                            tokenizer=None,
                        )

                    model._get_attention_metadata.assert_called_once()
                    trace.assert_called_once()
                    extra_kwargs = trace.call_args.args[3]
                    self.assertIs(extra_kwargs["attention_metadata"], metadata)
                    self.assertIs(
                        model._get_attention_metadata.call_args.kwargs["positions"],
                        extra_kwargs["positions"],
                    )
                    torch.testing.assert_close(
                        extra_kwargs["positions"],
                        torch.arange(16, dtype=torch.int32) % 8,
                    )

    def test_validates_memory_policy_after_model_setup(self):
        from torchtitan.experiments.graph_trainer import precompile_main

        events = []
        compile_config = SimpleNamespace()
        config = SimpleNamespace(compile=compile_config)
        config_loader = MagicMock()
        config_loader.load.return_value = config
        setup_result = (
            object(),
            object(),
            compile_config,
            object(),
            object(),
            object(),
        )

        def common_setup(_config):
            events.append("setup")
            return setup_result

        def validate(actual_compile_config):
            self.assertIs(actual_compile_config, compile_config)
            self.assertEqual(events, ["setup"])
            events.append("validate")

        def precompile(*_args):
            self.assertEqual(events, ["setup", "validate"])
            events.append("precompile")

        with (
            patch.object(precompile_main, "ConfigLoader", return_value=config_loader),
            patch.object(precompile_main, "_common_setup", side_effect=common_setup),
            patch.object(
                precompile_main,
                "validate_memory_policy_config",
                side_effect=validate,
            ),
            patch.object(
                precompile_main,
                "_precompile_aot_fx_trace",
                side_effect=precompile,
            ),
            patch.object(precompile_main.dist, "destroy_process_group"),
        ):
            precompile_main.main()

        self.assertEqual(events, ["setup", "validate", "precompile"])


class TestPrecompileLossSetup(unittest.TestCase):
    def test_chunked_loss_setup_matches_trainer_boundary(self):
        from torchtitan.experiments.graph_trainer.chunked_loss import (
            ChunkedLossWrapperWithParamGrads,
        )
        from torchtitan.experiments.graph_trainer.precompile_main import (
            _prepare_loss_for_precompile,
        )

        lm_head = torch.nn.Linear(2, 3)
        model = SimpleNamespace(lm_head=lm_head, _skip_lm_head=False)
        loss_fn = ChunkedLossWrapperWithParamGrads.Config().build()

        _prepare_loss_for_precompile(model, loss_fn)

        self.assertIs(loss_fn.lm_head, lm_head)
        self.assertTrue(model._skip_lm_head)


class TestCudaGraphPass(unittest.TestCase):
    """Test cuda_graph_pass behavior."""

    def test_core_cuda_graph_adds_only_annotation_pass(self):
        from graph_trainer.cuda_graph import insert_kernel_annotations_pass

        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.passes import (
            final_inductor_compile_passes,
        )

        compile_config = GraphTrainerCompileConfig(inductor_compilation="regional")
        without_cuda_graph = final_inductor_compile_passes(compile_config)
        with_cuda_graph = final_inductor_compile_passes(
            compile_config,
            use_cuda_graph=True,
        )

        self.assertNotIn(insert_kernel_annotations_pass, without_cuda_graph)
        self.assertEqual(with_cuda_graph.count(insert_kernel_annotations_pass), 1)


class TestPrecompiledPassSelection(unittest.TestCase):
    def test_no_load_time_graph_passes(self):
        from torchtitan.experiments.graph_trainer.passes import (
            construct_default_graph_passes,
        )

        config = SimpleNamespace(
            compile=SimpleNamespace(precompile_artifact_dir="artifacts")
        )
        self.assertEqual(construct_default_graph_passes(MagicMock(), config), [])


if __name__ == "__main__":
    unittest.main()
