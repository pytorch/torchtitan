# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphPipelineStepRunner


class TestGraphPipelineStepRunner(unittest.TestCase):
    def test_deferred_fsdp_requires_fsdp_and_multiple_microbatches(self):
        from unittest.mock import MagicMock

        from torchtitan.config import ParallelismConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            graph_train_step_runtime,
        )

        compile_config = GraphTrainerCompileConfig(
            enable_deferred_fsdp_gradient_sync=True
        )
        common_kwargs = {
            "model": nn.Linear(2, 2),
            "parallelism": ParallelismConfig(),
            "compile_config": compile_config,
            "device": torch.device("cpu"),
            "model_config": None,
            "loss_fn": MagicMock(),
        }
        with self.assertRaisesRegex(ValueError, "requires FSDP"):
            graph_train_step_runtime(
                num_microbatches=2,
                parallel_dims=MagicMock(fsdp_enabled=False),
                **common_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            graph_train_step_runtime(
                num_microbatches=1,
                parallel_dims=MagicMock(fsdp_enabled=True),
                **common_kwargs,
            )

    def test_rejects_monolithic_precompile_artifact(self):
        from unittest.mock import MagicMock

        from torchtitan.config import ParallelismConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            graph_train_step_runtime,
        )

        with self.assertRaisesRegex(
            ValueError,
            "artifacts contain one monolithic train-step graph",
        ):
            graph_train_step_runtime(
                nn.Linear(2, 2),
                num_microbatches=1,
                parallel_dims=MagicMock(),
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(
                    precompile_artifact_dir="artifacts",
                ),
                device=torch.device("cpu"),
                model_config=None,
                loss_fn=MagicMock(),
            )

    def test_rejects_unpartitioned_graph_features(self):
        from unittest.mock import MagicMock

        from torchtitan.config import ParallelismConfig
        from torchtitan.experiments.graph_trainer.configs import (
            EpOverlapConfig,
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            graph_train_step_runtime,
        )

        configs_and_errors = (
            (
                GraphTrainerCompileConfig(ep_overlap=EpOverlapConfig(enabled=True)),
                "trace-input preparers",
            ),
            (
                GraphTrainerCompileConfig(memory_policy="sac_and_offload"),
                "preserve offload and reload pairs",
            ),
        )
        for compile_config, error in configs_and_errors:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                graph_train_step_runtime(
                    nn.Linear(2, 2),
                    num_microbatches=1,
                    parallel_dims=MagicMock(),
                    parallelism=ParallelismConfig(),
                    compile_config=compile_config,
                    device=torch.device("cpu"),
                    model_config=None,
                    loss_fn=MagicMock(),
                )

    def test_pipeline_runner_executes_gradient_accumulation_as_one_stage(self):
        from unittest.mock import MagicMock

        import torch.distributed as dist

        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            graph_train_step_runtime,
        )

        dist.init_process_group(
            "gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        try:
            torch.manual_seed(42)
            model = nn.Linear(4, 3, dtype=torch.float64)
            reference = nn.Linear(4, 3, dtype=torch.float64)
            reference.load_state_dict(model.state_dict())
            parallel_dims = ParallelDims(1, 1, 1, 1, 1, 1, 1)
            parallel_dims.build_mesh()

            def loss_fn(pred, target, global_valid_tokens):
                return ((pred - target) ** 2).sum() / global_valid_tokens

            runtime = graph_train_step_runtime(
                model,
                num_microbatches=3,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(
                    enable=False,
                    enable_passes=False,
                ),
                device=torch.device("cpu"),
                model_config=None,
                loss_fn=loss_fn,
            )
            schedule_actions = runtime.schedule.pipeline_order_with_comms[0]
            self.assertEqual(
                sum(
                    action.computation_type.name == "REDUCE_GRAD"
                    for action in schedule_actions
                ),
                0,
            )
            self.assertEqual(
                sum(
                    action.computation_type.name == "UNSHARD"
                    for action in schedule_actions
                ),
                3,
            )
            runner = GraphPipelineStepRunner(
                runtime,
                num_microbatches=3,
                use_cuda_graph=False,
            )
            inputs = [torch.randn(2, 4, dtype=torch.float64) for _ in range(3)]
            targets = [torch.randn(2, 3, dtype=torch.float64) for _ in range(3)]
            global_valid_tokens = torch.tensor(18, dtype=torch.float64)

            expected_loss = torch.zeros((), dtype=torch.float64)
            for value, target in zip(inputs, targets, strict=True):
                loss = loss_fn(reference(value), target, global_valid_tokens)
                loss.backward()
                expected_loss += loss.detach()

            actual_loss = runner(
                tuple(
                    (value, target, {})
                    for value, target in zip(inputs, targets, strict=True)
                ),
                global_valid_tokens,
            )

            torch.testing.assert_close(actual_loss, expected_loss)
            for actual, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(actual.grad, expected.grad)

            reuse_runtime = graph_train_step_runtime(
                nn.Linear(4, 3),
                num_microbatches=3,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(fsdp_reshard_after_forward="never"),
                compile_config=GraphTrainerCompileConfig(
                    enable=False,
                    enable_passes=False,
                ),
                device=torch.device("cpu"),
                model_config=None,
                loss_fn=loss_fn,
            )
            reuse_action_names = [
                action.computation_type.name
                for action in reuse_runtime.schedule.pipeline_order_with_comms[0]
            ]
            self.assertEqual(reuse_action_names.count("UNSHARD"), 1)
            self.assertEqual(reuse_action_names.count("RESHARD"), 1)
            self.assertEqual(reuse_action_names.count("REDUCE_GRAD"), 0)

            deferred_parallel_dims = MagicMock(fsdp_enabled=True)
            deferred_parallel_dims.get_optional_mesh.return_value = (
                parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
            )
            deferred_runtime = graph_train_step_runtime(
                nn.Linear(4, 3),
                num_microbatches=3,
                parallel_dims=deferred_parallel_dims,
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(
                    enable=False,
                    enable_passes=False,
                    enable_deferred_fsdp_gradient_sync=True,
                ),
                device=torch.device("cpu"),
                model_config=None,
                loss_fn=loss_fn,
            )
            deferred_action_names = [
                action.computation_type.name
                for action in deferred_runtime.schedule.pipeline_order_with_comms[0]
            ]
            self.assertEqual(deferred_action_names.count("REDUCE_GRAD"), 1)
        finally:
            dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_graph_captures_complete_accumulation_window(self):
        import torch.distributed as dist

        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed import ParallelDims
        from torchtitan.distributed.cudagraph import cudagraph_teardown
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            graph_train_step_runtime,
        )

        dist.init_process_group(
            "gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        try:
            torch.manual_seed(42)
            model = nn.Linear(4, 3, device="cuda")
            reference = nn.Linear(4, 3, device="cuda")
            reference.load_state_dict(model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            parallel_dims = ParallelDims(1, 1, 1, 1, 1, 1, 1)
            parallel_dims.build_mesh()

            def loss_fn(pred, target, global_valid_tokens):
                return ((pred - target) ** 2).sum() / global_valid_tokens

            runtime = graph_train_step_runtime(
                model,
                num_microbatches=3,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(
                    enable=False,
                    enable_passes=False,
                ),
                device=torch.device("cuda"),
                model_config=None,
                loss_fn=loss_fn,
                use_cuda_graph=True,
            )
            runner = GraphPipelineStepRunner(
                runtime,
                num_microbatches=3,
                use_cuda_graph=True,
            )
            global_valid_tokens = torch.tensor(18, device="cuda")

            for _ in range(4):
                optimizer.zero_grad(set_to_none=False)
                reference_optimizer.zero_grad(set_to_none=False)
                microbatches = tuple(
                    (
                        torch.randn(2, 4, device="cuda"),
                        torch.randn(2, 3, device="cuda"),
                        {},
                    )
                    for _ in range(3)
                )
                expected_loss = torch.zeros((), device="cuda")
                for inputs, target, _ in microbatches:
                    loss = loss_fn(reference(inputs), target, global_valid_tokens)
                    loss.backward()
                    expected_loss.add_(loss.detach())

                actual_loss = runner(microbatches, global_valid_tokens)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual_loss, expected_loss)
                for actual, expected in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(actual.grad, expected.grad)
                optimizer.step()
                reference_optimizer.step()
        finally:
            cudagraph_teardown()
            dist.destroy_process_group()
