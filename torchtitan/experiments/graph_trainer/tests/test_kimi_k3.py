# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock

import grain.python as grain
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.graph_trainer.kimi_k3 import model_registry
from torchtitan.experiments.graph_trainer.kimi_k3.config_registry import (
    graph_trainer_kimi_k3_15b_compute_bound,
    graph_trainer_kimi_k3_16b,
    graph_trainer_kimi_k3_debugmodel,
)
from torchtitan.experiments.graph_trainer.kimi_k3.data import KimiK3TextProcessor
from torchtitan.experiments.graph_trainer.kimi_k3.kda import (
    _kda_bwd,
    _kda_fwd,
    GraphTrainerKDAKernel,
)
from torchtitan.experiments.graph_trainer.kimi_k3.model import GraphTrainerKimiK3Model
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.models.kimi_k3.kda import KimiKDAKernel


class TestGraphTrainerKimiK3(TestCase):
    def test_text_processor_respects_token_batch_size(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = list(range(33))
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=16,
            num_tokens_per_batch=8,
            read_options=grain.ReadOptions(),
        )
        processor = KimiK3TextProcessor.Config().build(context=context)
        sequence = processor({"text": "unused"}, np.random.default_rng())

        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.input_ids, np.arange(8))
        self.assertEqual(sequence.labels, np.arange(1, 9))

    def test_16b_model_registry(self):
        model_spec = model_registry("16B")
        self.assertEqual(model_spec.name, "graph_trainer/kimi_k3")
        self.assertEqual(model_spec.flavor, "16B")
        self.assertIsInstance(model_spec.model, GraphTrainerKimiK3Model.Config)
        model_config = model_spec.model
        self.assertIsNone(model_config.vision_encoder)

        with torch.device("meta"):
            model = model_config.build()
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(num_parameters, 15_392_357_504)

    def test_16b_text_model_registry(self):
        model_spec = model_registry("16B-text")
        self.assertIsInstance(model_spec.model, GraphTrainerKimiK3Model.Config)
        model_config = model_spec.model
        self.assertIsNone(model_config.vision_encoder)

        with torch.device("meta"):
            model = model_config.build()
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(num_parameters, 15_392_357_504)

    def test_15b_compute_bound_model_registry(self):
        model_spec = model_registry("15B-compute-bound")
        self.assertEqual(model_spec.name, "graph_trainer/kimi_k3")
        self.assertEqual(model_spec.flavor, "15B-compute-bound")

        model_config = model_spec.model
        self.assertIsInstance(model_config, GraphTrainerKimiK3Model.Config)
        with torch.device("meta"):
            model = model_config.build()
        num_parameters, num_flops_per_token = model_config.get_nparams_and_flops(
            model, 4096
        )
        self.assertEqual(num_parameters, 14_765_996_160)
        self.assertEqual(num_flops_per_token, 17_552_855_808)
        self.assertEqual(len(model_config.layers), 64)
        self.assertEqual(
            sum(layer.attention is not None for layer in model_config.layers), 16
        )

        config = graph_trainer_kimi_k3_15b_compute_bound()
        self.assertEqual(
            config.training.num_tokens_per_microbatch_per_dp_rank, 65536
        )
        self.assertEqual(config.training.max_context_length, 4096)
        self.assertFalse(config.training.disable_cuda_graphs)
        self.assertEqual(config.parallelism.data_parallel_shard_degree, 2)
        self.assertTrue(config.compile.require_cudagraph)
        self.assertEqual(config.compile.memory_policy, "full")
        self.assertEqual(config.compile.inductor_compilation, "full")

    def test_training_configs_use_graph_kda_and_text_data(self):
        for config in (
            graph_trainer_kimi_k3_debugmodel(),
            graph_trainer_kimi_k3_16b(),
            graph_trainer_kimi_k3_15b_compute_bound(),
        ):
            if config.model_spec is None:
                raise AssertionError("Kimi K3 training config has no model spec")
            self.assertIsInstance(
                config.model_spec.model,
                GraphTrainerKimiK3Model.Config,
            )
            self.assertIsInstance(config.dataloader, GrainDataLoader.Config)
            model_config = config.model_spec.model
            dataloader_config = config.dataloader
            self.assertIsNone(model_config.vision_encoder)
            self.assertIsInstance(config.tokenizer, HuggingFaceTokenizer.Config)
            self.assertIsInstance(
                dataloader_config.dataset,
                SingleDatasetConfig,
            )
            for layer in model_config.layers:
                if layer.delta_attention is not None:
                    self.assertIsInstance(
                        layer.delta_attention.kernel,
                        GraphTrainerKDAKernel.Config,
                    )

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_kda_forward_backward_traces_as_custom_ops(self):
        kernel = GraphTrainerKDAKernel.Config(lower_bound=-5.0).build().cuda()

        def parameter(*shape: int) -> torch.Tensor:
            if len(shape) == 1:
                tensor = torch.randn(
                    shape[0] * 2,
                    device="cuda",
                    dtype=torch.bfloat16,
                )[::2]
            else:
                tensor = torch.randn(
                    *shape[:-2],
                    shape[-1],
                    shape[-2],
                    device="cuda",
                    dtype=torch.bfloat16,
                ).transpose(-1, -2)
            return tensor.detach().requires_grad_(True)

        inputs = (
            parameter(1, 64, 2, 64),
            parameter(1, 64, 2, 64),
            parameter(1, 64, 2, 64),
            parameter(1, 64, 2, 64),
            parameter(1, 64, 2),
            parameter(2),
            parameter(2, 64),
        )

        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_(True) for tensor in inputs
        )
        reference_kernel = KimiKDAKernel.Config(lower_bound=-5.0).build().cuda()
        reference_output = reference_kernel(*reference_inputs)
        reference_grads = torch.autograd.grad(reference_output.sum(), reference_inputs)
        output = kernel(*inputs)
        grads = torch.autograd.grad(output.sum(), inputs)
        self.assertEqual(output, reference_output)
        self.assertEqual(grads, reference_grads)

        grad_output = parameter(*output.shape)
        real_backward = _kda_bwd(grad_output, *inputs, -5.0)
        mode = FakeTensorMode()
        fake_inputs = tuple(mode.from_tensor(tensor.detach()) for tensor in inputs)
        fake_grad_output = mode.from_tensor(grad_output.detach())
        with mode:
            fake_output = _kda_fwd(*fake_inputs, -5.0)
            fake_backward = _kda_bwd(fake_grad_output, *fake_inputs, -5.0)
        self.assertEqual(fake_output.stride(), output.stride())
        self.assertEqual(
            [tensor.stride() for tensor in fake_backward],
            [tensor.stride() for tensor in real_backward],
        )

        def train_step(*args):
            output = kernel(*args)
            grads = torch.autograd.grad(output.sum(), args)
            return [output, *grads]

        traced = minimal_fx_tracer(train_step, module=kernel)(*inputs)
        targets = {str(node.target) for node in traced.gm.graph.nodes}
        self.assertTrue(any("graph_trainer_kda_fwd" in target for target in targets))
        self.assertTrue(any("graph_trainer_kda_bwd" in target for target in targets))


if __name__ == "__main__":
    run_tests()
