# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan_recipes.tests.models.qwen3_8_flash_next import (
    qwen38_flash_next_debugmodel,
)


class TestPipelineOwnership(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    @property
    def backend(self):
        return "gloo"

    @with_comms
    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_later_local_part_owns_ple_routing_and_checkpoint_state(self):
        """PLE in virtual stage 2 is prepared by the first local model part."""
        torch.set_num_threads(1)
        torch.manual_seed(42)
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=4,
            pipeline_parallel_module_fqns_per_model_part=[
                ["tok_embeddings"],
                ["layers.0"],
                ["layers.1", "layers.2", "layers.3"],
                [
                    "layers.4",
                    "layers.5",
                    "layers.6",
                    "layers.7",
                    "norm",
                    "lm_head",
                    "hyper_connection_mixer",
                ],
            ],
        )
        context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=2,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        recipe = qwen38_flash_next_debugmodel(seq_len=16, parallelism=parallelism)
        recipe.optim.optimizer.optimizers = [
            replace(config, fused=False, lr=0.02 if index == 0 else 0.01)
            for index, config in enumerate(recipe.optim.optimizer.optimizers)
        ]
        recipe.optim.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=2)
        table_key = "layers.1.ple.ple_embedding.ngram_embedding.weight"
        with context.activate_spmd(), torch.device("meta"):
            model = recipe.model.build()
        full_state = model.state_dict()
        full_keys = set(full_state)
        table_shape = tuple(full_state[table_key].shape)
        _, parts, _, has_last_stage = model.pipeline(
            parallelism_context=context,
            training=recipe.training,
            precision=recipe.precision,
            parallelism=parallelism,
            local_compile=recipe.model.local_compile,
            activation_remat=None,
            dump_folder="",
            device=torch.device("cpu"),
            model_config=recipe.model,
            loss_fn=recipe.loss.build(),
        )
        del model, full_state
        with context.activate_spmd():
            for part in parts:
                part.to_empty(device="cpu")
                part.init_states()
        pp_mesh = context.get_mesh("pp")
        owns_table = pp_mesh.get_local_rank() == 0
        dp_rank = context.get_mesh("dp").get_local_rank()
        self.assertEqual(
            [name for name, _ in parts[0].get_submodule("layers").named_children()],
            [] if owns_table else ["0"],
        )
        self.assertEqual(
            [name for name, _ in parts[1].get_submodule("layers").named_children()],
            ["1", "2", "3"] if owns_table else ["4", "5", "6", "7"],
        )
        self.assertFalse(
            {id(parameter) for parameter in parts[0].parameters()}
            & {id(parameter) for parameter in parts[1].parameters()}
        )
        local_keys = [key for part in parts for key in part.state_dict()]
        pp_keys: list[list[str]] = [[] for _ in range(pp_mesh.size())]
        dist.all_gather_object(pp_keys, local_keys, group=pp_mesh.get_group())
        all_keys = [key for keys in pp_keys for key in keys]
        self.assertEqual(set(all_keys), full_keys)
        self.assertEqual(len(all_keys), len(full_keys))
        self.assertNotIn(table_key, parts[0].state_dict())
        self.assertEqual(table_key in parts[1].state_dict(), owns_table)

        optim = recipe.optim.build(
            model_parts=parts,
            parallelism_context=context,
            training_steps=10,
            pp_has_last_stage=has_last_stage,
        )
        type(parts[0])._register_optimizer_hooks(optim.optimizers, parts, context)
        self.assertEqual(len(optim.optimizers), 3 if owns_table else 2)
        self.assertEqual(len(optim.lr_schedulers), len(optim.optimizers))
        for step in range(1, 3):
            tokens_T = torch.arange(16).add(32 * dp_rank + step)
            with context.activate_spmd():
                _, _, kwargs, _ = parts[0].preprocess_inputs(
                    {
                        "input": tokens_T,
                        "labels": tokens_T.roll(-1),
                        "positions": torch.arange(16).remainder(8),
                        "padding_mask": torch.zeros(16, dtype=torch.bool),
                    },
                    parallelism_context=context,
                    parallelism=parallelism,
                    global_routing_token_counts=torch.tensor([64.0]),
                )
            self.assertEqual("ple_routing" in kwargs, owns_table)
            if owns_table:
                table_weight_RE = parts[1].get_parameter(table_key)
                self.assertIsInstance(table_weight_RE, DTensor)
                assert isinstance(table_weight_RE, DTensor)
                self.assertEqual(
                    kwargs["ple_routing"].num_local_rows,
                    table_weight_RE.to_local().shape[0],
                )
                table_weight_RE.grad = torch.full_like(table_weight_RE, dp_rank + step)
            optim.step(torch.tensor(1.0), current_step=step)
            optim.zero_grad()

        optimizer_state = optim.optimizers.state_dict()
        checkpoint_state = ModelWrapper(parts).state_dict()
        self.assertEqual(table_key in checkpoint_state, owns_table)
        moment_key = f"state.{table_key}.exp_avg"
        self.assertEqual(moment_key in optimizer_state, owns_table)
        if owns_table:
            for value in (
                checkpoint_state[table_key],
                optimizer_state[moment_key],
                optimizer_state[f"state.{table_key}.exp_avg_sq"],
            ):
                self.assertIsInstance(value, DTensor)
                assert isinstance(value, DTensor)
                self.assertEqual(tuple(value.shape), table_shape)
                (chunk,) = value.__create_chunk_list__()
                self.assertEqual(tuple(chunk.sizes), tuple(value.to_local().shape))
                self.assertEqual(
                    tuple(chunk.offsets), (dp_rank * value.to_local().shape[0], 0)
                )
            self.assertGreater(optimizer_state[moment_key].abs().max().item(), 0)
        expected_lrs = optim.lr_schedulers.get_host_lrs_per_scheduler()
        restored = recipe.optim.build(
            model_parts=parts,
            parallelism_context=context,
            training_steps=10,
            pp_has_last_stage=has_last_stage,
        )
        type(parts[0])._register_optimizer_hooks(restored.optimizers, parts, context)
        restored.optimizers.load_state_dict(optimizer_state)
        restored.lr_schedulers.load_state_dict(optim.lr_schedulers.state_dict())
        self.assertEqual(
            restored.lr_schedulers.get_host_lrs_per_scheduler(), expected_lrs
        )
        if owns_table:
            restored_state = restored.optimizers.state_dict()
            self.assertEqual(restored_state[moment_key], optimizer_state[moment_key])
