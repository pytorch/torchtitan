# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan_recipes.tests.models.qwen3_8_flash_next import (
    qwen38_flash_next_debugmodel,
)


pytestmark = pytest.mark.multi_gpu


class TestFlashNextFSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_owned_ple_training_and_async_checkpoint(self):
        """Exercise FSDP+EP, persistent rows, fused AdamW and remat."""
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=self.world_size, expert_parallel_degree=2
        )
        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        recipe = qwen38_flash_next_debugmodel(seq_len=32, parallelism=parallelism)
        device = torch.device(self.device_type, self.rank)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        checkpoint_pg = dist.new_group(backend="gloo")
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path, src=0)
        try:
            for use_remat in (False, True):
                with self.subTest(use_remat=use_remat):
                    with torch.device("meta"):
                        model = recipe.model.build()
                    model.parallelize(
                        parallelism_context=pc,
                        precision=PrecisionConfig(),
                        parallelism=parallelism,
                        local_compile=LocalCompileConfig(regions=[]),
                        activation_remat=FullActivationRematConfig()
                        if use_remat
                        else None,
                        dump_folder=path[0],
                    )
                    model.to_empty(device=device)
                    model.init_states()
                    moe = model.layers["0"].feed_forward
                    self.assertEqual(
                        moe.routed_experts.w13.weight.to_local().shape[0] * 2,
                        moe.num_experts,
                    )
                    optimizer = recipe.optim.optimizer.build(model_parts=[model])
                    type(model)._register_optimizer_hooks(optimizer, [model], pc)
                    table = model.layers["1"].ple.ple_embedding.ngram_embedding
                    parameter = table.weight
                    self.assertIsInstance(parameter, DTensor)
                    self.assertEqual(
                        parameter.to_local().shape[0] * self.world_size,
                        table.num_embeddings,
                    )
                    observed_rows = []
                    hook = table.register_forward_pre_hook(
                        lambda module, _: observed_rows.append(
                            module.weight.to_local().shape[0]
                        )
                    )
                    # Match engine initialization: remat needs this thread's SPMD mesh.
                    with (
                        torch.autograd.set_multithreading_enabled(False),
                        sdpa_kernel(SDPBackend.MATH),
                    ):
                        for microbatch in range(2):
                            tokens = torch.arange(32, device=device).add(
                                32 * self.rank + microbatch
                            )
                            input_dict = {
                                "input": tokens,
                                "labels": tokens.roll(-1),
                                "positions": torch.arange(32, device=device).remainder(
                                    16
                                ),
                                "padding_mask": torch.zeros(
                                    32, device=device, dtype=torch.bool
                                ),
                            }
                            with pc.activate_spmd():
                                (
                                    input_T,
                                    labels_T,
                                    model_kwargs,
                                    _,
                                ) = model.preprocess_inputs(
                                    input_dict,
                                    parallelism_context=pc,
                                    parallelism=parallelism,
                                    global_routing_token_counts=torch.tensor(
                                        [32.0 * 2 * self.world_size], device=device
                                    ),
                                )
                                logits_TV = model(input_T, **model_kwargs)
                                loss = F.cross_entropy(
                                    logits_TV.float(), labels_T, reduction="sum"
                                )
                                (loss / (32 * 2 * self.world_size)).backward()
                            self.assertIs(table.weight, parameter)
                            self.assertEqual(parameter.dtype, torch.float32)
                    hook.remove()
                    self.assertTrue(observed_rows)
                    self.assertEqual(
                        set(observed_rows), {parameter.to_local().shape[0]}
                    )
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(
                        torch.isfinite(
                            clip_grad_norm_(model.parameters(), 0.5, foreach=True)
                        )
                    )
                    optimizer.step()
                    wrapper = ModelWrapper(model)
                    lookup_optimizer = next(iter(optimizer))
                    expected_weight = parameter.detach().clone()
                    expected_moments = {
                        key: lookup_optimizer.state[parameter][key].clone()
                        for key in ("exp_avg", "exp_avg_sq")
                    }
                    checkpoint_path = path[0] + f"/{use_remat}"
                    state = {"model": wrapper, "optimizer": optimizer}
                    dcp.async_save(
                        state,
                        checkpoint_id=checkpoint_path,
                        process_group=checkpoint_pg,
                    ).result()
                    with torch.no_grad():
                        parameter.zero_()
                        for key in expected_moments:
                            lookup_optimizer.state[parameter][key].zero_()
                    dcp.load(
                        state,
                        checkpoint_id=checkpoint_path,
                        process_group=checkpoint_pg,
                    )
                    self.assertEqual(parameter, expected_weight, atol=0, rtol=0)
                    for key, expected in expected_moments.items():
                        self.assertEqual(
                            lookup_optimizer.state[parameter][key],
                            expected,
                            atol=0,
                            rtol=0,
                        )
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path[0])
