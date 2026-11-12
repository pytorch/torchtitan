# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.rl.generator.model import ModelStateDictLoader


def _copy_from(new_state_dict: dict[str, torch.Tensor]):
    """A fetch that copies from `new_state_dict`, as the generator's cpu_prefetch fetch does."""

    def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
        with torch.no_grad():
            for key, tensor in state_dict.items():
                tensor.copy_(new_state_dict[key])
        return list(state_dict)

    return fetch


def test_load_model_state_dict_requires_every_param(tmp_path) -> None:
    """Raise when fetch skips a parameter, whose storage was just reallocated; an
    unfilled buffer (e.g. a vLLM attention scale the trainer does not publish) keeps
    its value."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Module()
        model.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
        model.register_buffer("scale", torch.ones(4))
        mesh = init_device_mesh("cpu", (1,))
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        loader = ModelStateDictLoader(
            model, checkpointer_config=None, model_state_dict_load_mode="gpu_fetch"
        )

        def fetch_published(published: set[str]):
            def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
                return [key for key in state_dict if key in published]

            return fetch

        params = {name for name, _ in model.named_parameters()}
        loader.load(fetch_published(params))
        assert model.scale.equal(torch.ones(4))

        with pytest.raises(RuntimeError, match="did not fill 1 weights.*layers.1.bias"):
            loader.load(fetch_published(params - {"layers.1.bias"}))
    finally:
        dist.destroy_process_group()


def test_load_model_state_dict_fetches_params_in_their_own_dtype(tmp_path) -> None:
    """Fetch every parameter in its own dtype, as the trainer sends it: a float32
    parameter (preserve_dtype_params, e.g. GDN A_log) next to bfloat16 ones."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Module()
        model.layer = nn.Linear(4, 4, dtype=torch.bfloat16)
        model.A_log = nn.Parameter(torch.zeros(4), requires_grad=False)
        mesh = init_device_mesh("cpu", (1,))
        fully_shard(model.layer, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1 for k, v in model.state_dict().items()}
        loader = ModelStateDictLoader(
            model, checkpointer_config=None, model_state_dict_load_mode="gpu_fetch"
        )
        fetched_dtypes = {}
        copy = _copy_from(new_state_dict)

        def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
            fetched_dtypes.update({k: v.dtype for k, v in state_dict.items()})
            return copy(state_dict)

        loader.load(fetch)

        assert fetched_dtypes == {k: v.dtype for k, v in new_state_dict.items()}
        assert fetched_dtypes["A_log"] == torch.float32
        assert model.A_log.equal(new_state_dict["A_log"].full_tensor())
        assert model.layer.weight.equal(new_state_dict["layer.weight"].full_tensor())
    finally:
        dist.destroy_process_group()


class TestLoadModelStateDictTwoRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_fetches_shards_into_the_unsharded_params(self) -> None:
        """Load twice; each rank fetches its own (uneven) shards, one FSDP group
        allocated at a time, and every rank ends with the full new weights.

        Guards that each group's sharded buffers are allocated before its fetch and
        freed after, and that a group left mid-forward (the tied group after a
        standalone lm_head call) still receives the new weights, all-gathered outside
        the forward state: in it, FSDP all-gathers on prefetch streams that do not wait
        for the load's writes on GPU.
        """
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(15, 16)
        model.layers = nn.ModuleList(nn.Linear(16, 15) for _ in range(2))
        model.lm_head = nn.Linear(16, 15, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        model.register_buffer("scale", torch.zeros(4))
        mesh = init_device_mesh("cpu", (self.world_size,))
        fully_shard(
            [model.tok_embeddings, model.lm_head],
            mesh=mesh,
            reshard_after_forward=False,
        )
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        initial_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        loader = ModelStateDictLoader(
            model, checkpointer_config=None, model_state_dict_load_mode="gpu_fetch"
        )
        groups = [
            group
            for module in (model.tok_embeddings, *model.layers)
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]

        def is_allocated(group) -> bool:
            return all(
                p._sharded_param_data.untyped_storage().size() > 0
                for p in group.fsdp_params
            )

        # As in compute_logits: calling lm_head alone leaves the tied group's forward
        # incomplete, so module.reshard() would skip it.
        model._get_fsdp_state()._lazy_init()  # the root's forward would do this
        with torch.no_grad():
            model.lm_head(torch.zeros(1, 16))

        comm_ctx = model._get_fsdp_state()._comm_ctx
        get_all_gather_streams = comm_ctx.get_all_gather_streams
        all_gather_states = []

        def record_all_gather_state(async_op, training_state):
            all_gather_states.append(training_state)
            return get_all_gather_streams(async_op, training_state)

        for version in (1, 2):
            new_state_dict = {k: v + version for k, v in initial_state_dict.items()}
            fetched_keys = []

            copy = _copy_from(new_state_dict)

            def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
                fetched_keys.append(sorted(state_dict))
                # Exactly the fetched group holds sharded buffers; none for buffers.
                self.assertEqual(
                    sum(map(is_allocated, groups)), int("scale" not in state_dict)
                )
                return copy(state_dict)

            with mock.patch.object(
                comm_ctx, "get_all_gather_streams", record_all_gather_state
            ):
                loader.load(fetch)
            self.assertEqual(set(all_gather_states), {TrainingState.IDLE})

            self.assertEqual(
                fetched_keys,
                [
                    ["lm_head.weight", "tok_embeddings.weight"],
                    ["layers.0.bias", "layers.0.weight"],
                    ["layers.1.bias", "layers.1.weight"],
                    ["scale"],
                ],
            )
            self.assertFalse(any(map(is_allocated, groups)))
            for name, param in model.named_parameters(remove_duplicate=False):
                self.assertEqual(param, new_state_dict[name].full_tensor(), msg=name)
            self.assertEqual(model.scale, new_state_dict["scale"])

    @with_comms
    def test_unshards_every_group_after_fetch_error(self) -> None:
        """Unshard every group after a failed fetch on one rank, then raise.

        unshard() all-gathers across the FSDP group, so if rank 1 stopped after
        its fetch failed, rank 0 would hang in the next group's all-gather.
        """
        model = nn.Module()
        model.layers = nn.ModuleList(nn.Linear(16, 16) for _ in range(3))
        model.register_buffer("scale", torch.zeros(4))
        mesh = init_device_mesh("cpu", (self.world_size,))
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        loader = ModelStateDictLoader(
            model, checkpointer_config=None, model_state_dict_load_mode="gpu_fetch"
        )
        fetched_keys = []

        def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
            fetched_keys.append(sorted(state_dict))
            if self.rank == 1:
                raise RuntimeError("fetch failed")
            return list(state_dict)

        if self.rank == 1:
            with self.assertRaisesRegex(RuntimeError, "fetch failed"):
                loader.load(fetch)
            self.assertEqual(fetched_keys, [["layers.0.bias", "layers.0.weight"]])
        else:
            loader.load(fetch)
            self.assertEqual(len(fetched_keys), 4)
        self.assertTrue(
            all(
                p._sharded_param_data.untyped_storage().size() == 0
                for block in model.layers
                for group in block._get_fsdp_state()._fsdp_param_groups
                for p in group.fsdp_params
            )
        )
