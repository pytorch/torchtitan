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

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.rl.generator import model as generator_model
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.model import ModelStateDictLoader


def _copy_from(new_state_dict: dict[str, torch.Tensor]):
    """A fetch that copies from `new_state_dict`, as the generator's cpu_prefetch fetch does."""

    def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
        with torch.no_grad():
            for key, tensor in state_dict.items():
                tensor.copy_(new_state_dict[key])
        return list(state_dict)

    return fetch


def _copy_published(new_state_dict: dict[str, torch.Tensor], published: set[str]):
    """A fetch that fills only the ``published`` keys, as a TorchStore read of a push
    that left the others out."""
    copy = _copy_from(new_state_dict)

    def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
        return copy({k: v for k, v in state_dict.items() if k in published})

    return fetch


def _build_loader(
    model: nn.Module,
    *,
    loads_checkpoint: bool,
    frozen: frozenset[str] = frozenset(),
    model_state_dict_load_mode: ModelStateDictLoadMode = "gpu_fetch",
) -> ModelStateDictLoader:
    """Build the loader as a generator does, for a model whose config freezes
    ``frozen``. The checkpoint load is skipped."""
    with mock.patch.object(generator_model, "_load_initial_weights"):
        return ModelStateDictLoader(
            model,
            checkpointer_config=(
                CheckpointManager.Config(initial_load_path="/base")
                if loads_checkpoint
                else None
            ),
            model_state_dict_load_mode=model_state_dict_load_mode,
            frozen_param_keys=frozen,
            hf_assets_path="/hf",
        )


@pytest.mark.parametrize("loads_checkpoint", [False, True])
def test_load_model_state_dict_leaves_out_only_frozen_params(
    tmp_path, loads_checkpoint: bool
) -> None:
    """Every load must fill every parameter, whose storage was just reallocated, except
    the frozen ones (e.g. a LoRA base, held since the checkpoint load), which keep
    their weight. A group of only frozen parameters (layers.1) is never resharded. An
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
        initial_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        params = {name for name, _ in model.named_parameters()}
        loader = _build_loader(
            model,
            loads_checkpoint=loads_checkpoint,
            frozen=(
                frozenset(params - {"layers.0.weight"})
                if loads_checkpoint
                else frozenset()
            ),
        )
        loaded = {k: v + 1 for k, v in initial_state_dict.items()}

        loader.load(_copy_published(loaded, params))
        if not loads_checkpoint:
            with pytest.raises(
                RuntimeError, match="did not fill 1 weights.*layers.1.bias"
            ):
                loader.load(_copy_published(loaded, params - {"layers.1.bias"}))
            return
        later = {k: v + 2 for k, v in initial_state_dict.items()}
        with mock.patch.object(
            generator_model, "_reshard", wraps=generator_model._reshard
        ) as reshard:
            loader.load(_copy_published(later, {"layers.0.weight"}))
        resharded = [call.args[0] for call in reshard.call_args_list]
        assert model.layers[0] in resharded
        assert model.layers[1] not in resharded

        # Published or not, the frozen parameters are never fetched.
        for name, param in model.named_parameters():
            expected = later if name == "layers.0.weight" else initial_state_dict
            assert param.equal(expected[name].full_tensor()), name
        assert model.scale.equal(torch.ones(4))
        # A trainable parameter is never left out.
        with pytest.raises(
            RuntimeError, match="did not fill 1 weights.*layers.0.weight"
        ):
            loader.load(_copy_published(later, set()))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("loads_checkpoint", [False, True])
def test_receive_buffers_skip_frozen_params(
    tmp_path, monkeypatch, loads_checkpoint: bool
) -> None:
    """A generator receives only the trainable parameters (e.g. LoRA adapters) and the
    buffers, so it keeps cpu_prefetch receive buffers for those only. Without frozen
    parameters (and so without a checkpoint load) it receives the whole model."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Module()
        model.base = nn.Linear(4, 4)
        model.lora_a = nn.Linear(4, 2, bias=False)
        model.register_buffer("scale", torch.ones(4))
        frozen = frozenset({"base.weight", "base.bias"} if loads_checkpoint else ())
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        model.set_keep_unsharded_storage(True)
        # Pinned host memory needs a GPU; keep the receive buffers' state dict as is.
        monkeypatch.setattr(
            generator_model,
            "_create_cpu_state_dict",
            lambda state_dict, pin_memory: dict(state_dict),
        )
        loader = _build_loader(
            model,
            loads_checkpoint=loads_checkpoint,
            frozen=frozen,
            model_state_dict_load_mode="cpu_prefetch",
        )

        assert loader.prefetched_model_state_dict is not None
        assert set(loader.prefetched_model_state_dict) == (
            {"base.weight", "base.bias", "lora_a.weight", "scale"} - frozen
        )
    finally:
        dist.destroy_process_group()


def test_frozen_params_need_a_checkpoint_load() -> None:
    """Weight syncs never carry frozen parameters, so without a checkpoint load nothing
    would fill them."""
    with pytest.raises(ValueError, match="Set generator.checkpointer"):
        _build_loader(
            nn.Linear(2, 2), loads_checkpoint=False, frozen=frozenset({"weight"})
        )


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
            model,
            checkpointer_config=None,
            model_state_dict_load_mode="gpu_fetch",
            frozen_param_keys=frozenset(),
            hf_assets_path="/hf",
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
            model,
            checkpointer_config=None,
            model_state_dict_load_mode="gpu_fetch",
            frozen_param_keys=frozenset(),
            hf_assets_path="/hf",
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
    def test_partial_load_keeps_the_shards_it_leaves_out(self) -> None:
        """A load that fills only the trainable parameters (LoRA adapters) keeps every
        frozen parameter on every rank: each rank refills its own (uneven) shard from the
        unsharded weight before the group's all-gather."""
        model = nn.Module()
        model.layers = nn.ModuleList(nn.Linear(16, 15) for _ in range(2))
        mesh = init_device_mesh("cpu", (self.world_size,))
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        initial_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        params = {name for name, _ in model.named_parameters()}
        loader = _build_loader(
            model,
            loads_checkpoint=True,
            frozen=frozenset(params - {"layers.1.weight"}),
        )
        first = {k: v + 1 for k, v in initial_state_dict.items()}
        second = {k: v + 2 for k, v in initial_state_dict.items()}

        loader.load(_copy_published(first, params))
        loader.load(_copy_published(second, {"layers.1.weight"}))

        for name, param in model.named_parameters():
            expected = second if name == "layers.1.weight" else initial_state_dict
            self.assertEqual(param, expected[name].full_tensor(), msg=name)

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
            model,
            checkpointer_config=None,
            model_state_dict_load_mode="gpu_fetch",
            frozen_param_keys=frozenset(),
            hf_assets_path="/hf",
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
