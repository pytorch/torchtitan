# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import contextmanager
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import (
    flex_shard,
    get_global_shape,
    get_placements,
)
from torchtitan.experiments.flex_shard.example.muon import (
    build_comm_free_muon_optimizers,
    build_muon_param_groups,
    CombinedOptimizer,
    comm_free_muon_buckets,
    GroupedMuon,
)
from torchtitan.experiments.flex_shard.example.owned import (
    assign_layer_owners_lpt,
    make_owned_placement_fn,
    Owned,
)
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    expected_shard,
    make_transformer_model,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


def _init_params_deterministically(model: nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _average_reference_grads(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def _is_owned_2d(param: nn.Parameter) -> bool:
    """Whether build_muon_param_groups routes this param to Muon (owned 2D matrix)."""
    placements = get_placements(param)
    global_shape = get_global_shape(param)
    return (
        placements is not None
        and isinstance(placements[0], Owned)
        and global_shape is not None
        and len(global_shape) == 2
    )


def _checkpoint_execution_units(model: nn.Module) -> None:
    """Wrap each execution-unit module in activation checkpointing (recompute all)."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def context_fn():
        def policy_fn(ctx, func, *args, **kwargs):
            return CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(policy_fn)

    model.tok_embeddings = checkpoint_wrapper(
        model.tok_embeddings, context_fn=context_fn
    )
    model.pos_embeddings = checkpoint_wrapper(
        model.pos_embeddings, context_fn=context_fn
    )
    for idx, layer in enumerate(model.layers):
        model.layers[idx] = checkpoint_wrapper(layer, context_fn=context_fn)
    model.norm = checkpoint_wrapper(model.norm, context_fn=context_fn)
    model.output = checkpoint_wrapper(model.output, context_fn=context_fn)


def _assert_owned_muon_parity(
    testcase: FSDPTest,
    reference: nn.Module,
    flex_model: nn.Module,
    rank: int,
    world_size: int,
    *,
    attr: str,
) -> None:
    """Compare a flex-sharded model's local tensors against a full reference.

    ``Owned`` params must equal the full reference on the owner rank (and be empty
    elsewhere); ``Shard`` params must equal the reference's local chunk. ``attr`` is
    ``"data"`` or ``"grad"``. Params are matched by order, since reshard-after-forward
    inserts checkpoint wrappers that change the flex model's FQNs.
    """
    for (_, ref), (_, param) in zip(
        reference.named_parameters(),
        flex_model.named_parameters(),
        strict=True,
    ):
        flex_value = param.detach() if attr == "data" else param.grad
        ref_value = ref.detach() if attr == "data" else ref.grad
        placement = get_placements(param)[0]
        if isinstance(placement, Owned):
            if rank == placement.owner_rank:
                testcase.assertIsNotNone(flex_value)
                testcase.assertEqual(flex_value, ref_value)
            else:
                testcase.assertEqual(param.numel(), 0)
        else:
            testcase.assertIsNotNone(flex_value)
            testcase.assertEqual(
                flex_value,
                expected_shard(ref_value, rank=rank, world_size=world_size),
            )


@contextmanager
def _assert_no_collectives(testcase: FSDPTest):
    """Fail if any torch.distributed collective runs inside the ``with`` block."""
    collective_names = [
        "all_gather",
        "all_gather_into_tensor",
        "_all_gather_base",
        "all_reduce",
        "broadcast",
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "_reduce_scatter_base",
    ]
    observed: list[str] = []
    patchers = [
        mock.patch.object(
            dist,
            name,
            side_effect=lambda *a, _n=name, **k: observed.append(_n),
        )
        for name in collective_names
        if hasattr(dist, name)
    ]
    for patcher in patchers:
        patcher.start()
    try:
        yield observed
    finally:
        for patcher in patchers:
            patcher.stop()
    testcase.assertEqual(observed, [], f"optimizer step issued collectives: {observed}")


class TestCommFreeMuonHelpers(TestCase):
    """CPU-only tests for the placement/bucket helpers (no FlexShard runtime)."""

    def test_assign_layer_owners_lpt_homogeneous(self) -> None:
        self.assertEqual(assign_layer_owners_lpt([100, 100, 100, 100], 2), [0, 1, 0, 1])

    def test_assign_layer_owners_lpt_heterogeneous(self) -> None:
        self.assertEqual(assign_layer_owners_lpt([10, 1, 1, 1, 1], 2), [0, 1, 1, 1, 1])

    def test_assign_layer_owners_lpt_more_ranks_than_layers(self) -> None:
        self.assertEqual(assign_layer_owners_lpt([100], 4), [0])

    def test_assign_layer_owners_lpt_balances_load(self) -> None:
        numels = [7, 6, 5, 4, 3, 2, 1]
        owners = assign_layer_owners_lpt(numels, 3)
        loads = [0, 0, 0]
        for numel, owner in zip(numels, owners, strict=True):
            loads[owner] += numel
        self.assertLessEqual(max(loads) - min(loads), max(numels))

    def test_assign_layer_owners_lpt_rejects_bad_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "world_size"):
            assign_layer_owners_lpt([1, 2], 0)

    def test_make_owned_placement_fn_assigns_single_owner(self) -> None:
        placement_fn = make_owned_placement_fn(1)
        params = [
            ("a", nn.Parameter(torch.zeros(2, 2))),
            ("b", nn.Parameter(torch.zeros(3))),
        ]
        placements = placement_fn(params, None)
        self.assertEqual(set(placements), {"a", "b"})
        for value in placements.values():
            (placement,) = value
            self.assertIsInstance(placement, Owned)
            self.assertEqual(placement.owner_rank, 1)

    def test_comm_free_muon_buckets_structure(self) -> None:
        num_layers = 4
        world_size = 2
        rest_patterns = ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"]
        _, model = make_transformer_model(n_layers=num_layers)
        buckets = comm_free_muon_buckets(model, world_size)
        expected_owners = assign_layer_owners_lpt([100] * num_layers, world_size)

        self.assertEqual(len(buckets), num_layers + len(rest_patterns))
        for i in range(num_layers):
            self.assertEqual(buckets[i].patterns, [f"layers.{i}.*"])
            self.assertTrue(buckets[i].reshard_after_forward)
            placements = buckets[i].placement_fn(
                [(f"layers.{i}.attention.wq.weight", nn.Parameter(torch.zeros(2, 2)))],
                None,
            )
            (placement,) = next(iter(placements.values()))
            self.assertIsInstance(placement, Owned)
            self.assertEqual(placement.owner_rank, expected_owners[i])

        # Each rest pattern is its own bucket (reshard-after-forward safe).
        rest_buckets = buckets[num_layers:]
        self.assertEqual(
            [bucket.patterns for bucket in rest_buckets],
            [[pattern] for pattern in rest_patterns],
        )
        for bucket in rest_buckets:
            placements = bucket.placement_fn(
                [("output.weight", nn.Parameter(torch.zeros(2, 2)))],
                None,
            )
            (placement,) = next(iter(placements.values()))
            self.assertIsInstance(placement, Shard)

    def test_comm_free_muon_buckets_reshard_after_forward_false(self) -> None:
        _, model = make_transformer_model(n_layers=2)
        buckets = comm_free_muon_buckets(model, 2, reshard_after_forward=False)
        for bucket in buckets:
            self.assertFalse(bucket.reshard_after_forward)

    def test_comm_free_muon_buckets_rejects_unknown_balance(self) -> None:
        _, model = make_transformer_model(n_layers=2)
        with self.assertRaisesRegex(ValueError, "balance"):
            comm_free_muon_buckets(model, 2, balance="bogus")

    def test_reshard_policy_recomputes_unshard_collectives(self) -> None:
        from torch.utils.checkpoint import CheckpointPolicy

        from torchtitan.experiments.flex_shard.flex_shard.reshard_after_forward import (
            _reshard_after_forward_policy,
        )

        for op in (
            torch.ops.c10d.broadcast_.default,
            torch.ops._c10d_functional.broadcast.default,
            torch.ops.c10d.allgather_.default,
        ):
            self.assertEqual(
                _reshard_after_forward_policy(None, op),
                CheckpointPolicy.MUST_RECOMPUTE,
            )
        self.assertEqual(
            _reshard_after_forward_policy(None, torch.ops.aten.mm.default),
            CheckpointPolicy.PREFER_RECOMPUTE,
        )

    def test_grouped_muon_matches_per_expert_torch_muon(self) -> None:
        # GroupedMuon on a (num_experts, m, n) stack must equal running
        # torch.optim.Muon on each expert's 2D slice separately.
        torch.manual_seed(0)
        num_experts, m, n = 6, 48, 32
        kwargs = dict(lr=0.02, momentum=0.9, weight_decay=0.01, nesterov=True)
        w0 = torch.randn(num_experts, m, n)

        grouped = nn.Parameter(w0.clone())
        grouped_opt = GroupedMuon([grouped], **kwargs)

        per_expert = [nn.Parameter(w0[e].clone()) for e in range(num_experts)]
        per_expert_opt = torch.optim.Muon(per_expert, **kwargs)

        for _ in range(3):  # multiple steps to exercise the momentum buffer
            g = torch.randn(num_experts, m, n)
            grouped_opt.zero_grad(set_to_none=True)
            per_expert_opt.zero_grad(set_to_none=True)
            grouped.grad = g.clone()
            for e in range(num_experts):
                per_expert[e].grad = g[e].clone()
            grouped_opt.step()
            per_expert_opt.step()

        expected = torch.stack([p.detach() for p in per_expert])
        self.assertEqual(grouped.detach(), expected)

    def test_grouped_muon_matches_per_matrix_for_4d(self) -> None:
        # >3D stacks (e.g. (E1, E2, m, n)) flatten leading dims and still match.
        torch.manual_seed(1)
        shape = (2, 3, 40, 24)
        m, n = shape[-2], shape[-1]
        kwargs = dict(lr=0.02, momentum=0.9, weight_decay=0.0)
        w0 = torch.randn(*shape)

        grouped = nn.Parameter(w0.clone())
        grouped_opt = GroupedMuon([grouped], **kwargs)

        flat0 = w0.reshape(-1, m, n)
        per_matrix = [nn.Parameter(flat0[i].clone()) for i in range(flat0.shape[0])]
        per_matrix_opt = torch.optim.Muon(per_matrix, **kwargs)

        for _ in range(2):
            g = torch.randn(*shape)
            grouped_opt.zero_grad(set_to_none=True)
            per_matrix_opt.zero_grad(set_to_none=True)
            grouped.grad = g.clone()
            gf = g.reshape(-1, m, n)
            for i in range(gf.shape[0]):
                per_matrix[i].grad = gf[i].clone()
            grouped_opt.step()
            per_matrix_opt.step()

        expected = torch.stack([p.detach() for p in per_matrix]).reshape(*shape)
        self.assertEqual(grouped.detach(), expected)

    def test_grouped_muon_rejects_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "ndim >= 3"):
            GroupedMuon([nn.Parameter(torch.zeros(4, 4))], lr=0.01)


class TestCommFreeMuon(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    @skip_if_lt_x_gpu(2)
    def test_param_groups_partition_by_ownership(self) -> None:
        mesh = self._mesh()
        _, model = make_transformer_model(device=device_type.type, n_layers=4)
        flex_shard(
            model,
            mesh,
            buckets=comm_free_muon_buckets(model, self.world_size),
        )

        muon_params, other_params = build_muon_param_groups(model, mesh)

        self.assertGreater(len(muon_params), 0)
        for param in muon_params:
            self.assertEqual(param.dim(), 2)
            self.assertGreater(param.numel(), 0)
            placement = get_placements(param)[0]
            self.assertIsInstance(placement, Owned)
            self.assertEqual(placement.owner_rank, self.rank)
        for param in other_params:
            self.assertGreater(param.numel(), 0)
            placement = get_placements(param)[0]
            owned_here_2d = (
                isinstance(placement, Owned)
                and placement.owner_rank == self.rank
                and param.dim() == 2
            )
            self.assertFalse(owned_here_2d)

        grouped = {id(p) for p in muon_params} | {id(p) for p in other_params}
        for _, param in model.named_parameters():
            if param.numel() == 0:
                self.assertNotIn(id(param), grouped)

    def _run_muon_parity(
        self,
        *,
        reshard_after_forward: bool,
        checkpoint_layers: bool,
    ) -> None:
        mesh = self._mesh()
        args, model = make_transformer_model(device=device_type.type, n_layers=4)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        if checkpoint_layers:
            _checkpoint_execution_units(model)
            _checkpoint_execution_units(reference)

        flex_shard(
            model,
            mesh,
            buckets=comm_free_muon_buckets(
                model,
                self.world_size,
                reshard_after_forward=reshard_after_forward,
            ),
        )

        muon_kwargs = dict(lr=0.02, momentum=0.9, weight_decay=0.0)
        adamw_kwargs = dict(lr=0.01, weight_decay=0.0)
        optim = build_comm_free_muon_optimizers(
            model,
            mesh,
            muon_kwargs=muon_kwargs,
            adamw_kwargs=adamw_kwargs,
        )

        # Reference: the same Muon / AdamW split on the full (unsharded) params.
        # Match by parameter order, since reshard-after-forward inserts checkpoint
        # wrappers that change the flex model's FQNs.
        ref_muon: list[nn.Parameter] = []
        ref_other: list[nn.Parameter] = []
        for (_, ref_param), (_, flex_param) in zip(
            reference.named_parameters(),
            model.named_parameters(),
            strict=True,
        ):
            if _is_owned_2d(flex_param):
                ref_muon.append(ref_param)
            else:
                ref_other.append(ref_param)
        ref_optim = CombinedOptimizer(
            [
                torch.optim.Muon(ref_muon, **muon_kwargs),
                torch.optim.AdamW(ref_other, **adamw_kwargs),
            ]
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()
        ref_loss.backward()
        _average_reference_grads(reference)

        _assert_owned_muon_parity(
            self, reference, model, self.rank, self.world_size, attr="grad"
        )

        optim.step()
        ref_optim.step()

        _assert_owned_muon_parity(
            self, reference, model, self.rank, self.world_size, attr="data"
        )

    @skip_if_lt_x_gpu(2)
    def test_matches_single_device_muon(self) -> None:
        self._run_muon_parity(reshard_after_forward=False, checkpoint_layers=False)

    @skip_if_lt_x_gpu(2)
    def test_matches_single_device_muon_reshard_after_forward(self) -> None:
        self._run_muon_parity(reshard_after_forward=True, checkpoint_layers=False)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_composes_with_activation_checkpointing(self) -> None:
        self._run_muon_parity(reshard_after_forward=True, checkpoint_layers=True)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_recomputes_broadcast_in_backward(self) -> None:
        # With reshard, each Owned layer's broadcast runs in forward and again in
        # the backward recompute; without reshard it runs only in forward. This
        # confirms the unsharded param is actually freed and rebroadcast.
        mesh = self._mesh()

        def count_broadcasts(reshard_after_forward: bool) -> int:
            args, model = make_transformer_model(device=device_type.type, n_layers=4)
            _init_params_deterministically(model)
            flex_shard(
                model,
                mesh,
                buckets=comm_free_muon_buckets(
                    model,
                    self.world_size,
                    reshard_after_forward=reshard_after_forward,
                ),
            )
            torch.manual_seed(42 + self.rank + 1)
            x = transformer_inputs(args, batch_size=3, device=device_type)
            real_broadcast = dist.broadcast
            count = 0

            def counting_broadcast(*a, **k):
                nonlocal count
                count += 1
                return real_broadcast(*a, **k)

            with mock.patch.object(dist, "broadcast", side_effect=counting_broadcast):
                model(x).sum().backward()
            return count

        without_reshard = count_broadcasts(False)
        with_reshard = count_broadcasts(True)
        self.assertGreater(without_reshard, 0)
        self.assertGreater(with_reshard, without_reshard)

    @skip_if_lt_x_gpu(2)
    def test_optimizer_step_is_communication_free(self) -> None:
        mesh = self._mesh()
        args, model = make_transformer_model(device=device_type.type, n_layers=4)
        _init_params_deterministically(model)
        flex_shard(
            model,
            mesh,
            buckets=comm_free_muon_buckets(model, self.world_size),
        )
        optim = build_comm_free_muon_optimizers(
            model,
            mesh,
            muon_kwargs=dict(lr=0.02, momentum=0.9, weight_decay=0.0),
            adamw_kwargs=dict(lr=0.01, weight_decay=0.0),
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)
        optim.zero_grad(set_to_none=True)
        model(x).sum().backward()

        with _assert_no_collectives(self):
            optim.step()


if __name__ == "__main__":
    run_tests()
