# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
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
    BucketSpec,
    flex_shard,
    get_global_shape,
    get_placements,
)
from torchtitan.experiments.flex_shard.example.muon import (
    _default_muon_predicate,
    build_comm_free_muon_optimizers,
    build_muon_param_groups,
    build_ragged_shard_muon_optimizers,
    CombinedOptimizer,
    comm_free_muon_buckets,
    grouped_ragged_shard_muon_buckets,
    GroupedMuon,
)
from torchtitan.experiments.flex_shard.example.owned import (
    assign_layer_owners_lpt,
    assign_matrix_owners_per_layer_balanced,
    make_owned_placement_fn,
    Owned,
)
from torchtitan.experiments.flex_shard.example.ragged_shard import (
    GroupedRaggedShard,
    RaggedShard,
)
from torchtitan.experiments.flex_shard.example.shard import per_param_placements, Shard
from torchtitan.experiments.flex_shard.tests.common import (
    expected_shard,
    make_transformer_model,
    transformer_inputs,
)


device_type = torch.device(get_devtype())

try:
    from torchtitan.models.deepseek_v3 import deepseekv3_configs as _deepseekv3_configs
except Exception:  # pragma: no cover - deepseek_v3 may be unavailable in some envs
    _deepseekv3_configs = None


class _FakeMesh:
    """A 1D ``DeviceMesh`` stand-in for single-process bucket-structure tests.

    The bucket builders only need ``mesh.size()`` (to derive the world size) and
    stamp the mesh onto each ``BucketSpec``; the structure tests inspect bucket
    patterns and placements, not the mesh, so a real distributed mesh is not
    required to check the layout for a hypothetical world size.
    """

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _build_deepseekv3_kimi_model() -> nn.Module:
    """Build the DeepSeek-V3 debugmodel -- Kimi-K2's architecture family.

    MLA attention + DeepSeekMoE: a dense first layer, then MoE layers with 3D
    grouped experts ``(num_experts, m, n)``, shared experts, and a router gate,
    plus tok_embeddings / final norm / lm_head. Exercises GroupedMuon (the 3D
    experts) alongside torch.optim.Muon (the 2D attention / FFN matrices).
    """
    config = _deepseekv3_configs["debugmodel"](
        attn_backend="flex", moe_comm_backend="standard"
    )
    return config.build()


def _make_per_matrix_moe_model() -> nn.Module:
    """Tiny MoE-ish model for balance='per-matrix'.

    Each layer has two 2D matrices (`wq`, `wo`), a 3D grouped-expert stack, and a 1D
    norm; the model has `tok_embeddings` and `output`. Exercises per-matrix Owned (2D)
    + Shard(0) experts (-> GroupedMuon) + Shard rest.
    """
    dim, ffn, num_experts, vocab, n_layers = 32, 64, 8, 128, 4

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(ffn, dim, bias=False)
            self.experts = nn.Parameter(torch.randn(num_experts, ffn, dim))
            self.attn_norm = nn.Parameter(torch.ones(dim))

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
            self.output = nn.Linear(dim, vocab, bias=False)

    return _Model()


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


def _ragged_local_shard(full, placement, info, rank, world_size):
    """This rank's local shard of a full tensor under any FlexShard placement.

    Mirrors ``RaggedShardMuon._local_update_shard``: for ``GroupedRaggedShard`` the
    byte-balanced cut crosses matrix boundaries, so slice the flat tensor at this rank's
    offset within the param; otherwise use the placement's own ``extract_local_shard``.
    """
    if isinstance(placement, GroupedRaggedShard):
        layout = info.bucket_layout.param_layouts[info.fqn]
        start = layout.local_global_offset - layout.param_offset
        return full.reshape(-1)[start : start + info.local_numel].view(info.local_shape)
    return placement.extract_local_shard(full, rank, world_size)


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

    def test_assign_matrix_owners_per_layer_balanced(self) -> None:
        world_size = 4
        # 8 layers, each the same heterogeneous matrix set (o_proj-like 59 dominates).
        layer = [59, 19, 11, 8, 4, 15, 15, 15]
        layers = [list(layer) for _ in range(8)]
        owners = assign_matrix_owners_per_layer_balanced(layers, world_size)

        self.assertEqual(len(owners), len(layers))
        largest = max(layer)
        global_load = [0] * world_size
        for layer_numels, layer_owners in zip(layers, owners, strict=True):
            self.assertEqual(len(layer_owners), len(layer_numels))
            load = [0] * world_size
            for numel, owner in zip(layer_numels, layer_owners, strict=True):
                self.assertIn(owner, range(world_size))
                load[owner] += numel
                global_load[owner] += numel
            # per-layer balance: within-layer spread is bounded by the largest matrix
            self.assertLessEqual(max(load) - min(load), largest)
        # rotation keeps running totals balanced across layers too
        self.assertLessEqual(max(global_load) - min(global_load), largest)

    def test_assign_matrix_owners_per_layer_balanced_rejects_bad_world_size(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "world_size"):
            assign_matrix_owners_per_layer_balanced([[1, 2]], 0)

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
        buckets = comm_free_muon_buckets(model, _FakeMesh(world_size))
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
        buckets = comm_free_muon_buckets(
            model, _FakeMesh(2), reshard_after_forward=False
        )
        for bucket in buckets:
            self.assertFalse(bucket.reshard_after_forward)

    def test_comm_free_muon_buckets_rejects_unknown_balance(self) -> None:
        _, model = make_transformer_model(n_layers=2)
        with self.assertRaisesRegex(ValueError, "balance"):
            comm_free_muon_buckets(model, _FakeMesh(2), balance="bogus")

    def test_comm_free_muon_buckets_per_matrix_structure(self) -> None:
        model = (
            _make_per_matrix_moe_model()
        )  # 4 layers; each: wq, wo (2D), experts(3D), norm(1D)
        buckets = comm_free_muon_buckets(
            model,
            _FakeMesh(2),
            balance="per-matrix",
            rest_patterns=["tok_embeddings.*", "output.*"],
        )

        def placement_name(bucket):
            placements = bucket.placement_fn(
                [(bucket.patterns[0], nn.Parameter(torch.zeros(1)))], None
            )
            return type(next(iter(placements.values()))[0]).__name__

        owned = [b for b in buckets if placement_name(b) == "Owned"]
        sharded = [b for b in buckets if placement_name(b) == "Shard"]
        # 4 layers x 2 matrices (wq, wo) -> one Owned bucket each
        self.assertEqual(len(owned), 4 * 2)
        self.assertTrue(all(len(b.patterns) == 1 for b in owned))
        self.assertTrue(
            all(b.patterns[0].endswith((".wq.weight", ".wo.weight")) for b in owned)
        )
        # 4 layers x (experts + norm) + 2 rest -> Shard buckets
        self.assertEqual(len(sharded), 4 * 2 + 2)

    def test_grouped_ragged_shard_muon_buckets_structure(self) -> None:
        model = (
            _make_per_matrix_moe_model()
        )  # 4 layers; each: wq, wo (2D), experts(3D), norm(1D)
        mesh = _FakeMesh(2)
        buckets = grouped_ragged_shard_muon_buckets(
            model,
            mesh,
            rest_patterns=["tok_embeddings.*", "output.*"],
        )

        def placement_of(bucket):
            placements = bucket.placement_fn(
                [(bucket.patterns[0], nn.Parameter(torch.zeros(2, 2)))], mesh
            )
            return next(iter(placements.values()))[0]

        grouped = [
            b for b in buckets if isinstance(placement_of(b), GroupedRaggedShard)
        ]
        sharded = [b for b in buckets if type(placement_of(b)).__name__ == "Shard"]
        # One GroupedRaggedShard bucket per layer, holding that layer's 2 matrices.
        self.assertEqual(len(grouped), 4)
        for b in grouped:
            self.assertEqual(len(b.patterns), 2)
            self.assertTrue(
                all(p.endswith((".wq.weight", ".wo.weight")) for p in b.patterns)
            )
        # 4 layers x (experts + norm) + 2 rest -> Shard buckets.
        self.assertEqual(len(sharded), 4 * 2 + 2)
        # Ragged Muon maps params by FQN, so reshard-after-forward must stay off.
        self.assertTrue(all(not b.reshard_after_forward for b in buckets))

    def test_reshard_policy_only_overrides_semantic_unshard(self) -> None:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        from torchtitan.experiments.flex_shard.flex_shard.ops import UNSHARD_BUCKET_OP
        from torchtitan.experiments.flex_shard.flex_shard.reshard_after_forward import (
            _compose_with_ac_policy,
            _ReshardAfterForwardRecomputeState,
        )

        def user_context_fn():
            def user_policy(ctx, func, *args, **kwargs):
                return CheckpointPolicy.PREFER_SAVE

            return create_selective_checkpoint_contexts(user_policy)

        bucket_id = 1
        recompute_state = _ReshardAfterForwardRecomputeState()
        merged_context_fn = _compose_with_ac_policy(
            user_context_fn,
            recompute_state,
            frozenset({bucket_id}),
        )
        forward_ctx, recompute_ctx = merged_context_fn()
        self.assertEqual(
            forward_ctx.policy_fn(None, UNSHARD_BUCKET_OP),
            CheckpointPolicy.MUST_RECOMPUTE,
        )
        for op in (
            torch.ops.aten.mm.default,
            torch.ops.aten.add.Tensor,
            torch.ops._c10d_functional.all_to_all_single.default,
        ):
            self.assertEqual(
                forward_ctx.policy_fn(None, op),
                CheckpointPolicy.PREFER_SAVE,
            )
        before_depth = len(_get_current_dispatch_mode_stack())
        with recompute_ctx:
            self.assertTrue(recompute_state.is_recomputing(bucket_id))
            self.assertEqual(
                len(_get_current_dispatch_mode_stack()),
                before_depth + 1,
            )
        self.assertEqual(len(_get_current_dispatch_mode_stack()), before_depth)

    def test_full_ac_recompute_context_marks_raf_recompute(self) -> None:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        from torchtitan.experiments.flex_shard.flex_shard.reshard_after_forward import (
            _make_full_ac_recompute_context_fn,
            _ReshardAfterForwardRecomputeState,
        )

        bucket_id = 123
        recompute_state = _ReshardAfterForwardRecomputeState()
        context_fn = _make_full_ac_recompute_context_fn(
            recompute_state,
            frozenset({bucket_id}),
        )
        forward_ctx, recompute_ctx = context_fn()

        with forward_ctx:
            self.assertFalse(recompute_state.is_recomputing(bucket_id))
        before_depth = len(_get_current_dispatch_mode_stack())
        with recompute_ctx:
            self.assertTrue(recompute_state.is_recomputing(bucket_id))
            self.assertEqual(len(_get_current_dispatch_mode_stack()), before_depth)
        self.assertFalse(recompute_state.is_recomputing(bucket_id))
        self.assertEqual(len(_get_current_dispatch_mode_stack()), before_depth)

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

    @unittest.skipIf(_deepseekv3_configs is None, "deepseek_v3 unavailable")
    def test_deepseekv3_kimi_experts_are_grouped_muon_eligible(self) -> None:
        # Kimi-K2 family (DeepSeek-V3): MoE experts are 3D stacks
        # (num_experts, m, n), so they are Muon-eligible via GroupedMuon (batched
        # Newton-Schulz), not torch.optim.Muon (which is 2D-only).
        model = _build_deepseekv3_kimi_model()
        expert_params = [
            (name, p)
            for name, p in model.named_parameters()
            if "experts.w" in name and "shared_experts" not in name
        ]
        self.assertTrue(expert_params)
        num_experts = {p.shape[0] for _, p in expert_params}
        self.assertEqual(len(num_experts), 1)  # dim 0 is the expert count
        self.assertGreater(next(iter(num_experts)), 1)
        for name, p in expert_params:
            self.assertEqual(p.ndim, 3, name)  # (num_experts, m, n)
            # the default predicate routes these stacks to GroupedMuon
            self.assertTrue(_default_muon_predicate(name, p.shape))


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
            buckets=comm_free_muon_buckets(model, mesh),
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
            buckets=comm_free_muon_buckets(
                model,
                mesh,
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
                buckets=comm_free_muon_buckets(
                    model,
                    mesh,
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
            buckets=comm_free_muon_buckets(model, mesh),
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

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(_deepseekv3_configs is None, "deepseek_v3 unavailable")
    def test_comm_free_muon_on_deepseekv3_moe(self) -> None:
        # Kimi-K2 architecture family: MLA attention + DeepSeekMoE with 3D grouped
        # experts. Verifies FlexShard accepts the model, the 3D experts route to
        # GroupedMuon (not AdamW), and the optimizer step is communication-free.
        mesh = self._mesh()
        model = _build_deepseekv3_kimi_model()

        # DSV3 stores `layers` as a ModuleDict, so build per-layer Owned buckets
        # from the layer FQNs (comm_free_muon_buckets assumes a ModuleList), plus
        # Shard buckets for embeddings / final norm / lm_head.
        layer_numel: dict[str, int] = {}
        for fqn, p in model.named_parameters():
            if fqn.startswith("layers."):
                lid = fqn.split(".")[1]
                layer_numel[lid] = layer_numel.get(lid, 0) + p.numel()
        layer_ids = sorted(layer_numel, key=int)
        owners = assign_layer_owners_lpt(
            [layer_numel[lid] for lid in layer_ids], self.world_size
        )
        buckets = [
            BucketSpec(
                [f"layers.{lid}.*"],
                placement_fn=make_owned_placement_fn(owners[i]),
                mesh=mesh,
                reshard_after_forward=False,
            )
            for i, lid in enumerate(layer_ids)
        ]
        buckets += [
            BucketSpec(
                [pattern],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=False,
            )
            for pattern in ("tok_embeddings.*", "norm.*", "lm_head.*")
        ]
        # FlexShard must accept the MLA + MoE (incl. 3D grouped experts) structure.
        flex_shard(model, buckets=buckets)

        # Router gate (2D, tiny) stays on AdamW, like Keller's recipe.
        def muon_pred(fqn, global_shape):
            return (
                global_shape is not None
                and len(global_shape) >= 2
                and "router.gate" not in fqn
            )

        muon_params, other_params = build_muon_param_groups(
            model, mesh, muon_param_predicate=muon_pred
        )
        name_by_id = {id(p): n for n, p in model.named_parameters()}
        grouped = [p for p in muon_params if p.ndim >= 3]

        # The 3D grouped experts are Muon-eligible (-> GroupedMuon), never AdamW.
        self.assertGreater(len(grouped), 0)
        self.assertTrue(any("experts.w" in name_by_id[id(p)] for p in grouped))
        self.assertTrue(all(p.ndim != 3 for p in other_params))

        optim = build_comm_free_muon_optimizers(
            model,
            mesh,
            muon_kwargs=dict(lr=0.02, momentum=0.9, weight_decay=0.0),
            adamw_kwargs=dict(lr=0.01, weight_decay=0.0),
            muon_param_predicate=muon_pred,
        )
        self.assertIn("GroupedMuon", {type(o).__name__ for o in optim.optimizers})

        # Dummy grads on each optimized param, then assert the step is collective-free.
        for p in muon_params + other_params:
            p.grad = torch.randn_like(p)
        with _assert_no_collectives(self):
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_expert_dim_sharding_routes_to_grouped_muon_comm_free(self) -> None:
        # Expert-parallel layout: shard the grouped-expert tensor along the expert
        # (leading) dim. Each rank holds whole experts (E/N, m, n), so the param is
        # routed to GroupedMuon and the step stays communication-free -- no Owned
        # broadcast of the whole stack, smaller balanced 1/N shards instead.
        mesh = self._mesh()
        num_experts, m, n = 8, 64, 48

        class _MoEish(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 3D grouped experts -> sharded along the expert dim (Shard(0))
                self.experts = nn.Parameter(torch.randn(num_experts, m, n))
                # a 2D attention-like matrix -> whole on one rank (Owned)
                self.proj = nn.Parameter(torch.randn(m, n))

        torch.manual_seed(0)
        model = _MoEish().to(device_type)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["experts"],
                    placement_fn=per_param_placements,  # Shard(0)
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
                BucketSpec(
                    ["proj"],
                    placement_fn=make_owned_placement_fn(0),  # Owned
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
            ],
        )

        experts = dict(model.named_parameters())["experts"]
        placement = get_placements(experts)[0]
        self.assertIsInstance(placement, Shard)
        self.assertEqual(placement.dim, 0)
        self.assertEqual(experts.dim(), 3)  # local shard keeps whole matrices
        self.assertEqual(
            experts.shape[0], num_experts // self.world_size
        )  # E/N experts

        muon_params, other_params = build_muon_param_groups(model, mesh)
        # The expert-dim-sharded 3D param is Muon-eligible (-> GroupedMuon), not AdamW.
        self.assertIn(id(experts), {id(p) for p in muon_params})
        self.assertNotIn(id(experts), {id(p) for p in other_params})

        optim = build_comm_free_muon_optimizers(
            model,
            mesh,
            muon_kwargs=dict(lr=0.02, momentum=0.9, weight_decay=0.0),
            adamw_kwargs=dict(lr=0.01, weight_decay=0.0),
        )
        self.assertIn("GroupedMuon", {type(o).__name__ for o in optim.optimizers})

        # Dummy grads on each optimized param, then assert the step is collective-free.
        for p in muon_params + other_params:
            p.grad = torch.randn_like(p)
        with _assert_no_collectives(self):
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_per_matrix_balance_owns_matrices_shards_experts_comm_free(self) -> None:
        # balance="per-matrix": each non-MoE 2D matrix -> its own Owned bucket
        # (balanced within the layer); the 3D experts -> Shard(0) -> GroupedMuon;
        # norms / embeddings / output -> Shard. Verify placements/routing + a
        # communication-free step.
        mesh = self._mesh()
        torch.manual_seed(0)
        model = _make_per_matrix_moe_model().to(device_type)
        n_layers = len(model.layers)

        flex_shard(
            model,
            buckets=comm_free_muon_buckets(
                model,
                mesh,
                balance="per-matrix",
                reshard_after_forward=False,
                rest_patterns=["tok_embeddings.*", "output.*"],
            ),
        )

        params = dict(model.named_parameters())
        for i in range(n_layers):
            wq = get_placements(params[f"layers.{i}.wq.weight"])[0]
            wo = get_placements(params[f"layers.{i}.wo.weight"])[0]
            self.assertIsInstance(wq, Owned)
            self.assertIsInstance(wo, Owned)
            # per-layer balance: the layer's 2 matrices land on different ranks (N=2)
            self.assertNotEqual(wq.owner_rank, wo.owner_rank)
            # experts and norm are Shard(0)
            self.assertIsInstance(
                get_placements(params[f"layers.{i}.experts"])[0], Shard
            )
            self.assertIsInstance(
                get_placements(params[f"layers.{i}.attn_norm"])[0], Shard
            )
        self.assertIsInstance(get_placements(params["tok_embeddings.weight"])[0], Shard)
        self.assertIsInstance(get_placements(params["output.weight"])[0], Shard)

        muon_params, other_params = build_muon_param_groups(model, mesh)
        # the Shard(0) 3D experts are routed to the Muon group (-> GroupedMuon)
        muon_ids = {id(p) for p in muon_params}
        experts_here = [
            p for n, p in model.named_parameters() if n.endswith(".experts")
        ]
        self.assertTrue(experts_here)
        for e in experts_here:
            self.assertIn(id(e), muon_ids)

        optim = build_comm_free_muon_optimizers(
            model,
            mesh,
            muon_kwargs=dict(lr=0.02, momentum=0.9, weight_decay=0.0),
            adamw_kwargs=dict(lr=0.01, weight_decay=0.0),
        )
        self.assertIn("GroupedMuon", {type(o).__name__ for o in optim.optimizers})

        for p in muon_params + other_params:
            p.grad = torch.randn_like(p)
        with _assert_no_collectives(self):
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_grouped_ragged_shard_muon_matches_single_device_muon(self) -> None:
        # Memory-balanced recipe: each layer's 2D matrices share one GroupedRaggedShard
        # bucket (byte-balanced, crossing matrix boundaries); the step all-gathers and
        # runs Newton-Schulz on the full matrix. Must match single-device Muon
        # bit-for-bit, with bucket bytes balanced 1/N (no whole-matrix hotspot), and --
        # unlike the Owned path -- the step does issue a collective (one all-gather/bucket).
        mesh = self._mesh()
        torch.manual_seed(0)
        model = _make_per_matrix_moe_model().to(device_type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        n_layers = len(model.layers)

        flex_shard(
            model,
            buckets=grouped_ragged_shard_muon_buckets(
                model,
                mesh,
                rest_patterns=["tok_embeddings.*", "output.*"],
            ),
        )

        muon_kwargs = dict(lr=0.02, momentum=0.9, weight_decay=0.0)
        adamw_kwargs = dict(lr=0.01, weight_decay=0.0)
        optim = build_ragged_shard_muon_optimizers(
            model, mesh, muon_kwargs=muon_kwargs, adamw_kwargs=adamw_kwargs
        )
        opt_types = {type(o).__name__ for o in optim.optimizers}
        self.assertIn("RaggedShardMuon", opt_types)  # ragged 2D matrices
        self.assertIn("GroupedMuon", opt_types)  # 3D experts (comm-free)

        info_by_fqn = {}
        for bs in model.sharded_bucket_storages:
            info_by_fqn.update(bs.param_infos)

        # Byte-perfect balance: every layer's matrix bucket is split evenly across ranks
        # (so wq + wo together, not each whole on one rank).
        for i in range(n_layers):
            rank_numels = info_by_fqn[f"layers.{i}.wq.weight"].bucket_layout.rank_numels
            self.assertEqual(max(rank_numels), min(rank_numels))

        # Reference: the same Muon / GroupedMuon / AdamW split on the full params.
        ref_by_fqn = dict(reference.named_parameters())
        ref_muon, ref_grouped, ref_other = [], [], []
        for fqn, flex_p in model.named_parameters():
            placement = get_placements(flex_p)[0]
            global_shape = get_global_shape(flex_p)
            ref_p = ref_by_fqn[fqn]
            if isinstance(placement, RaggedShard) and len(global_shape) == 2:
                ref_muon.append(ref_p)
            elif (
                isinstance(placement, Shard)
                and placement.dim == 0
                and len(global_shape) >= 3
            ):
                ref_grouped.append(ref_p)
            else:
                ref_other.append(ref_p)
        ref_optim = CombinedOptimizer(
            [
                torch.optim.Muon(ref_muon, **muon_kwargs),
                GroupedMuon(ref_grouped, **muon_kwargs),
                torch.optim.AdamW(ref_other, **adamw_kwargs),
            ]
        )

        # Identical full grads on every rank (== the AVG-reduced grad); the flex model
        # gets each param's local shard of that full grad.
        torch.manual_seed(1234)
        full_grads = {
            fqn: torch.randn_like(p) for fqn, p in reference.named_parameters()
        }
        for fqn, ref_p in reference.named_parameters():
            ref_p.grad = full_grads[fqn].clone()
        for fqn, flex_p in model.named_parameters():
            placement = get_placements(flex_p)[0]
            shard = _ragged_local_shard(
                full_grads[fqn], placement, info_by_fqn[fqn], self.rank, self.world_size
            )
            flex_p.grad = shard.clone().contiguous()

        # The step is NOT communication-free: it all-gathers each ragged bucket.
        real_all_gather = dist.all_gather
        gathers = 0

        def counting_all_gather(*a, **k):
            nonlocal gathers
            gathers += 1
            return real_all_gather(*a, **k)

        with mock.patch.object(dist, "all_gather", side_effect=counting_all_gather):
            optim.step()
        self.assertGreaterEqual(gathers, n_layers)
        ref_optim.step()

        # Bit-exact parity: every local shard equals the reference's matching shard.
        for fqn, flex_p in model.named_parameters():
            placement = get_placements(flex_p)[0]
            expected = _ragged_local_shard(
                ref_by_fqn[fqn].detach(),
                placement,
                info_by_fqn[fqn],
                self.rank,
                self.world_size,
            )
            self.assertEqual(flex_p.detach(), expected)


if __name__ == "__main__":
    run_tests()
