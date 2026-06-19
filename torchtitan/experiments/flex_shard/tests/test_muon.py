# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.experiments.flex_shard import flex_shard
from torchtitan.experiments.flex_shard.example.owned import Owned
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.flex_shard.sharded_param import (
    get_placements,
    set_sharding_info,
)
from torchtitan.experiments.flex_shard.muon import (
    _discover_expert_buckets,
    assign_layer_owners_lpt,
    build_comm_efficient_muon_optimizer,
    comm_efficient_muon_buckets,
    FlexShardMuonOptimizers,
    GatherGroupedMuon,
    GroupedMuon,
    is_owned_2d,
    layer_newton_schulz_cost,
    make_owned_placement_fn,
    newton_schulz_flops,
)
from torchtitan.experiments.flex_shard.tests.common import (
    expected_shard,
    make_transformer_model,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


class _StubMesh:
    """Minimal stand-in for a DeviceMesh in non-distributed bucket-builder tests."""

    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


def _reference_muon_param_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition a full (single-device) model: 2D layer weights vs the rest.

    Mirrors the distributed rule (only ``layers.<i>.*`` are ``Owned``, and 2D
    owned matrices go to Muon) so the reference optimizer matches.
    """
    muon, adamw = [], []
    for name, param in model.named_parameters():
        if name.startswith("layers.") and param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)
    return muon, adamw


# ---------------------------------------------------------------------------
# CPU unit tests: cost model + LPT assignment + bucket structure (no GPU/dist).
# ---------------------------------------------------------------------------
class TestCommEfficientMuonAssignment(TestCase):
    def test_newton_schulz_flops_uses_min_dim(self):
        # m * n * min(m, n); transpose-symmetric.
        self.assertEqual(newton_schulz_flops(torch.Size([4, 8])), 4 * 8 * 4)
        self.assertEqual(newton_schulz_flops(torch.Size([8, 4])), 8 * 4 * 4)
        self.assertEqual(newton_schulz_flops(torch.Size([16, 4])), 16 * 4 * 4)

    def test_layer_cost_sums_only_2d_matrices(self):
        named = [
            ("attn.wq.weight", nn.Parameter(torch.empty(8, 8))),
            ("attn.norm.weight", nn.Parameter(torch.empty(8))),  # 1D ignored
            ("mlp.w1.weight", nn.Parameter(torch.empty(16, 8))),
        ]
        expected = newton_schulz_flops(torch.Size([8, 8])) + newton_schulz_flops(
            torch.Size([16, 8])
        )
        self.assertEqual(layer_newton_schulz_cost(named), expected)

    def test_lpt_homogeneous_is_balanced(self):
        owners = assign_layer_owners_lpt([10, 10, 10, 10], world_size=2)
        # Equal layers -> each rank owns two, perfectly balanced.
        loads = [0, 0]
        for cost, owner in zip([10, 10, 10, 10], owners):
            loads[owner] += cost
        self.assertEqual(loads, [20, 20])

    def test_lpt_heterogeneous_makespan(self):
        costs = [10, 8, 6, 5, 4, 2]
        owners = assign_layer_owners_lpt(costs, world_size=3)
        loads = [0, 0, 0]
        for cost, owner in zip(costs, owners):
            loads[owner] += cost
        # Greedy LPT lands at {12, 12, 11}; max is within 4/3 of the 35/3 ideal.
        self.assertEqual(sorted(loads), [11, 12, 12])
        self.assertLessEqual(max(loads), (4 * sum(costs)) // (3 * 3) + 1)

    def test_lpt_is_deterministic(self):
        costs = [7, 3, 9, 1, 5, 5, 2]
        self.assertEqual(
            assign_layer_owners_lpt(costs, 4),
            assign_layer_owners_lpt(costs, 4),
        )

    def test_lpt_rejects_nonpositive_world_size(self):
        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            assign_layer_owners_lpt([1, 2], world_size=0)

    def test_make_owned_placement_fn_assigns_single_owner(self):
        placement_fn = make_owned_placement_fn(owner_rank=2)
        named = [("a.weight", nn.Parameter(torch.empty(2, 2)))]
        placements = placement_fn(named, _StubMesh(4))
        self.assertEqual(placements, {"a.weight": (Owned(2),)})

    def test_buckets_structure(self):
        args, model = make_transformer_model(n_layers=4)
        buckets = comm_efficient_muon_buckets(model, _StubMesh(2))

        layer_buckets = [b for b in buckets if b.patterns[0].startswith("layers.")]
        self.assertEqual(len(layer_buckets), args.n_layers)
        for bucket in layer_buckets:
            # Each layer bucket is a single-owner Owned bucket.
            placements = bucket.placement_fn(
                list(model.named_parameters()), _StubMesh(2)
            )
            owners = {p[0].owner_rank for p in placements.values()}
            self.assertEqual(len(owners), 1)
            self.assertIsInstance(next(iter(placements.values()))[0], Owned)

        # Non-layer params (tok/pos embeddings, norm, output) get their own buckets.
        rest_patterns = {
            b.patterns[0] for b in buckets if not b.patterns[0].startswith("layers.")
        }
        self.assertIn("output.*", rest_patterns)
        self.assertIn("norm.*", rest_patterns)

    def test_buckets_require_num_layers_ge_world_size(self):
        _, model = make_transformer_model(n_layers=2)
        with self.assertRaisesRegex(ValueError, "num_layers >= world_size"):
            comm_efficient_muon_buckets(model, _StubMesh(8))

    def test_param_routing_skips_empty_and_routes_by_placement(self):
        # Annotate params as FlexShard would, without needing CUDA/flex_shard.
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.empty(4, 4))  # owned 2D -> Muon
                self.norm = nn.Parameter(torch.empty(4))  # owned 1D -> AdamW
                self.other = nn.Parameter(torch.empty(0, 0))  # other rank's empty
                self.emb = nn.Parameter(torch.empty(8, 4))  # sharded 2D -> AdamW

        model = Tiny()
        set_sharding_info(model.w, (Owned(0),), torch.Size([4, 4]), (4, 1), None)
        set_sharding_info(model.norm, (Owned(0),), torch.Size([4]), (1,), None)
        set_sharding_info(model.other, (Owned(1),), torch.Size([4, 4]), (4, 1), None)
        set_sharding_info(model.emb, (Shard(0),), torch.Size([8, 4]), (4, 1), None)

        self.assertTrue(is_owned_2d(model.w))
        self.assertFalse(is_owned_2d(model.norm))  # 1D
        self.assertFalse(is_owned_2d(model.other))  # empty (numel == 0)
        self.assertFalse(is_owned_2d(model.emb))  # not Owned

        # Routing is inline in __init__ (no standalone _build_param_groups): build the
        # container and check which optimizer each param landed in.
        config = FlexShardMuonOptimizers.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern="x", optimizer_name="Muon", optimizer_kwargs={}
                ),
                ParamGroupConfig(
                    pattern="y", optimizer_name="AdamW", optimizer_kwargs={}
                ),
            ],
            implementation="for-loop",
        )
        container = FlexShardMuonOptimizers(config, model_parts=[model])
        muon_ids: set[int] = set()
        adamw_ids: set[int] = set()
        for opt in container.optimizers:
            ids = {id(p) for g in opt.param_groups for p in g["params"]}
            if isinstance(opt, torch.optim.AdamW):
                adamw_ids |= ids
            else:
                muon_ids |= ids
        self.assertEqual(muon_ids, {id(model.w)})
        self.assertEqual(adamw_ids, {id(model.norm), id(model.emb)})
        # The other rank's empty (0, 0) shard is skipped entirely.
        self.assertNotIn(id(model.other), muon_ids | adamw_ids)


# ---------------------------------------------------------------------------
# GPU parity test: communication-efficient Muon vs single-device Muon.
# ---------------------------------------------------------------------------
def _init_params_deterministically(model: nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(), dtype=param.dtype, device=param.device
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _average_reference_grads(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


@contextlib.contextmanager
def _assert_no_collectives():
    """Fail if any eager collective fires inside the block (comm-efficient step check)."""
    names = [
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter_tensor",
        "broadcast",
        "reduce",
    ]
    saved = {}

    def _raiser(name):
        def _fn(*args, **kwargs):
            raise AssertionError(f"unexpected collective '{name}' during step")

        return _fn

    for name in names:
        if hasattr(dist, name):
            saved[name] = getattr(dist, name)
            setattr(dist, name, _raiser(name))
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(dist, name, fn)


class TestCommEfficientMuon(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _check_param_parity(self, reference, model):
        """Assert each local param equals its placement-expected slice of reference."""
        for (ref_name, ref_param), (name, param) in zip(
            reference.named_parameters(),
            model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(ref_name, name)
            placements = get_placements(param)
            if placements is not None and isinstance(placements[0], Owned):
                owner = placements[0].owner_rank
                if self.rank == owner:
                    expected = ref_param.detach()
                else:
                    expected = ref_param.detach().new_empty([0] * ref_param.ndim)
            else:
                expected = expected_shard(
                    ref_param.detach(), rank=self.rank, world_size=self.world_size
                )
            self.assertEqual(param.detach(), expected, msg=f"param {name} mismatch")

    @skip_if_lt_x_gpu(2)
    def test_matches_single_device_muon(self):
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        args, model = make_transformer_model(device=device_type.type, n_layers=4)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=comm_efficient_muon_buckets(
                model, mesh, reshard_after_forward=False
            ),
        )

        muon_kwargs = {"lr": 0.02, "weight_decay": 0.0, "momentum": 0.9}
        adamw_kwargs = {"lr": 0.01, "weight_decay": 0.0}
        optim = build_comm_efficient_muon_optimizer(
            model, mesh, muon_kwargs=muon_kwargs, adamw_kwargs=adamw_kwargs
        )
        ref_muon_params, ref_adamw_params = _reference_muon_param_groups(reference)
        # Match the container's AdamW kernel (for-loop) so parity isn't masked by a
        # foreach-vs-for-loop fusion difference unrelated to comm-efficient Muon.
        ref_optims = [
            torch.optim.Muon(ref_muon_params, **muon_kwargs),
            torch.optim.AdamW(ref_adamw_params, **adamw_kwargs, foreach=False),
        ]

        # Distinct per-rank batches -> data-parallel; reference averages grads.
        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)

        for _ in range(3):
            optim.zero_grad(set_to_none=True)
            for ref_optim in ref_optims:
                ref_optim.zero_grad(set_to_none=True)

            loss = model(x).sum()
            ref_loss = reference(x).sum()
            loss.backward()
            ref_loss.backward()
            _average_reference_grads(reference)

            # The optimizer step must be communication-efficient.
            with _assert_no_collectives():
                optim.step()
            for ref_optim in ref_optims:
                ref_optim.step()

            self._check_param_parity(reference, model)

    @skip_if_lt_x_gpu(2)
    def test_owned_2d_go_to_muon_rest_to_adamw(self):
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        _, model = make_transformer_model(device=device_type.type, n_layers=4)
        flex_shard(
            model,
            buckets=comm_efficient_muon_buckets(
                model, mesh, reshard_after_forward=False
            ),
        )
        optim = build_comm_efficient_muon_optimizer(
            model, mesh, muon_kwargs={"lr": 0.01}, adamw_kwargs={"lr": 0.01}
        )

        muon_opt = next(
            (o for o in optim.optimizers if isinstance(o, torch.optim.Muon)), None
        )
        muon_params = set()
        if muon_opt is not None:
            for group in muon_opt.param_groups:
                muon_params.update(id(p) for p in group["params"])

        for name, param in model.named_parameters():
            if param.numel() == 0:
                continue
            placements = get_placements(param)
            owned_2d_here = (
                placements is not None
                and isinstance(placements[0], Owned)
                and placements[0].owner_rank == self.rank
                and param.ndim == 2
            )
            self.assertEqual(id(param) in muon_params, owned_2d_here, msg=name)


class TestCommEfficientMuonTraining(FSDPTest):
    """End-to-end: train torchtitan's real llama3 debug model with comm-efficient Muon."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_llama3_debugmodel_loss_decreases(self):
        from torchtitan.models.llama3 import llama3_configs

        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        # Same seed on every rank -> identical init and identical (overfit) batch,
        # so the data-parallel reduce is well-defined and the loss signal is clean.
        torch.manual_seed(0)
        model = llama3_configs["debugmodel"](attn_backend="flex").build()
        model = model.to(device_type)

        batch_size, seq_len, vocab_size = 4, 256, 2048
        positions = torch.arange(seq_len, device=device_type).repeat(batch_size, 1)
        # The masked attention backends (Flex/Varlen) need a torch whose
        # flex_attention API matches torchtitan; skip cleanly if this environment's
        # is incompatible (e.g. missing FA3, or an older create_block_mask) rather
        # than reporting a failure unrelated to Muon.
        try:
            attention_masks = model.get_attention_masks(positions)
        except (ImportError, TypeError) as exc:
            raise unittest.SkipTest(
                f"attention backend unavailable in this environment: {exc}"
            ) from exc

        # 6-layer dense model, world_size 2 -> num_layers >= world_size holds.
        flex_shard(
            model,
            buckets=comm_efficient_muon_buckets(
                model, mesh, reshard_after_forward=False
            ),
        )
        optim = build_comm_efficient_muon_optimizer(
            model,
            mesh,
            muon_kwargs={"lr": 0.02, "weight_decay": 0.0, "momentum": 0.9},
            adamw_kwargs={"lr": 0.01, "weight_decay": 0.0},
        )

        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device_type)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device_type)

        losses = []
        for _ in range(15):
            optim.zero_grad(set_to_none=True)
            logits = model(tokens, positions, attention_masks)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            optim.step()
            losses.append(loss.detach().item())

        self.assertTrue(all(torch.isfinite(torch.tensor(losses))), msg=str(losses))
        # Overfitting a fixed batch: the loss should drop clearly over the run.
        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        self.assertLess(late, 0.9 * early, msg=f"loss did not decrease: {losses}")


class TestDeepSeekV3MuonBuckets(TestCase):
    """CPU structural tests for the DeepSeek V3 FlexShard + Muon bucket builder."""

    def _build_debug_model(self) -> nn.Module:
        from torchtitan.models.deepseek_v3 import model_registry

        return model_registry("debugmodel").model.build()

    def _mp_policy(self):
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )

        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )

    def test_bucket_structure(self):
        from torchtitan.experiments.flex_shard.deepseek_v3.parallelize import (
            build_muon_buckets,
        )

        model = self._build_debug_model()
        num_layers = len(model.layers)
        mesh = _StubMesh(2)
        buckets = build_muon_buckets(
            model,
            dp_mesh=mesh,
            expert_mesh=mesh,
            mp_policy=self._mp_policy(),
            reshard_after_forward=True,
            reshard_last=True,
        )
        flat_patterns = [p for b in buckets for p in b.patterns]
        self.assertIn("tok_embeddings.*", flat_patterns)
        self.assertIn("norm.*", flat_patterns)
        self.assertIn("lm_head.*", flat_patterns)

        # Layer buckets carry explicit FQNs (not globs); split dense vs expert.
        params = dict(model.named_parameters())
        layer_buckets = [
            b for b in buckets if all(p.startswith("layers.") for p in b.patterns)
        ]
        expert_buckets = [
            b for b in layer_buckets if any(".moe.experts." in p for p in b.patterns)
        ]
        dense_buckets = [b for b in layer_buckets if b not in expert_buckets]

        # One Owned dense bucket per layer; each maps its FQNs to a single owner.
        self.assertEqual(len(dense_buckets), num_layers)
        for bucket in dense_buckets:
            named = [(p, params[p]) for p in bucket.patterns]
            placements = bucket.placement_fn(named, mesh)
            owners = {pl[0].owner_rank for pl in placements.values()}
            self.assertEqual(len(owners), 1)
            self.assertIsInstance(next(iter(placements.values()))[0], Owned)

        # MoE layers get a separate Shard expert bucket (layer 0 is dense -> none).
        self.assertEqual(len(expert_buckets), num_layers - 1)
        named = [(p, params[p]) for p in expert_buckets[0].patterns]
        placement = expert_buckets[0].placement_fn(named, mesh)
        self.assertIsInstance(next(iter(placement.values()))[0], Shard)

    def test_requires_num_layers_ge_dp_shard(self):
        from torchtitan.experiments.flex_shard.deepseek_v3.parallelize import (
            build_muon_buckets,
        )

        model = self._build_debug_model()
        big = _StubMesh(len(model.layers) + 1)
        with self.assertRaisesRegex(ValueError, "num_layers >= dp_shard"):
            build_muon_buckets(
                model,
                dp_mesh=big,
                expert_mesh=big,
                mp_policy=self._mp_policy(),
                reshard_after_forward=True,
                reshard_last=True,
            )

    def test_gather_muon_bucket_structure(self):
        # Gather-for-NS baseline: per-layer 2D matrices in one dense bucket, non-2D
        # params (norms) in their own buckets, embeddings/norm/lm_head as rest buckets.
        from torchtitan.experiments.flex_shard.deepseek_v3.parallelize import (
            build_gather_muon_buckets,
        )
        from torchtitan.experiments.flex_shard.example.shard import per_param_placements

        model = self._build_debug_model()
        buckets = build_gather_muon_buckets(
            model,
            dp_mesh=_StubMesh(2),
            mp_policy=self._mp_policy(),
            reshard_after_forward=False,
            dense_placement_fn=per_param_placements,
        )
        flat = [p for b in buckets for p in b.patterns]
        self.assertIn("tok_embeddings.*", flat)
        self.assertIn("lm_head.*", flat)
        self.assertIn("norm.*", flat)

        # The layer-0 dense bucket: a multi-pattern bucket of exact 2D-matrix FQNs.
        layer0_dense = [
            b
            for b in buckets
            if len(b.patterns) > 1
            and all(p.startswith("layers.0.") for p in b.patterns)
        ]
        self.assertEqual(len(layer0_dense), 1)
        self.assertIn("layers.0.attention.wq.weight", layer0_dense[0].patterns)
        # 1D norms are NOT in the dense (Muon) bucket.
        self.assertNotIn("layers.0.attention_norm.weight", layer0_dense[0].patterns)


class TestQwen3MuonBuckets(TestCase):
    """CPU structural tests for the dense Qwen3 FlexShard + Muon bucket builder.

    Qwen3 is dense (no MoE), so the shared builder should produce one Owned bucket
    per layer and *no* expert buckets -- the model-agnostic path exercised by the
    benchmark ladder.
    """

    def _build_debug_model(self) -> nn.Module:
        from torchtitan.models.qwen3 import model_registry

        spec = model_registry("debugmodel")
        # Untie lm_head from tok_embeddings (as the benchmark configs do) so the LM
        # head is its own param/bucket rather than an alias of the embedding.
        spec.model.enable_weight_tying = False
        return spec.model.build()

    def _mp_policy(self):
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )

        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )

    def test_dense_bucket_structure(self):
        from torchtitan.experiments.flex_shard.qwen3.parallelize import (
            build_muon_buckets,
        )

        model = self._build_debug_model()
        num_layers = len(model.layers)
        mesh = _StubMesh(2)
        buckets = build_muon_buckets(
            model,
            dp_mesh=mesh,
            expert_mesh=mesh,
            mp_policy=self._mp_policy(),
            reshard_after_forward=True,
            reshard_last=True,
        )
        flat = [p for b in buckets for p in b.patterns]
        self.assertIn("tok_embeddings.*", flat)
        self.assertIn("norm.*", flat)
        self.assertIn("lm_head.*", flat)

        params = dict(model.named_parameters())
        layer_buckets = [
            b for b in buckets if all(p.startswith("layers.") for p in b.patterns)
        ]
        # Dense model: one Owned bucket per layer, and zero expert buckets.
        self.assertEqual(len(layer_buckets), num_layers)
        self.assertFalse(
            any(".moe.experts." in p for b in layer_buckets for p in b.patterns)
        )
        for bucket in layer_buckets:
            named = [(p, params[p]) for p in bucket.patterns]
            placements = bucket.placement_fn(named, mesh)
            owners = {pl[0].owner_rank for pl in placements.values()}
            self.assertEqual(len(owners), 1)
            self.assertIsInstance(next(iter(placements.values()))[0], Owned)

        # The owner bucket holds all of a layer's dense params (2D matrices + 1D norms);
        # Muon-vs-AdamW routing happens later in the optimizer, not the bucket.
        layer0 = next(b for b in layer_buckets if b.patterns[0].startswith("layers.0."))
        self.assertIn("layers.0.attention.qkv_linear.wq.weight", layer0.patterns)
        self.assertIn("layers.0.attention_norm.weight", layer0.patterns)

    def test_permatrix_bucket_structure(self):
        # Per-2D-tensor allocation (case 2): one single-FQN Owned bucket per matrix,
        # owners from LPT over all matrices -> uses more distinct owners than layers.
        from torchtitan.experiments.flex_shard.muon.bucketing import (
            build_permatrix_muon_buckets,
        )

        model = self._build_debug_model()
        mesh = _StubMesh(2)
        buckets = build_permatrix_muon_buckets(
            model,
            dp_mesh=mesh,
            mp_policy=self._mp_policy(),
            reshard_after_forward=True,
            reshard_last=True,
        )
        params = dict(model.named_parameters())
        # Each 2D matrix is its own single-pattern Owned bucket.
        owned_2d = [
            b
            for b in buckets
            if len(b.patterns) == 1
            and b.patterns[0] in params
            and params[b.patterns[0]].ndim == 2
        ]
        num_2d = sum(
            1 for n, p in params.items() if n.startswith("layers.") and p.ndim == 2
        )
        self.assertEqual(len(owned_2d), num_2d)
        for bucket in owned_2d:
            placements = bucket.placement_fn(
                [(bucket.patterns[0], params[bucket.patterns[0]])], mesh
            )
            self.assertIsInstance(next(iter(placements.values()))[0], Owned)
        # rest patterns present
        flat = [p for b in buckets for p in b.patterns]
        self.assertIn("tok_embeddings.*", flat)
        self.assertIn("lm_head.*", flat)

    def test_auto_granularity_thresholds(self):
        # Unified selector: layer (W<=L) -> matrix (L<W<2L) -> twolevel (2L<=W<=L*M).
        from torchtitan.experiments.flex_shard.muon.bucketing import (
            resolve_auto_granularity,
        )

        model = self._build_debug_model()  # 8 layers, 7 matrices/layer = 56 matrices
        num_layers = len(model.layers)  # 8
        self.assertEqual(resolve_auto_granularity(model, 4), "layer")
        self.assertEqual(resolve_auto_granularity(model, num_layers), "layer")
        self.assertEqual(resolve_auto_granularity(model, 12), "matrix")  # 8 < 12 < 16
        self.assertEqual(resolve_auto_granularity(model, 16), "twolevel")  # >= 2*8
        self.assertEqual(resolve_auto_granularity(model, 56), "twolevel")  # == num_mats
        with self.assertRaises(NotImplementedError):
            resolve_auto_granularity(model, 64)  # > num_matrices -> needs L3

    def test_cost_weighted_group_sizes(self):
        # Heterogeneous two-level level-1 apportionment.
        from torchtitan.experiments.flex_shard.muon.bucketing import (
            _cost_weighted_group_sizes,
        )

        # Homogeneous -> ~uniform (== plain two-level).
        self.assertEqual(
            _cost_weighted_group_sizes([10, 10, 10, 10], [7, 7, 7, 7], 8),
            [2, 2, 2, 2],
        )
        # Heavy layer gets proportionally more ranks (cost 100 vs 10/10).
        self.assertEqual(
            _cost_weighted_group_sizes([100, 10, 10], [7, 7, 7], 8), [6, 1, 1]
        )
        # Cap at matrix count: a cost-dominant layer can't exceed its 2 matrices.
        self.assertEqual(_cost_weighted_group_sizes([1000, 10], [2, 7], 8), [2, 6])
        # world_size > num_matrices: only sum(caps) ranks placed (rest idle -> L3).
        self.assertEqual(_cost_weighted_group_sizes([10, 10], [2, 2], 8), [2, 2])


class TestQwen3MoeMuonBuckets(TestCase):
    """CPU structural test for the public Qwen3-MoE FlexShard + Muon surface.

    The Qwen3-MoE configs (``debugmodel_moe`` / ``30B-A3B``) place dense 2D matrices as
    ``Owned`` buckets on the dp mesh and the 3D MoE expert stacks as ``Shard`` buckets on
    the (EP) expert mesh. Each expert bucket is a 3D ``.moe.experts.`` stack -- exactly what
    the optimizer routing identifies (:func:`_is_expert_bucket`) and sends to full Muon
    (GroupedMuon / GatherGroupedMuon), not AdamW.
    """

    def test_moe_expert_buckets_land_on_expert_mesh(self):
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.flex_shard.qwen3.parallelize import (
            build_muon_buckets,
        )
        from torchtitan.models.qwen3 import model_registry

        model = model_registry("debugmodel_moe").model.build()
        params = dict(model.named_parameters())
        # Distinct dp vs expert mesh objects so we can assert experts land on the EP mesh.
        dp_mesh, expert_mesh = _StubMesh(2), _StubMesh(2)
        buckets = build_muon_buckets(
            model,
            dp_mesh=dp_mesh,
            expert_mesh=expert_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            reshard_after_forward=True,
            reshard_last=True,
        )
        layer_buckets = [
            b for b in buckets if all(p.startswith("layers.") for p in b.patterns)
        ]
        expert_buckets = [
            b for b in layer_buckets if any(".moe.experts." in p for p in b.patterns)
        ]
        dense_buckets = [b for b in layer_buckets if b not in expert_buckets]
        self.assertTrue(expert_buckets, "Qwen3-MoE should produce expert buckets")

        # Dense 2D matrices -> Owned bucket on the dp mesh.
        for bucket in dense_buckets:
            self.assertIs(bucket.mesh, dp_mesh)
            placements = bucket.placement_fn(
                [(p, params[p]) for p in bucket.patterns], dp_mesh
            )
            self.assertIsInstance(next(iter(placements.values()))[0], Owned)

        # Expert stacks -> Shard bucket on the EP (expert) mesh; each is a 3D
        # ``.moe.experts.`` stack, so the optimizer routes it to full Muon.
        for bucket in expert_buckets:
            self.assertIs(bucket.mesh, expert_mesh)
            for p in bucket.patterns:
                self.assertIn(".moe.experts.", p)
                self.assertEqual(params[p].ndim, 3)
            placements = bucket.placement_fn(
                [(p, params[p]) for p in bucket.patterns], expert_mesh
            )
            self.assertIsInstance(next(iter(placements.values()))[0], Shard)

    def test_expert_routing_requires_both_3d_and_marker(self):
        """A 3D *non-expert* stack must NOT route to grouped Muon (falls to AdamW).

        :func:`_is_expert_bucket` gates on two signals -- 3D ``global_shape`` AND the
        ``.moe.experts.`` marker -- so it is only true when both hold. The positive
        (marker + 3D) case is covered above; this pins the negative boundary: a 3D
        parameter without the marker, and a 2D parameter under the marker, are both
        rejected, so neither is misrouted to GroupedMuon / GatherGroupedMuon (they fall
        through to AdamW). Without the marker gate a future 3D non-expert param would
        silently take the expert path.
        """
        from types import SimpleNamespace

        from torchtitan.experiments.flex_shard.muon.containers import _is_expert_bucket

        def info(shape, fqn):
            return SimpleNamespace(global_shape=shape, fqn=fqn)

        # Both signals -> expert.
        self.assertTrue(_is_expert_bucket([info((4, 16, 8), "layers.0.moe.experts.w")]))
        # 3D but no marker (e.g. a hypothetical batched non-expert weight) -> NOT expert.
        self.assertFalse(_is_expert_bucket([info((4, 16, 8), "layers.0.attn.w_3d")]))
        # Marker but 2D (a per-matrix expert projection, not the stacked [E, m, n]) -> NOT.
        self.assertFalse(_is_expert_bucket([info((16, 8), "layers.0.moe.experts.w")]))
        # Mixed bucket (one 3D non-expert) -> NOT (requires all params to qualify).
        self.assertFalse(
            _is_expert_bucket(
                [
                    info((4, 16, 8), "layers.0.moe.experts.w"),
                    info((4, 16, 8), "layers.0.other.w_3d"),
                ]
            )
        )
        self.assertFalse(_is_expert_bucket([]))


class TestCommByteCounter(TestCase):
    """CPU tests for the benchmark comm-byte counter's tensor extraction."""

    def test_volume_bytes_picks_send_tensor(self):
        from torchtitan.experiments.flex_shard.muon import comm_counter

        send = torch.empty(4, 8, dtype=torch.float32)  # 32 elems * 4 B = 128
        # all_gather(output_list, input): the input tensor is args[1].
        self.assertEqual(comm_counter._volume_bytes("all_gather", ([], send), {}), 128)
        # reduce_scatter_tensor(output=, input=): keyword input.
        self.assertEqual(
            comm_counter._volume_bytes("reduce_scatter_tensor", (), {"input": send}),
            128,
        )
        # broadcast(tensor, src=): the tensor is args[0].
        self.assertEqual(
            comm_counter._volume_bytes("broadcast", (send,), {"src": 0}), 128
        )

    def test_functional_volume_bytes_picks_first_arg(self):
        from torchtitan.experiments.flex_shard.muon import comm_counter

        send = torch.empty(4, 8, dtype=torch.float32)  # 32 elems * 4 B = 128
        # _c10d_functional.all_gather_into_tensor(input, group_size, group_name): args[0].
        self.assertEqual(comm_counter._functional_volume_bytes((send, 2, "group")), 128)
        # Coalesced variants pass a list of input tensors -> summed.
        self.assertEqual(
            comm_counter._functional_volume_bytes(([send, send], 2, "group")), 256
        )
        self.assertEqual(comm_counter._functional_volume_bytes(()), 0)

    def test_reset_and_read(self):
        from torchtitan.experiments.flex_shard.muon import comm_counter

        comm_counter.reset()
        self.assertEqual(comm_counter.read(), 0)
        comm_counter._state["bytes"] += 100
        self.assertEqual(comm_counter.read(), 100)
        comm_counter.reset()
        self.assertEqual(comm_counter.read(), 0)


class TestCommByteCounterFunctional(FSDPTest):
    """The counter must count DTensor / functional-collective sends, not just eager ones.

    DTensor redistribute / ``full_tensor()`` (e.g. :class:`DTensorMuon`'s in-step
    all-gather) dispatch to ``torch.ops._c10d_functional.*`` ops, which do NOT route
    through the eager ``torch.distributed.*`` python symbols. If only the eager symbols
    were patched, the DTensor / FSDP2 baselines would register **zero** step comm and look
    as collective-free as comm-efficient ``Owned`` Muon -- the measurement bug this guards
    against. Verifies ``full_tensor()`` counts exactly this rank's send shard, and that the
    eager path still counts exactly once (disjoint namespaces -> no double-count).
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_functional_collectives_are_counted(self):
        from torch.distributed.tensor import distribute_tensor, Shard as DTShard

        from torchtitan.experiments.flex_shard.muon import comm_counter

        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("dp",)
        )
        comm_counter.install()

        m, n = 8, 16  # m divisible by world_size(2): even Shard(0) split.
        dt = distribute_tensor(
            torch.randn(m, n, device=device_type.type), mesh, [DTShard(0)]
        )
        local = dt.to_local()
        send_bytes = local.numel() * local.element_size()  # this rank's [m/ws, n] shard

        # full_tensor() all-gathers each rank's shard -> a functional collective. Counted
        # bytes must equal this rank's send shard (and be > 0: the bug would read 0).
        comm_counter.reset()
        _ = dt.full_tensor()
        torch.cuda.synchronize()
        self.assertGreater(comm_counter.read(), 0)
        self.assertEqual(comm_counter.read(), send_bytes)

        # The eager symbol still counts, exactly once (no double-count from the functional
        # patch: eager dispatches through c10d::, functional through _c10d_functional::).
        comm_counter.reset()
        eager_send = torch.randn(n, device=device_type.type)
        out = torch.empty(n * self.world_size, device=device_type.type)
        dist.all_gather_into_tensor(out, eager_send)
        torch.cuda.synchronize()
        self.assertEqual(
            comm_counter.read(), eager_send.numel() * eager_send.element_size()
        )


class TestGatherGroupedMuonShard1(FSDPTest):
    """3D MoE experts sharded *within* the matrix -- ``Shard(1)``, world_size > num_experts.

    When ``efsdp > num_local_experts`` each rank holds only a slice of every expert matrix,
    so :class:`GroupedMuon`'s local per-expert Newton-Schulz is invalid. ``GatherGroupedMuon``
    must all-gather each expert stack over the efsdp mesh, run per-expert (batched) NS on the
    reconstructed ``[E, m, n]``, and write back this rank's ``[E, m_local, n]`` slice. This
    checks it matches a single-device :class:`GroupedMuon` on the whole expert stack
    (bit-exact: same gathered NS input -> same per-expert update -> same local slice).
    """

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_matches_single_device_grouped_muon(self):
        from torchtitan.experiments.flex_shard import BucketSpec
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.flex_shard.muon.bucketing import expert_placement_fn

        # num_experts(2) < world_size(4) => expert_placement_fn picks Shard(1)
        # (each rank holds m/4 rows of both experts -- the incomplete-tensor regime).
        num_experts, m, n = 2, 16, 8
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("efsdp",)
        )

        # Realistic expert FQN "layers.0.moe.experts.weight" so the expert marker
        # (".moe.experts.") matches -- routing identifies experts by marker, not raw ndim.
        class _Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(num_experts, m, n, device=device_type.type)
                )

        class _MoE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = _Experts()

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.moe = _MoE()

        class _ExpertModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([_Layer()])

        torch.manual_seed(0)
        model = _ExpertModel()
        with torch.no_grad():
            model.layers[0].moe.experts.weight.normal_()
        reference = copy.deepcopy(model)  # full, unsharded experts
        ref_experts = reference.layers[0].moe.experts.weight

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*.moe.experts.*"],
                    placement_fn=expert_placement_fn,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32
                    ),
                    reshard_after_forward=False,
                )
            ],
        )

        buckets = _discover_expert_buckets(model)
        self.assertEqual(len(buckets), 1)
        placement = buckets[0]["placement"]
        expert_param = buckets[0]["params"][0]  # this rank's persistent shard

        # Confirm we actually hit the Shard(1) (incomplete-tensor) regime. (Read the
        # placement off the persistent param/bucket, not model.experts -- the latter is
        # the FlexShard unsharded-param property and raises outside the forward hook.)
        self.assertIsInstance(placement, Shard)
        self.assertGreaterEqual(placement.dim, 1)

        kwargs = {"lr": 0.02, "weight_decay": 0.0, "momentum": 0.9}
        optim = GatherGroupedMuon(buckets, mesh, **kwargs)
        ref_optim = GroupedMuon([ref_experts], **kwargs)

        for step in range(3):
            # Same deterministic full grad on every rank; the sharded model gets only its
            # local slice (what autograd would hand each rank), the reference the whole grad.
            torch.manual_seed(100 + step)
            full_grad = torch.randn(num_experts, m, n, device=device_type.type)
            ref_experts.grad = full_grad.clone()
            expert_param.grad = placement.extract_local_shard(
                full_grad, self.rank, self.world_size
            ).clone()

            optim.step()
            ref_optim.step()

            expected = placement.extract_local_shard(
                ref_experts.detach(), self.rank, self.world_size
            )
            torch.testing.assert_close(expert_param.detach(), expected, rtol=0, atol=0)


class TestGatherMuonVsFSDP2Muon(FSDPTest):
    """Gather Muon (FlexShard ``Shard``) and fsdp2 Muon (``DTensorMuon``) are identical.

    Both reconstruct the full 2D matrix (one all-gather per step) and run the *same*
    ``_zeropower_via_newtonschulz`` with the same momentum update, then write back only
    this rank's shard. They differ only in *how* the gather happens (FlexShard bucket
    unshard vs ``DTensor.full_tensor``), not in the math -- so given the same weight and
    gradient they must produce bit-identical updates. Shown by both matching a
    single-device ``torch.optim.Muon`` (hence each other), step after step.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_gather_and_dtensor_muon_match_single_device(self):
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import distribute_tensor

        from torchtitan.experiments.flex_shard import BucketSpec
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.flex_shard.muon import (
            _discover_dense_gather_buckets,
            DTensorMuon,
            GatherMuon,
        )
        from torchtitan.experiments.flex_shard.muon.bucketing import shard0_placement_fn

        m, n = 16, 8  # divisible by world_size(2): even row split, no padding
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )

        class _BodyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # named param "layers.0.weight" -> the gather-Muon "2D under layers." rule.
                self.layers = nn.ModuleList([nn.Linear(n, m, bias=False)])

        torch.manual_seed(0)
        reference = _BodyModel().to(device_type.type)  # full, single device
        model_gather = copy.deepcopy(reference)
        model_dtensor = copy.deepcopy(reference)

        # Gather Muon path: FlexShard Shard(0) on the 2D body matrix.
        flex_shard(
            model_gather,
            buckets=[
                BucketSpec(
                    ["layers.0.weight"],
                    placement_fn=shard0_placement_fn,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32
                    ),
                    reshard_after_forward=False,
                )
            ],
        )
        gbuckets = _discover_dense_gather_buckets(model_gather)
        self.assertEqual(len(gbuckets), 1)
        g_placement = gbuckets[0]["placement"]
        g_param = gbuckets[0]["params"][0]  # this rank's persistent shard
        self.assertIsInstance(g_placement, Shard)

        # fsdp2 Muon path: core fully_shard -> DTensor Shard(0).
        fully_shard(model_dtensor.layers[0], mesh=mesh)
        d_param = model_dtensor.layers[0].weight

        kwargs = {"lr": 0.02, "weight_decay": 0.1, "momentum": 0.95, "nesterov": True}
        gather_optim = GatherMuon(gbuckets, mesh, **kwargs)
        dtensor_optim = DTensorMuon([d_param], **kwargs)
        ref_optim = torch.optim.Muon([reference.layers[0].weight], **kwargs)

        for step in range(4):
            # Same deterministic full grad on every rank; each sharded model gets only its
            # local slice (what autograd would hand each rank), the reference the whole grad.
            torch.manual_seed(100 + step)
            full_grad = torch.randn(m, n, device=device_type.type)

            reference.layers[0].weight.grad = full_grad.clone()
            g_param.grad = g_placement.extract_local_shard(
                full_grad, self.rank, self.world_size
            ).clone()
            d_param.grad = distribute_tensor(full_grad, mesh, d_param.placements)

            ref_optim.step()
            gather_optim.step()
            dtensor_optim.step()

            ref_full = reference.layers[0].weight.detach()
            # Gather Muon: local shard equals the reference's same-placement slice.
            torch.testing.assert_close(
                g_param.detach(),
                g_placement.extract_local_shard(ref_full, self.rank, self.world_size),
                rtol=0,
                atol=0,
            )
            # fsdp2 Muon: local DTensor shard equals the reference's same-placement slice.
            torch.testing.assert_close(
                d_param.to_local(),
                distribute_tensor(ref_full, mesh, d_param.placements).to_local(),
                rtol=0,
                atol=0,
            )


class TestQKClipMaskAware(TestCase):
    """MuonClip captures S_max over the *attended* positions (real mask), not causal-only.

    A document/packed mask makes attention block cross-document pairs; clipping must be
    based on the logits the kernel actually attends to, or ``tau`` bounds the wrong score
    set. ``_max_logit_per_head`` materializes the FlexAttention ``BlockMask``'s own
    ``mask_mod`` rather than a hand-rolled causal mask.
    """

    @skip_if_lt_x_gpu(1)
    def test_document_mask_excludes_cross_document_logits(self):
        from torch.nn.attention.flex_attention import create_block_mask

        from torchtitan.experiments.flex_shard.muon import _max_logit_per_head

        torch.manual_seed(0)
        bsz, seqlen, n_heads, head_dim = 1, 8, 2, 4
        q = torch.randn(bsz, seqlen, n_heads, head_dim, device=device_type.type)
        k = torch.randn(bsz, seqlen, n_heads, head_dim, device=device_type.type)

        # Plant a huge logit between query pos 7 (doc 1) and key pos 0 (doc 0): causal
        # attends it (7 >= 0), a document mask must not.
        q[0, 7] = 10.0
        k[0, 0] = 10.0  # dot = 10 * 10 * head_dim = 400

        doc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device_type.type)

        def causal(b, h, qi, kv):
            return qi >= kv

        def doc_causal(b, h, qi, kv):
            return (qi >= kv) & (doc[qi] == doc[kv])

        bm_causal = create_block_mask(
            causal, bsz, 1, seqlen, seqlen, device=device_type.type
        )
        bm_doc = create_block_mask(
            doc_causal, bsz, 1, seqlen, seqlen, device=device_type.type
        )

        smax_causal = _max_logit_per_head(q, k, 1.0, bm_causal)
        smax_doc = _max_logit_per_head(q, k, 1.0, bm_doc)
        smax_fallback = _max_logit_per_head(q, k, 1.0, None)

        # Causal sees the planted cross-doc logit; the document mask excludes it.
        self.assertGreater(float(smax_causal.max()), 100.0)
        self.assertLess(float(smax_doc.max()), float(smax_causal.max()))
        # No-mask fallback (hand-rolled causal) matches the BlockMask causal exactly.
        torch.testing.assert_close(smax_fallback, smax_causal)


class TestQKClipDataParallel(FSDPTest):
    """MuonClip reduces S_max across data-parallel ranks before clipping (global batch).

    Each rank captures S_max from its own microbatch; the clip must use the global-batch
    per-head max (``all_reduce(MAX)``) or the owner would clip on only its rank's logits
    and sharded rows would scale from divergent local maxima.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_smax_reduced_to_global_batch_max(self):
        import math

        from torchtitan.experiments.flex_shard.muon import QKClip

        n_heads, nope, rope, v_head_dim, in_dim = 2, 4, 2, 4, 8
        qk_head_dim = nope + rope

        class _StubMLA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_heads = n_heads
                self.qk_nope_head_dim = nope
                self.qk_rope_head_dim = rope
                self.v_head_dim = v_head_dim
                self.q_lora_rank = 0
                self.softmax_scale = 1.0
                self.wq = nn.Linear(in_dim, n_heads * qk_head_dim, bias=False)
                self.wkv_b = nn.Linear(
                    in_dim, n_heads * (nope + v_head_dim), bias=False
                )
                self.inner_attention = nn.Identity()  # capture target (never called)

        torch.manual_seed(0)  # identical init on every rank
        attn = _StubMLA().to(device_type.type)
        wq_init = attn.wq.weight.detach().clone()

        tau = 100.0
        qkclip = QKClip([attn], tau=tau)  # shard_map=None -> Owned/plain row-scale path

        # Per-rank LOCAL maxima: each rank sees the large logit for a DIFFERENT head, so
        # the global per-head max is [200, 200] and BOTH heads must clip (gamma = 0.5).
        if self.rank == 0:
            attn._qkclip_smax = torch.tensor([200.0, 50.0], device=device_type.type)
        else:
            attn._qkclip_smax = torch.tensor([50.0, 200.0], device=device_type.type)

        qkclip.step()

        # Head 1's nope rows: rank 0's local S_max[1]=50 < tau would NOT clip on its own;
        # only the DP-reduced global max (200) does -> scaled by sqrt(tau / 200).
        h1_nope = slice(qk_head_dim, qk_head_dim + nope)
        torch.testing.assert_close(
            attn.wq.weight.detach()[h1_nope],
            wq_init[h1_nope] * math.sqrt(tau / 200.0),
        )

        # Both ranks must end bit-identical (clipped on the same global max).
        gathered = [torch.empty_like(attn.wq.weight) for _ in range(self.world_size)]
        dist.all_gather(gathered, attn.wq.weight.detach().contiguous())
        torch.testing.assert_close(gathered[0], gathered[1])


class TestQKClipShardedProjection(FSDPTest):
    """MuonClip scales a real FlexShard-sharded q/k projection, not just a full nn.Linear.

    For the gather methods the q/k projections are row-sharded (``Shard(0)``), and the
    container wires the buckets' ``(placement, info, mesh)`` into :class:`QKClip` via
    ``shard_map`` (see ``_setup_qkclip``). The sharded branch of ``_scale_rows`` slices the
    full per-head row-scale down to this rank's rows -- a path the plain data-parallel
    nn.Linear test never exercises (it has no shard_map). This checks the sharded local
    shard ends bit-exact with the same-placement slice of a single-device full QK-clip,
    so the row-count/slice arithmetic is correct rather than a silent no-op.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_sharded_projection_matches_full_clip_slice(self):
        from torchtitan.experiments.flex_shard import BucketSpec
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.flex_shard.muon import (
            _discover_dense_gather_buckets,
            QKClip,
        )
        from torchtitan.experiments.flex_shard.muon.bucketing import shard0_placement_fn

        n_heads, nope, rope, v_head_dim, in_dim = 2, 4, 2, 4, 8
        qk_head_dim = nope + rope

        class _StubMLA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_heads = n_heads
                self.qk_nope_head_dim = nope
                self.qk_rope_head_dim = rope
                self.v_head_dim = v_head_dim
                self.q_lora_rank = 0
                self.softmax_scale = 1.0
                # Rows = n_heads * head_dim, divisible by world_size(2) -> even Shard(0).
                self.wq = nn.Linear(in_dim, n_heads * qk_head_dim, bias=False)
                self.wkv_b = nn.Linear(
                    in_dim, n_heads * (nope + v_head_dim), bias=False
                )
                self.inner_attention = nn.Identity()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Under "layers." so the q/k projections form dense gather buckets.
                self.layers = nn.ModuleList([_StubMLA()])

        torch.manual_seed(0)  # identical init on every rank
        reference = _Model().to(device_type.type)  # full, single device
        model = copy.deepcopy(reference)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["layers.0.wq.weight", "layers.0.wkv_b.weight"],
                    placement_fn=shard0_placement_fn,
                    mesh=init_device_mesh(
                        device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
                    ),
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32
                    ),
                    reshard_after_forward=False,
                )
            ],
        )

        # Build the shard_map exactly as the container's _setup_qkclip does: each gathered
        # 2D param -> its (placement, info, mesh).
        buckets = _discover_dense_gather_buckets(model)
        shard_map: dict[int, tuple] = {}
        for bucket in buckets:
            placement, storage = bucket["placement"], bucket["storage"]
            for info, param in zip(bucket["infos"], bucket["params"], strict=True):
                shard_map[id(param)] = (placement, info, storage._mesh)
        self.assertEqual(len(shard_map), 2)  # wq + wkv_b both sharded

        sharded_attn = model.layers[0]
        ref_attn = reference.layers[0]
        tau = 100.0
        sharded_clip = QKClip([model], tau=tau, shard_map=shard_map)
        ref_clip = QKClip([reference], tau=tau)  # shard_map=None -> full row-scale

        # Head 0 over tau (clips, gamma=0.5), head 1 under (no clip). Identical on every
        # rank so the step()'s all_reduce(MAX) is a no-op and both ranks clip the same.
        smax = torch.tensor([200.0, 50.0], device=device_type.type)
        sharded_attn._qkclip_smax = smax.clone()
        ref_attn._qkclip_smax = smax.clone()

        sharded_clip.step()
        ref_clip.step()

        persistent = dict(model.named_parameters())
        for name in ("layers.0.wq.weight", "layers.0.wkv_b.weight"):
            placement, info, mesh = shard_map[id(persistent[name])]
            ref_full = dict(reference.named_parameters())[name].detach()
            expected = placement.extract_local_shard(
                ref_full, self.rank, self.world_size
            )
            torch.testing.assert_close(
                persistent[name].detach(), expected, rtol=0, atol=0
            )


class TestDTensorMuonExperts(FSDPTest):
    """fsdp2 DTensorMuon optimizes 3D MoE expert stacks with Muon (batched per-expert NS).

    The fsdp2 baseline routes experts to DTensorMuon (not AdamW), matching the Owned/gather
    full-Muon recipe. DTensorMuon all-gathers the ``[E, m, n]`` stack via ``full_tensor()``,
    runs one batched per-expert NS, and writes back this rank's expert shard -- bit-exact
    with a single-device :class:`GroupedMuon` over the whole stack.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_3d_experts_match_single_device_grouped(self):
        from torch.distributed.tensor import distribute_tensor, Shard as DTShard

        from torchtitan.experiments.flex_shard.muon import DTensorMuon, GroupedMuon

        num_experts, m, n = 4, 16, 8  # experts divisible by world_size(2)
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("dp",)
        )
        torch.manual_seed(0)
        full = torch.randn(num_experts, m, n, device=device_type.type)
        reference = nn.Parameter(full.clone())  # full stack, single device
        # Core fully_shard shards experts as Shard(0) DTensors over the dp mesh.
        dt = nn.Parameter(distribute_tensor(full.clone(), mesh, [DTShard(0)]))

        kw = {"lr": 0.02, "weight_decay": 0.1, "momentum": 0.95, "nesterov": True}
        d_opt = DTensorMuon([dt], **kw)
        ref_opt = GroupedMuon([reference], **kw)

        for step in range(3):
            torch.manual_seed(100 + step)
            grad = torch.randn(num_experts, m, n, device=device_type.type)
            reference.grad = grad.clone()
            dt.grad = distribute_tensor(grad.clone(), mesh, [DTShard(0)])

            ref_opt.step()
            d_opt.step()

            # This rank's expert shard must equal the reference's Shard(0) slice. Both run
            # the same batched NS on the same gathered [E, m, n] (full_tensor is a pure
            # gather, momentum is element-wise over disjoint experts), so require bit-wise
            # equality (rtol=0, atol=0), not just closeness.
            expected = distribute_tensor(
                reference.detach(), mesh, [DTShard(0)]
            ).to_local()
            torch.testing.assert_close(dt.to_local(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
