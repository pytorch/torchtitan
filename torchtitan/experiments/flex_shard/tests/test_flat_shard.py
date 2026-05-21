#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.flex_shard.example.flat_shard import (
    flat_shard_placements,
    FlatShard,
)
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    _assign_params_to_buckets,
    ParamInfo,
)
from torchtitan.experiments.flex_shard.flex_shard.utils import (
    _strip_checkpoint_wrapped_module_path,
    _validate_bucket_uniform_dtype_and_placement,
)
from torchtitan.experiments.flex_shard.tests.common import (
    make_transformer_model,
    single_rank_cpu_mesh,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


def _make_param_info(fqn: str, param: torch.Tensor, placement: FlatShard) -> ParamInfo:
    return ParamInfo(
        fqn=fqn,
        global_shape=param.shape,
        global_stride=param.stride(),
        dtype=param.dtype,
        requires_grad=True,
        placements=(placement,),
        local_shape=torch.Size([0]),
        local_numel=0,
        storage_nbytes=0,
        global_numel=param.numel(),
    )


def _expected_flat_shard(
    reference_params: list[tuple[str, torch.Tensor]],
    fqn: str,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    placements = flat_shard_placements(
        [(name, nn.Parameter(param.detach())) for name, param in reference_params],
        mesh=None,
    )
    placement = placements[fqn][0]
    assert isinstance(placement, FlatShard)
    param = dict(reference_params)[fqn]
    return placement.extract_local_shard(param.detach(), rank, world_size).contiguous()


def _average_reference_grads(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def _init_params_deterministically(model: nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _prefer_recompute_context_fn():
    def prefer_recompute_policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(prefer_recompute_policy)


def _checkpoint_transformer_execution_units(model: nn.Module) -> None:
    model.tok_embeddings = checkpoint_wrapper(
        model.tok_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    model.pos_embeddings = checkpoint_wrapper(
        model.pos_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    for idx, layer in enumerate(model.layers):
        model.layers[idx] = checkpoint_wrapper(
            layer,
            context_fn=_prefer_recompute_context_fn,
        )
    model.norm = checkpoint_wrapper(
        model.norm,
        context_fn=_prefer_recompute_context_fn,
    )
    model.output = checkpoint_wrapper(
        model.output,
        context_fn=_prefer_recompute_context_fn,
    )


def flat_transformer_bucket_specs(
    num_layers: int,
    *,
    reshard_after_forward: bool = False,
) -> list[BucketSpec]:
    return (
        [
            BucketSpec(
                ["tok_embeddings.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=reshard_after_forward,
            ),
            BucketSpec(
                ["pos_embeddings.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=reshard_after_forward,
            ),
        ]
        + [
            BucketSpec(
                [f"layers.{idx}.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=reshard_after_forward,
            )
            for idx in range(num_layers)
        ]
        + [
            BucketSpec(
                ["norm.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=reshard_after_forward,
            ),
            BucketSpec(
                ["output.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=reshard_after_forward,
            ),
        ]
    )


def _flat_shard_placement_map(
    named_params: list[tuple[str, nn.Parameter]],
    buckets: list[BucketSpec],
) -> dict[str, FlatShard]:
    assignments = _assign_params_to_buckets([fqn for fqn, _ in named_params], buckets)
    param_dict = dict(named_params)
    placement_map: dict[str, FlatShard] = {}
    for bucket_idx, fqns in enumerate(assignments):
        bucket_named_params = [(fqn, param_dict[fqn]) for fqn in fqns]
        bucket_placements = buckets[bucket_idx].placement_fn(
            bucket_named_params,
            mesh=None,
        )
        for fqn, placements in bucket_placements.items():
            placement = placements[0]
            assert isinstance(placement, FlatShard)
            placement_map[fqn] = placement
    return placement_map


def check_flat_shard_parity(
    cls,
    reference_module: nn.Module,
    flex_sharded_module: nn.Module,
    buckets: list[BucketSpec],
    rank: int,
    world_size: int,
) -> None:
    reference_named_params = list(reference_module.named_parameters())
    placement_map = _flat_shard_placement_map(reference_named_params, buckets)
    for (ref_name, ref_param), (name, param) in zip(
        reference_named_params,
        flex_sharded_module.named_parameters(),
        strict=True,
    ):
        cls.assertEqual(ref_name, name)
        cls.assertEqual(
            param.detach(),
            placement_map[name]
            .extract_local_shard(ref_param.detach(), rank, world_size)
            .contiguous(),
        )
        if ref_param.grad is None:
            cls.assertIsNone(param.grad)
            continue
        cls.assertIsNotNone(param.grad)
        cls.assertEqual(
            param.grad.detach(),
            placement_map[name]
            .extract_local_shard(ref_param.grad.detach(), rank, world_size)
            .contiguous(),
        )

    ref_state_dict = reference_module.state_dict()
    state_dict = flex_sharded_module.state_dict()
    cls.assertEqual(list(ref_state_dict), list(state_dict))
    state_dict_placement_map = {
        _strip_checkpoint_wrapped_module_path(fqn): placement
        for fqn, placement in placement_map.items()
    }
    for key, value in state_dict.items():
        placement = state_dict_placement_map.get(key)
        if placement is None:
            cls.assertEqual(value, ref_state_dict[key])
            continue
        cls.assertEqual(
            value,
            placement.extract_local_shard(
                ref_state_dict[key],
                rank,
                world_size,
            ).contiguous(),
        )


class TestFlatShardPlacement(TestCase):
    def test_flat_shard_placements_assign_flat_offsets(self):
        named_params = [
            ("a.weight", nn.Parameter(torch.empty(2, 3))),
            ("b.weight", nn.Parameter(torch.empty(4))),
        ]

        with single_rank_cpu_mesh() as mesh:
            placements = flat_shard_placements(named_params, mesh)

        a = placements["a.weight"][0]
        b = placements["b.weight"][0]
        self.assertIsInstance(a, FlatShard)
        self.assertIsInstance(b, FlatShard)
        assert isinstance(a, FlatShard)
        assert isinstance(b, FlatShard)
        self.assertEqual(a.total_numel, 10)
        self.assertEqual(a.param_start, 0)
        self.assertEqual(a.param_numel, 6)
        self.assertEqual(b.total_numel, 10)
        self.assertEqual(b.param_start, 6)
        self.assertEqual(b.param_numel, 4)
        self.assertEqual(a.bucket_key, ("a.weight", "b.weight"))
        self.assertEqual(b.bucket_key, ("a.weight", "b.weight"))

    def test_interval_math_covers_empty_partial_full_crossing_and_uneven(self):
        self.assertEqual(FlatShard(10, 8, 1)._local_param_bounds(0, 3), (0, 0))
        self.assertEqual(FlatShard(8, 2, 2)._local_param_bounds(0, 2), (0, 2))
        self.assertEqual(FlatShard(10, 2, 5)._local_param_bounds(0, 3), (0, 2))
        self.assertEqual(FlatShard(10, 3, 2)._local_param_bounds(0, 3), (0, 1))
        self.assertEqual(FlatShard(10, 3, 2)._local_param_bounds(1, 3), (1, 2))
        self.assertEqual(FlatShard(5, 2, 3)._local_param_bounds(1, 2), (1, 3))

    def test_local_shape_is_1d_and_extracts_flat_slices(self):
        placement = FlatShard(7, 2, 4)
        param = torch.arange(4, dtype=torch.float32).view(2, 2)

        self.assertEqual(
            placement.compute_local_shape(param.shape, 0, 3), torch.Size([1])
        )
        self.assertEqual(
            placement.compute_local_shape(param.shape, 1, 3), torch.Size([3])
        )
        self.assertEqual(
            placement.compute_local_shape(param.shape, 2, 3), torch.Size([0])
        )
        self.assertEqual(
            placement.extract_local_shard(param, 0, 3), torch.tensor([0.0])
        )
        self.assertEqual(
            placement.extract_local_shard(param, 1, 3),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        self.assertEqual(placement.extract_local_shard(param, 2, 3).numel(), 0)

    def test_compatible_flat_shards_pass_bucket_validation(self):
        named_params = [
            ("a.weight", nn.Parameter(torch.empty(2))),
            ("b.weight", nn.Parameter(torch.empty(3))),
        ]
        placements = {
            "a.weight": (FlatShard(5, 0, 2, bucket_key=("a.weight", "b.weight")),),
            "b.weight": (FlatShard(5, 2, 3, bucket_key=("a.weight", "b.weight")),),
        }

        _validate_bucket_uniform_dtype_and_placement(
            [["a.weight", "b.weight"]],
            placements,
            [
                BucketSpec(
                    ["*"],
                    placement_fn=flat_shard_placements,
                    reshard_after_forward=False,
                )
            ],
            named_params,
        )

    def test_incompatible_flat_shards_fail_bucket_validation(self):
        named_params = [
            ("a.weight", nn.Parameter(torch.empty(2))),
            ("b.weight", nn.Parameter(torch.empty(3))),
        ]
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=False,
            )
        ]

        with self.assertRaisesRegex(ValueError, "bucket-compatible placements"):
            _validate_bucket_uniform_dtype_and_placement(
                [["a.weight", "b.weight"]],
                {
                    "a.weight": (FlatShard(5, 0, 2, bucket_key=("a.weight",)),),
                    "b.weight": (FlatShard(5, 2, 3, bucket_key=("b.weight",)),),
                },
                buckets,
                named_params,
            )

        with self.assertRaisesRegex(ValueError, "bucket-compatible placements"):
            _validate_bucket_uniform_dtype_and_placement(
                [["a.weight", "b.weight"]],
                {
                    "a.weight": (FlatShard(5, 0, 2, bucket_key=("a.weight",)),),
                    "b.weight": (Shard(0),),
                },
                buckets,
                named_params,
            )

    def test_single_rank_materialization(self):
        params = [
            ("a.weight", torch.arange(6, dtype=torch.float32).view(2, 3)),
            ("b.weight", torch.arange(4, dtype=torch.float32).add(10)),
        ]
        named_params = [(fqn, nn.Parameter(param.clone())) for fqn, param in params]

        with single_rank_cpu_mesh() as mesh:
            placements = flat_shard_placements(named_params, mesh)
            infos = [
                _make_param_info(fqn, param, placements[fqn][0])
                for fqn, param in params
            ]
            local_tensors = [
                placements[fqn][0].extract_local_shard(param, 0, 1)
                for fqn, param in params
            ]
            placement = placements["a.weight"][0]
            prepared = placement.prepare_unshard_bucket(
                local_tensors,
                infos,
                mesh,
                None,
            )
            placement.run_prepared_unshard(prepared)
            result = placement.finish_prepared_unshard(prepared)

        for full_param, (_, expected) in zip(result.full_params, params, strict=True):
            self.assertEqual(full_param, expected)

    def test_prepare_unshard_reuses_adjacent_local_views_without_padding(self):
        flat_storage = torch.arange(6, dtype=torch.float32)
        params = [
            ("a.weight", flat_storage[:2].clone()),
            ("b.weight", flat_storage[2:].clone()),
        ]
        named_params = [(fqn, nn.Parameter(param.clone())) for fqn, param in params]

        with single_rank_cpu_mesh() as mesh:
            placements = flat_shard_placements(named_params, mesh)
            infos = [
                _make_param_info(fqn, param, placements[fqn][0])
                for fqn, param in params
            ]
            local_tensors = [flat_storage[:2], flat_storage[2:]]
            placement = placements["a.weight"][0]
            prepared = placement.prepare_unshard_bucket(
                local_tensors,
                infos,
                mesh,
                None,
            )

            self.assertEqual(
                prepared.buffers[0].untyped_storage().data_ptr(),
                flat_storage.untyped_storage().data_ptr(),
            )
            self.assertEqual(prepared.buffers[0].storage_offset(), 0)
            placement.run_prepared_unshard(prepared)
            result = placement.finish_prepared_unshard(prepared)

        for full_param, (_, expected) in zip(result.full_params, params, strict=True):
            self.assertEqual(full_param, expected)


class TestFlatShardCollectives(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    def test_two_rank_full_param_reconstruction_with_uneven_boundary(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        params = [
            ("a.weight", torch.arange(3, dtype=torch.float32)),
            ("b.weight", torch.arange(4, dtype=torch.float32).add(10)),
            ("c.weight", torch.arange(3, dtype=torch.float32).add(20)),
        ]
        named_params = [(fqn, nn.Parameter(param.clone())) for fqn, param in params]
        placements = flat_shard_placements(named_params, mesh)
        infos = [
            _make_param_info(fqn, param, placements[fqn][0]) for fqn, param in params
        ]
        local_tensors = [
            placements[fqn][0].extract_local_shard(
                param,
                self.rank,
                self.world_size,
            )
            for fqn, param in params
        ]

        placement = placements["a.weight"][0]
        prepared = placement.prepare_unshard_bucket(local_tensors, infos, mesh, None)
        placement.run_prepared_unshard(prepared)
        result = placement.finish_prepared_unshard(prepared)

        for full_param, (_, expected) in zip(result.full_params, params, strict=True):
            self.assertEqual(full_param, expected)

    def test_unshard_and_reduce_grad_zero_padding_tail(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        params = [
            ("a.weight", torch.arange(2, dtype=torch.float32)),
            ("b.weight", torch.arange(3, dtype=torch.float32).add(10)),
        ]
        named_params = [(fqn, nn.Parameter(param.clone())) for fqn, param in params]
        placements = flat_shard_placements(named_params, mesh)
        infos = [
            _make_param_info(fqn, param, placements[fqn][0]) for fqn, param in params
        ]
        local_tensors = [
            placements[fqn][0].extract_local_shard(
                param,
                self.rank,
                self.world_size,
            )
            for fqn, param in params
        ]

        placement = placements["a.weight"][0]
        prepared_unshard = placement.prepare_unshard_bucket(
            local_tensors,
            infos,
            mesh,
            None,
        )
        if self.rank == 1:
            self.assertEqual(prepared_unshard.buffers[0][-1], torch.tensor(0.0))

        prepared_reduce = placement.prepare_reduce_grad(
            [param.add(self.rank * 100) for _, param in params],
            infos,
            mesh,
            None,
        )
        self.assertEqual(prepared_reduce.buffers[0][-1], torch.tensor(0.0))

    def test_two_rank_reduce_grad_returns_averaged_flat_shards(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        grads = [
            ("a.weight", torch.arange(3, dtype=torch.float32).add(self.rank * 100)),
            (
                "b.weight",
                torch.arange(4, dtype=torch.float32).add(10 + self.rank * 100),
            ),
            (
                "c.weight",
                torch.arange(3, dtype=torch.float32).add(20 + self.rank * 100),
            ),
        ]
        named_params = [(fqn, nn.Parameter(grad.clone())) for fqn, grad in grads]
        placements = flat_shard_placements(named_params, mesh)
        infos = [_make_param_info(fqn, grad, placements[fqn][0]) for fqn, grad in grads]

        placement = placements["a.weight"][0]
        prepared = placement.prepare_reduce_grad(
            [grad for _, grad in grads],
            infos,
            mesh,
            None,
        )
        result = placement.reduce_prepared_grad(prepared)

        for sharded_grad, (fqn, local_grad) in zip(
            result.sharded_grads,
            grads,
            strict=True,
        ):
            expected_full_avg = local_grad.add(50 - self.rank * 100)
            expected = placements[fqn][0].extract_local_shard(
                expected_full_avg,
                self.rank,
                self.world_size,
            )
            self.assertEqual(sharded_grad, expected)


class _UnevenTwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(5, 3))
        self.w2 = nn.Parameter(torch.empty(2, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.w1.t()).matmul(self.w2.t())


class TestFlatShardTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        buckets = flat_transformer_bucket_specs(
            args.n_layers,
            reshard_after_forward=False,
        )
        flex_shard(
            model,
            mesh,
            buckets=buckets,
        )

        torch.manual_seed(42 + self.rank + 1)
        inputs = [
            transformer_inputs(args, batch_size=3, device=device_type),
            transformer_inputs(args, batch_size=2, device=device_type),
        ]
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        for x in inputs:
            loss = model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()

        _average_reference_grads(reference)
        check_flat_shard_parity(
            self,
            reference,
            model,
            buckets,
            self.rank,
            self.world_size,
        )

        optim.step()
        ref_optim.step()

        check_flat_shard_parity(
            self,
            reference,
            model,
            buckets,
            self.rank,
            self.world_size,
        )

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_policy(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model).to(torch.bfloat16)
        buckets = [
            BucketSpec(
                ["tok_embeddings.*"],
                placement_fn=flat_shard_placements,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                ),
                reshard_after_forward=False,
            ),
            *flat_transformer_bucket_specs(args.n_layers, reshard_after_forward=False)[
                1:
            ],
        ]

        flex_shard(
            model,
            mesh,
            buckets=buckets,
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=2, device=device_type)
        output = model.tok_embeddings(x)
        ref_output = reference.tok_embeddings(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output, ref_output)

        output.float().sum().backward()
        ref_output.float().sum().backward()

        grad = model.tok_embeddings._parameters["weight"].grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad.dtype, torch.float32)

        ref_grad = reference.tok_embeddings.weight.grad.to(torch.float32)
        dist.all_reduce(ref_grad, op=dist.ReduceOp.AVG)
        placement = _flat_shard_placement_map(
            list(reference.named_parameters()),
            buckets,
        )["tok_embeddings.weight"]
        expected_grad = placement.extract_local_shard(
            ref_grad,
            self.rank,
            self.world_size,
        ).contiguous()
        self.assertEqual(grad, expected_grad)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_with_activation_checkpointing(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        _checkpoint_transformer_execution_units(model)
        _checkpoint_transformer_execution_units(reference)
        buckets = flat_transformer_bucket_specs(
            args.n_layers,
            reshard_after_forward=True,
        )

        flex_shard(
            model,
            mesh,
            buckets=buckets,
        )

        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        composed_context_fn = model.layers[0].checkpoint_fn.keywords["context_fn"]
        self.assertIsNot(composed_context_fn, _prefer_recompute_context_fn)
        forward_ctx, _ = composed_context_fn()
        self.assertEqual(
            forward_ctx.policy_fn(
                None,
                torch.ops._c10d_functional.all_gather_into_tensor.default,
            ),
            CheckpointPolicy.MUST_RECOMPUTE,
        )
        self.assertEqual(
            forward_ctx.policy_fn(None, torch.ops.aten.mm.default),
            CheckpointPolicy.PREFER_RECOMPUTE,
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()
        ref_loss.backward()

        _average_reference_grads(reference)
        check_flat_shard_parity(
            self,
            reference,
            model,
            buckets,
            self.rank,
            self.world_size,
        )

        optim.step()
        ref_optim.step()
        check_flat_shard_parity(
            self,
            reference,
            model,
            buckets,
            self.rank,
            self.world_size,
        )

    @skip_if_lt_x_gpu(2)
    def test_backward_grad_and_optimizer_step_parity(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _UnevenTwoLinear().to(device_type)
        with torch.no_grad():
            for idx, param in enumerate(model.parameters()):
                param.copy_(
                    torch.arange(
                        param.numel(),
                        dtype=param.dtype,
                        device=param.device,
                    ).view_as(param)
                    + idx
                )
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=flat_shard_placements,
                    reshard_after_forward=False,
                )
            ],
        )

        x = torch.randn(4, 3, device=device_type)
        dist.broadcast(x, src=0)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.01)

        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()
        ref_loss.backward()
        _average_reference_grads(reference)

        reference_params = [
            (fqn, param.detach()) for fqn, param in reference.named_parameters()
        ]
        for fqn, param in model.named_parameters():
            self.assertEqual(
                param.detach(),
                _expected_flat_shard(reference_params, fqn, self.rank, self.world_size),
            )
            self.assertIsNotNone(param.grad)
            reference_grad = dict(reference.named_parameters())[fqn].grad
            assert reference_grad is not None
            self.assertEqual(
                param.grad.detach(),
                _expected_flat_shard(
                    [
                        (name, ref_param.grad.detach())
                        for name, ref_param in reference.named_parameters()
                        if ref_param.grad is not None
                    ],
                    fqn,
                    self.rank,
                    self.world_size,
                ),
            )

        optim.step()
        ref_optim.step()
        reference_params = [
            (fqn, param.detach()) for fqn, param in reference.named_parameters()
        ]
        for fqn, param in model.named_parameters():
            self.assertEqual(
                param.detach(),
                _expected_flat_shard(reference_params, fqn, self.rank, self.world_size),
            )

    @skip_if_lt_x_gpu(2)
    def test_whole_transformer_bucket_forward_backward(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        args, model = make_transformer_model(device=device_type.type)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=flat_shard_placements,
                    reshard_after_forward=False,
                )
            ],
        )

        x = transformer_inputs(args, device=device_type)
        dist.broadcast(x, src=0)
        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    @skip_if_lt_x_gpu(2)
    def test_per_layer_transformer_buckets_forward_backward(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        args, model = make_transformer_model(device=device_type.type)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        buckets = [
            BucketSpec(
                ["tok_embeddings.*", "pos_embeddings.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=False,
            ),
            *[
                BucketSpec(
                    [f"layers.{idx}.*"],
                    placement_fn=flat_shard_placements,
                    reshard_after_forward=False,
                )
                for idx in range(args.n_layers)
            ],
            BucketSpec(
                ["norm.*", "output.*"],
                placement_fn=flat_shard_placements,
                reshard_after_forward=False,
            ),
        ]
        flex_shard(model, mesh, buckets=buckets)

        loss = model(transformer_inputs(args, device=device_type)).sum()
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    run_tests()
