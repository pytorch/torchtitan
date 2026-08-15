# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.distributed.flex_shard import (
    _optimizer_reshard_runtime,
    BlockShard,
    BucketConfig,
    build_flex_shard_muon,
    ComputeLayout,
    flex_optimizer_reshard,
    NoRedistribution,
    Owned,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _BucketOverlapPlan,
    _compose_bucket_overlap_plans,
    _LocalBucketPlan,
    _RedistributionBucketPlan,
    _validate_bucket_plans_across_ranks,
)
from torchtitan.distributed.flex_shard.muon import _get_muon_reshard_integration


def _adjust_muon_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    matrix_shape: torch.Size | tuple[int, ...],
) -> float:
    rows, columns = matrix_shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = max(1.0, rows / columns) ** 0.5
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * max(rows, columns) ** 0.5
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = (rows / columns) ** 0.5
    else:
        raise ValueError(f"unsupported adjust_lr_fn {adjust_lr_fn!r}")
    return lr * ratio


class TestBucketConfig(unittest.TestCase):
    def test_validates_redistribution_mesh_axis_names(self):
        for invalid_redistribution in (
            "dp_shard",
            ["dp_shard"],
            {"dp_shard"},
            None,
        ):
            with (
                self.subTest(redistribution=invalid_redistribution),
                self.assertRaisesRegex(ValueError, "tuple or NoRedistribution"),
            ):
                BucketConfig(
                    patterns=("*",),
                    redistribution_mesh_axis_names=cast(
                        Any,
                        invalid_redistribution,
                    ),
                )
        with self.assertRaisesRegex(ValueError, "use NoRedistribution"):
            BucketConfig(
                patterns=("*",),
                redistribution_mesh_axis_names=(),
            )
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            BucketConfig(
                patterns=("*",),
                redistribution_mesh_axis_names=("",),
            )
        with self.assertRaisesRegex(ValueError, "duplicate axes"):
            BucketConfig(
                patterns=("*",),
                redistribution_mesh_axis_names=("dp_shard", "dp_shard"),
            )
        no_redistribution = NoRedistribution()
        config = BucketConfig(
            patterns=("*",),
            redistribution_mesh_axis_names=no_redistribution,
        )
        self.assertIs(config.redistribution_mesh_axis_names, no_redistribution)


class TestBucketOverlapPlan(unittest.TestCase):
    @staticmethod
    def _redistribution_bucket(name: str) -> _RedistributionBucketPlan[str]:
        return _RedistributionBucketPlan(
            redistributed_items=(name,),
            redistribution_plans=(),
            group=cast(Any, SimpleNamespace(process_group=None)),
            storage_to_compute_schedule=cast(Any, None),
            compute_to_storage_schedule=cast(Any, None),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_composes_adjacent_local_buckets_without_merging_them(self):
        leading_local = _LocalBucketPlan(("leading",))
        first_redistribution = self._redistribution_bucket("first")
        first_local = _LocalBucketPlan(("first_local",))
        second_local = _LocalBucketPlan(("second_local",))
        second_redistribution = self._redistribution_bucket("second")

        plans = _compose_bucket_overlap_plans(
            (
                leading_local,
                first_redistribution,
                first_local,
                second_local,
                second_redistribution,
            )
        )

        self.assertEqual(len(plans), 2)
        overlap = plans[0]
        self.assertIsInstance(overlap, _BucketOverlapPlan)
        self.assertIs(overlap.redistribution_bucket, first_redistribution)
        self.assertEqual(
            overlap.local_buckets,
            (leading_local, first_local, second_local),
        )
        self.assertIs(plans[1], second_redistribution)

    def test_preserves_local_buckets_without_redistribution(self):
        first = _LocalBucketPlan(("first",))
        second = _LocalBucketPlan(("second",))

        self.assertEqual(
            _compose_bucket_overlap_plans((first, second)),
            (first, second),
        )

    def test_rank_validation_ignores_overlapped_local_bucket_membership(self):
        redistribution_bucket = self._redistribution_bucket("redistributed")
        first = _BucketOverlapPlan(
            redistribution_bucket,
            (_LocalBucketPlan(("first_local",)),),
        )
        second = _BucketOverlapPlan(
            redistribution_bucket,
            (_LocalBucketPlan(("second_local",)),),
        )
        captured_hash: torch.Tensor | None = None

        def capture_hash(gathered, local_hash, *, group):
            nonlocal captured_hash
            captured_hash = local_hash.clone()
            gathered[0].copy_(local_hash)

        with (
            patch(
                "torchtitan.distributed.flex_shard."
                "_optimizer_reshard_schedule.dist.get_world_size",
                return_value=1,
            ),
            patch(
                "torchtitan.distributed.flex_shard."
                "_optimizer_reshard_schedule.dist.all_gather",
                side_effect=capture_hash,
            ),
        ):
            _validate_bucket_plans_across_ranks(
                (first,),
                item_signature=lambda item: (item,),
            )

        self.assertIsNotNone(captured_hash)

        def replay_hash(gathered, local_hash, *, group):
            assert captured_hash is not None
            gathered[0].copy_(captured_hash)

        with (
            patch(
                "torchtitan.distributed.flex_shard."
                "_optimizer_reshard_schedule.dist.get_world_size",
                return_value=1,
            ),
            patch(
                "torchtitan.distributed.flex_shard."
                "_optimizer_reshard_schedule.dist.all_gather",
                side_effect=replay_hash,
            ),
        ):
            _validate_bucket_plans_across_ranks(
                (second,),
                item_signature=lambda item: (item,),
            )


class TestFlexOptimizerReshard(unittest.TestCase):
    def test_configures_supported_optimizer_in_place(self):
        optimizer = torch.optim.Muon(
            [
                {
                    "params": [torch.nn.Parameter(torch.ones(2, 2))],
                    "param_names": ["weight"],
                }
            ]
        )
        compute_sharding_by_fqn: dict[str, object] = {}
        bucket_configs: tuple[BucketConfig, ...] = ()

        with patch(
            "torchtitan.distributed.flex_shard.muon._configure_muon_reshard",
            autospec=True,
        ) as configure:
            configured = flex_optimizer_reshard(
                optimizer,
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                bucket_configs=bucket_configs,
            )

        self.assertIs(configured, optimizer)
        self.assertIs(type(configured), torch.optim.Muon)
        configure.assert_called_once_with(
            optimizer,
            compute_sharding_by_fqn=compute_sharding_by_fqn,
            bucket_configs=bucket_configs,
        )

    def test_rejects_unsupported_optimizer(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2, 2))], lr=0.1)

        with self.assertRaisesRegex(
            TypeError,
            "does not support optimizer type 'SGD'",
        ):
            flex_optimizer_reshard(
                optimizer,
                compute_sharding_by_fqn={},
                bucket_configs=(),
            )

    def test_missing_upstream_api_fails_before_configuration(self):
        optimizer = torch.optim.Muon([torch.nn.Parameter(torch.ones(2, 2))])

        with (
            patch.object(
                optimizer,
                "register_step_executor",
                None,
                create=True,
            ),
            self.assertRaisesRegex(RuntimeError, "Muon.register_step_executor"),
        ):
            flex_optimizer_reshard(
                optimizer,
                compute_sharding_by_fqn={},
                bucket_configs=(),
            )

    def test_failed_configuration_leaves_stock_muon_usable(self):
        fqn = "weight"
        optimizer = torch.optim.Muon(
            [
                {
                    "params": [torch.nn.Parameter(torch.ones(2, 2))],
                    "param_names": [fqn],
                }
            ]
        )

        with (
            patch.object(
                optimizer,
                "register_step_executor",
                create=True,
            ),
            patch.object(torch.optim, "muon_prepare", create=True),
            patch.object(torch.optim, "muon_orthogonalize", create=True),
            patch.object(torch.optim, "muon_apply", create=True),
            self.assertRaisesRegex(TypeError, "requires DTensor parameters"),
        ):
            flex_optimizer_reshard(
                optimizer,
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Replicate()},
                    )
                },
                bucket_configs=(
                    BucketConfig(
                        patterns=(fqn,),
                        redistribution_mesh_axis_names=NoRedistribution(),
                    ),
                ),
            )

        closure_called = False

        def closure() -> float:
            nonlocal closure_called
            closure_called = True
            return 0.0

        self.assertEqual(optimizer.step(closure), 0.0)
        self.assertTrue(closure_called)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestFlexShardMuon(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_explicit_replicated_compute(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(12, device=device).reshape(4, 3).float().div_(10)
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, (Replicate(),))
        )
        fqn = "layers.0.replicated"
        compute_sharding_by_fqn = {
            fqn: ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": Replicate(),
                },
            )
        }
        bucket_configs = (
            BucketConfig(
                patterns=(fqn,),
                redistribution_mesh_axis_names=NoRedistribution(),
            ),
        )
        unconfigured_optimizer = torch.optim.Muon(
            [{"params": [parameter], "param_names": [fqn]}],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        original_type = type(unconfigured_optimizer)
        original_step = unconfigured_optimizer.step.__func__
        hook_events = []
        unconfigured_optimizer.register_step_pre_hook(
            lambda *_args: hook_events.append("pre")
        )
        unconfigured_optimizer.register_step_post_hook(
            lambda *_args: hook_events.append("post")
        )
        with patch.object(torch.optim.Optimizer, "_patch_step_function") as patch_step:
            optimizer = flex_optimizer_reshard(
                unconfigured_optimizer,
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                bucket_configs=bucket_configs,
            )
        self.assertIs(optimizer, unconfigured_optimizer)
        self.assertIs(type(optimizer), original_type)
        self.assertIs(optimizer.step.__func__, original_step)
        patch_step.assert_not_called()

        with self.assertRaisesRegex(ValueError, "more than once"):
            flex_optimizer_reshard(
                optimizer,
                compute_sharding_by_fqn={},
                bucket_configs=(),
            )

        original_param_names = optimizer.param_groups[0]["param_names"]
        optimizer.param_groups[0]["param_names"] = ["renamed"]
        with self.assertRaisesRegex(RuntimeError, "parameter groups changed"):
            optimizer.step()
        matching_mutated_state = optimizer.state_dict()
        with self.assertRaisesRegex(ValueError, "current Muon"):
            optimizer.load_state_dict(matching_mutated_state)
        optimizer.param_groups[0]["param_names"] = original_param_names
        hook_events.clear()

        mismatched_state_dict = optimizer.state_dict()
        mismatched_state_dict["param_groups"][0]["param_names"] = ["renamed"]
        with self.assertRaisesRegex(ValueError, "configured FlexShard FQNs"):
            optimizer.load_state_dict(mismatched_state_dict)

        reference = torch.nn.Parameter(value.clone())
        reference_optimizer = torch.optim.Muon(
            [reference],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        grad = torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        parameter.grad = distribute_tensor(grad.clone(), mesh, (Replicate(),))
        reference.grad = grad.clone()

        optimizer.step()
        reference_optimizer.step()
        self.assertEqual(hook_events, ["pre", "post"])

        torch.testing.assert_close(
            parameter.to_local(),
            reference,
            rtol=0,
            atol=0,
        )

    @with_comms
    def test_dispatches_explicit_axis_alternatives(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(12, device=device).reshape(4, 3).float()
        fqn = "layers.0.weight"
        bucket_configs = (
            BucketConfig(
                patterns=(fqn,),
                redistribution_mesh_axis_names=("dp_shard",),
            ),
            BucketConfig(
                patterns=(fqn,),
                redistribution_mesh_axis_names=NoRedistribution(),
            ),
        )

        local_parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, (Replicate(),))
        )
        local_optimizer = build_flex_shard_muon(
            [{"params": [local_parameter], "param_names": [fqn]}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": Replicate()},
                )
            },
            bucket_configs=bucket_configs,
        )
        local_integration = _get_muon_reshard_integration(local_optimizer)
        self.assertEqual(len(local_integration._specs), 1)
        local_spec = local_integration._specs[0]
        self.assertEqual(
            (
                local_spec.redistribution_mesh_axis_names,
                local_spec.fqns,
            ),
            (NoRedistribution(), (fqn,)),
        )
        self.assertEqual(
            local_spec.diagnostic_label,
            "BucketConfig(redistribution_mesh_axis_names=NoRedistribution(), "
            f"fqns=({fqn!r},))",
        )

        redistributed_parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, (Shard(0),))
        )
        redistributed_optimizer = build_flex_shard_muon(
            [{"params": [redistributed_parameter], "param_names": [fqn]}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": Owned()},
                )
            },
            bucket_configs=bucket_configs,
        )
        redistributed_integration = _get_muon_reshard_integration(
            redistributed_optimizer
        )
        self.assertEqual(
            tuple(
                spec.redistribution_mesh_axis_names
                for spec in redistributed_integration._specs
            ),
            (("dp_shard",),),
        )

        with self.assertRaisesRegex(ValueError, "matching BucketConfigs declare"):
            build_flex_shard_muon(
                [{"params": [local_parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Replicate()},
                    )
                },
                bucket_configs=(bucket_configs[0],),
            )

        with self.assertRaisesRegex(ValueError, "matches multiple BucketConfigs"):
            build_flex_shard_muon(
                [{"params": [local_parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Replicate()},
                    )
                },
                bucket_configs=(bucket_configs[1], bucket_configs[1]),
            )

    @with_comms
    def test_rejects_incompatible_routes_in_one_physical_bucket(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        values = (
            torch.arange(12, device=device).reshape(4, 3).float(),
            torch.arange(12, 24, device=device).reshape(4, 3).float(),
        )
        fqns = ("layers.0.first", "layers.0.second")
        parameters = tuple(
            torch.nn.Parameter(distribute_tensor(value, mesh, (Shard(0),)))
            for value in values
        )
        replicated_compute = ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": Replicate()},
        )

        with self.assertRaisesRegex(ValueError, "incompatible routes"):
            build_flex_shard_muon(
                [{"params": parameters, "param_names": fqns}],
                compute_sharding_by_fqn={fqn: replicated_compute for fqn in fqns},
                bucket_configs=(
                    BucketConfig(
                        patterns=fqns,
                        redistribution_mesh_axis_names=("dp_shard",),
                    ),
                ),
            )

    @with_comms
    def test_validates_physical_bucket_participant_by_shard_index(self):
        standard_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        permuted_mesh = DeviceMesh(
            self.device_type,
            torch.tensor((1, 0)),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(12, device=device).reshape(4, 3).float()
        first_fqn = "layers.0.first"
        second_fqn = "layers.0.second"
        first = torch.nn.Parameter(
            distribute_tensor(value.clone(), standard_mesh, (Shard(0),))
        )
        second = torch.nn.Parameter(
            distribute_tensor(value.clone(), permuted_mesh, (Shard(0),))
        )
        owned_compute = ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": Owned()},
        )

        permuted_optimizer = build_flex_shard_muon(
            [{"params": (second,), "param_names": (second_fqn,)}],
            compute_sharding_by_fqn={second_fqn: owned_compute},
            bucket_configs=(
                BucketConfig(
                    patterns=(second_fqn,),
                    redistribution_mesh_axis_names=("dp_shard",),
                ),
            ),
        )
        permuted_integration = _get_muon_reshard_integration(permuted_optimizer)
        permuted_plan = permuted_integration._bucket_plans[0]
        self.assertIsInstance(permuted_plan, _RedistributionBucketPlan)
        permuted_plan = cast(_RedistributionBucketPlan, permuted_plan)
        self.assertEqual(permuted_plan.group.participant_by_shard_index, (1, 0))

        with self.assertRaisesRegex(ValueError, "participant-by-shard-index"):
            build_flex_shard_muon(
                [{"params": (first, second), "param_names": (first_fqn, second_fqn)}],
                compute_sharding_by_fqn={
                    first_fqn: owned_compute,
                    second_fqn: owned_compute,
                },
                bucket_configs=(
                    BucketConfig(
                        patterns=(first_fqn, second_fqn),
                        redistribution_mesh_axis_names=("dp_shard",),
                    ),
                ),
            )

    @with_comms
    def test_rejects_mixed_dtype_physical_bucket(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        fqns = ("layers.0.float", "layers.0.bfloat")
        parameters = (
            torch.nn.Parameter(
                distribute_tensor(
                    torch.arange(12, device=device).reshape(4, 3).float(),
                    mesh,
                    (Shard(0),),
                )
            ),
            torch.nn.Parameter(
                distribute_tensor(
                    torch.arange(12, device=device).reshape(4, 3).bfloat16(),
                    mesh,
                    (Shard(0),),
                )
            ),
        )
        owned_compute = ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": Owned()},
        )

        with self.assertRaisesRegex(ValueError, "uses torch.float32"):
            build_flex_shard_muon(
                [{"params": parameters, "param_names": fqns}],
                compute_sharding_by_fqn={fqn: owned_compute for fqn in fqns},
                bucket_configs=(
                    BucketConfig(
                        patterns=fqns,
                        redistribution_mesh_axis_names=("dp_shard",),
                    ),
                ),
            )

    @with_comms
    def test_matches_plain_muon_across_flat_checkpoint(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        stack_shapes = {
            # Two storage shards each own six rows, so their boundary splits
            # the middle matrix and exercises overshard redistribution.
            "layers.0.attention.oversharded": (3, 4, 3),
            # These aligned siblings remain local and share one batched NS call.
            "layers.0.attention.wq": (4, 5, 3),
            "layers.0.attention.wkv": (4, 5, 3),
        }

        def make_parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )

        def make_optimizer(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            ns_steps: int = 2,
            matrix_sharding: BlockShard | Shard | None = None,
        ):
            redistributed_fqn = "layers.0.redistributed"
            oversharded_fqn = "layers.0.attention.oversharded"
            aligned_fqns = ("layers.0.attention.wq", "layers.0.attention.wkv")
            if matrix_sharding is None:
                matrix_sharding = BlockShard(dim=0, block_size=4)
            aligned_compute_sharding = ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=5),
                }
            )
            optimizer = build_flex_shard_muon(
                [
                    {
                        "params": [
                            redistributed,
                            stacks[oversharded_fqn],
                            *(stacks[fqn] for fqn in aligned_fqns),
                        ],
                        "param_names": [
                            redistributed_fqn,
                            oversharded_fqn,
                            *aligned_fqns,
                        ],
                    }
                ],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.8,
                nesterov=True,
                ns_steps=ns_steps,
                compute_sharding_by_fqn={
                    redistributed_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "alternate_mesh": BlockShard(dim=0, block_size=1),
                            "dp_shard": Owned(),
                        },
                    ),
                    oversharded_fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": matrix_sharding},
                    ),
                    **{fqn: aligned_compute_sharding for fqn in aligned_fqns},
                },
                bucket_configs=[
                    BucketConfig(
                        patterns=(redistributed_fqn, oversharded_fqn),
                        redistribution_mesh_axis_names=("dp_shard",),
                    ),
                    BucketConfig(
                        patterns=aligned_fqns,
                        redistribution_mesh_axis_names=NoRedistribution(),
                    ),
                ],
            )
            integration = _get_muon_reshard_integration(optimizer)
            self.assertEqual(len(integration._specs), 2)
            self.assertEqual(len(integration._bucket_plans), 1)
            overlap = integration._bucket_plans[0]
            self.assertIsInstance(overlap, _BucketOverlapPlan)
            overlap = cast(_BucketOverlapPlan, overlap)
            self.assertEqual(
                {
                    item.fqn
                    for item in overlap.redistribution_bucket.redistributed_items
                },
                {redistributed_fqn, oversharded_fqn},
            )
            self.assertEqual(len(overlap.local_buckets), 1)
            self.assertEqual(
                {item.fqn for item in overlap.local_buckets[0].items},
                set(aligned_fqns),
            )
            return optimizer

        def set_grads(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            redistributed_grad: torch.Tensor,
            stack_grads: dict[str, torch.Tensor],
        ) -> None:
            redistributed.grad = distribute_tensor(
                redistributed_grad.clone(), mesh, (Shard(0),)
            )
            for name, parameter in stacks.items():
                grad = stack_grads[name]
                parameter.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))

        def assert_matches_reference(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            reference_redistributed: torch.nn.Parameter,
            reference_stacks: dict[str, tuple[torch.nn.Parameter, ...]],
            stacks_before: dict[str, torch.Tensor],
            reference_stacks_before: dict[str, tuple[torch.Tensor, ...]],
        ) -> None:
            rank = mesh.get_local_rank()
            expected_redistributed = reference_redistributed.detach().chunk(
                self.world_size, dim=0
            )[rank]
            torch.testing.assert_close(
                redistributed.to_local(),
                expected_redistributed,
                rtol=0,
                atol=0,
            )

            for name, parameter in stacks.items():
                reference_blocks = reference_stacks[name]
                expected = torch.cat(
                    [reference.detach() for reference in reference_blocks], dim=0
                )
                expected_before = torch.cat(reference_stacks_before[name], dim=0)
                local_rows, row_offset = Shard.local_shard_size_and_offset(
                    expected.shape[0], self.world_size, rank
                )
                expected = expected.narrow(0, row_offset, local_rows)
                expected_before = expected_before.narrow(0, row_offset, local_rows)
                decay = 1 - lr * weight_decay
                adjusted_lr = _adjust_muon_learning_rate(
                    lr, None, reference_blocks[0].shape
                )
                actual_update = (
                    stacks_before[name] * decay - parameter.to_local()
                ) / adjusted_lr
                expected_update = (expected_before * decay - expected) / adjusted_lr
                # Batched BF16 Newton-Schulz can differ slightly across GEMM schedules.
                torch.testing.assert_close(
                    actual_update,
                    expected_update,
                    rtol=0,
                    atol=2e-2,
                )

        values = {}
        start = 12
        for name, (num_matrices, rows, columns) in stack_shapes.items():
            numel = num_matrices * rows * columns
            values[name] = (
                torch.arange(start, start + numel, device=device)
                .reshape(num_matrices * rows, columns)
                .float()
                .div_(10)
            )
            start += numel

        redistributed_value = (
            torch.arange(12, device=device).reshape(4, 3).float().div_(10).add_(1)
        )
        redistributed = make_parameter(redistributed_value)
        stacks = {name: make_parameter(value) for name, value in values.items()}
        with self.assertRaisesRegex(
            ValueError,
            "cannot be partitioned into 5-row Muon matrices",
        ):
            make_optimizer(
                redistributed,
                stacks,
                matrix_sharding=BlockShard(dim=0, block_size=5),
            )
        with self.assertRaisesRegex(
            ValueError,
            "2D Muon compute cannot use Shard",
        ):
            make_optimizer(
                redistributed,
                stacks,
                matrix_sharding=Shard(0),
            )
        optimizer = make_optimizer(redistributed, stacks)
        self.assertIs(type(optimizer), torch.optim.Muon)
        with self.assertRaisesRegex(RuntimeError, "after registering"):
            optimizer.add_param_group({"params": []})

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_stacks = {
            name: tuple(
                torch.nn.Parameter(matrix.clone())
                for matrix in value.view(stack_shapes[name])
            )
            for name, value in values.items()
        }
        reference_optimizer = torch.optim.Muon(
            [
                reference_redistributed,
                *(
                    parameter
                    for stack in reference_stacks.values()
                    for parameter in stack
                ),
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        def step_and_assert(
            current_optimizer,
            current_redistributed: torch.nn.Parameter,
            current_stacks: dict[str, torch.nn.Parameter],
            redistributed_grad: torch.Tensor,
            stack_grads: dict[str, torch.Tensor],
        ) -> None:
            stacks_before = {
                name: parameter.to_local().clone()
                for name, parameter in current_stacks.items()
            }
            reference_stacks_before = {
                name: tuple(parameter.detach().clone() for parameter in stack)
                for name, stack in reference_stacks.items()
            }
            set_grads(
                current_redistributed,
                current_stacks,
                redistributed_grad,
                stack_grads,
            )
            reference_redistributed.grad = redistributed_grad.clone()
            for name, references in reference_stacks.items():
                for parameter, grad in zip(
                    references,
                    stack_grads[name].view(stack_shapes[name]),
                    strict=True,
                ):
                    parameter.grad = grad.clone()

            current_optimizer.step()
            reference_optimizer.step()
            assert_matches_reference(
                current_redistributed,
                current_stacks,
                reference_redistributed,
                reference_stacks,
                stacks_before,
                reference_stacks_before,
            )

        first_redistributed_grad = (
            torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        )
        first_stack_grads = {
            name: torch.arange(1, value.numel() + 1, device=device)
            .reshape_as(value)
            .float()
            .div_(19 + 2 * index)
            for index, (name, value) in enumerate(values.items())
        }
        first_stack_grads["layers.0.attention.wkv"] = (
            first_stack_grads["layers.0.attention.wkv"].flip(1).contiguous()
        )
        execute_packed_all_to_all = (
            _optimizer_reshard_runtime._execute_packed_all_to_all
        )
        with patch.object(
            _optimizer_reshard_runtime,
            "_execute_packed_all_to_all",
            wraps=execute_packed_all_to_all,
        ) as execute_collective:
            step_and_assert(
                optimizer,
                redistributed,
                stacks,
                first_redistributed_grad,
                first_stack_grads,
            )
        self.assertEqual(execute_collective.call_count, 2)

        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_redistributed = make_parameter(redistributed.full_tensor().detach())
        resumed_stacks = {
            name: make_parameter(parameter.full_tensor().detach())
            for name, parameter in stacks.items()
        }
        resumed_optimizer = make_optimizer(
            resumed_redistributed,
            resumed_stacks,
            ns_steps=3,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)

        step_and_assert(
            resumed_optimizer,
            resumed_redistributed,
            resumed_stacks,
            first_redistributed_grad.flip(0).contiguous(),
            {
                name: grad.flip(0).contiguous()
                for name, grad in first_stack_grads.items()
            },
        )


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestFlexShardMuonMultiMesh(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_matrix_batch_uses_one_active_mesh_axis(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(36, device=device, dtype=torch.float32).reshape(12, 3)
        fqn = "layers.0.attention.wq.weight"

        def make_optimizer(
            placements: tuple[Replicate | Shard, Replicate | Shard],
            compute_layout: ComputeLayout,
            redistribution_mesh_axis_names: tuple[str, ...] = ("dp_shard",),
        ) -> torch.optim.Muon:
            parameter = torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, placements)
            )
            return build_flex_shard_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={fqn: compute_layout},
                bucket_configs=[
                    BucketConfig(
                        patterns=(fqn,),
                        redistribution_mesh_axis_names=(redistribution_mesh_axis_names),
                    )
                ],
            )

        optimizer = make_optimizer(
            (Replicate(), Shard(0)),
            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=4),
                }
            ),
        )
        self.assertIs(type(optimizer), torch.optim.Muon)
        integration = _get_muon_reshard_integration(optimizer)
        compute_layout = integration._require_optimizer_reshard_binding().plan_items[0]
        self.assertFalse(compute_layout.storage_is_compute_ready)
        self.assertEqual(compute_layout.redistribution_storage_mesh_axes, (1,))

        with self.assertRaisesRegex(
            NotImplementedError,
            "multiple active mesh axes.*only one active BlockShard axis",
        ):
            make_optimizer(
                (Shard(0), Shard(0)),
                ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_replicate": BlockShard(dim=0, block_size=4),
                        "dp_shard": BlockShard(dim=0, block_size=4),
                    }
                ),
                ("dp_replicate", "dp_shard"),
            )

    @with_comms
    def test_rejects_missing_physical_bucket_axis_alternative(self):
        dense_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        sparse_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        device = torch.device(self.device_type, self.rank)
        dense_fqn = "layers.0.dense"
        expert_fqn = "layers.0.expert"
        dense = torch.nn.Parameter(
            distribute_tensor(
                torch.arange(32, device=device).reshape(8, 4).float(),
                dense_mesh,
                (Shard(0),),
            )
        )
        expert = torch.nn.Parameter(
            distribute_tensor(
                torch.arange(64, device=device).reshape(4, 4, 4).float(),
                sparse_mesh,
                (Shard(1), Shard(0)),
            )
        )

        with self.assertRaisesRegex(ValueError, "requires redistribution axes"):
            build_flex_shard_muon(
                [{"params": (dense, expert), "param_names": (dense_fqn, expert_fqn)}],
                compute_sharding_by_fqn={
                    dense_fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Owned()},
                    ),
                    expert_fqn: ComputeLayout(
                        shardings_by_mesh_axis={"efsdp": Shard(0)},
                    ),
                },
                bucket_configs=(
                    BucketConfig(
                        patterns=("layers.0.*",),
                        redistribution_mesh_axis_names=("dp_shard",),
                    ),
                ),
            )

    @with_comms
    def test_rejects_mixed_owned_and_placement_redistribution(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(64, device=device).reshape(8, 8).float()
        parameter = torch.nn.Parameter(
            distribute_tensor(value, mesh, (Shard(1), Shard(0)))
        )
        fqn = "layers.0.mixed_assignment"

        with self.assertRaisesRegex(
            NotImplementedError,
            "cannot combine Owned and placement redistribution",
        ):
            build_flex_shard_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "efsdp": Owned(),
                            "ep": Replicate(),
                        },
                    )
                },
                bucket_configs=[
                    BucketConfig(
                        patterns=(fqn,),
                        redistribution_mesh_axis_names=("efsdp", "ep"),
                    )
                ],
            )

    @with_comms
    def test_explicit_multi_mesh_physical_buckets(self):
        lr = 0.03
        weight_decay = 0.2
        dense_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        sparse_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        repeated_shard_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("ep", "efsdp"),
        )
        device = torch.device(self.device_type, self.rank)

        dense_value = (
            torch.arange(24, device=device).reshape(8, 3).float().div_(11).add_(1)
        )
        jointly_assigned_fqn = "layers.0.jointly_assigned"
        jointly_assigned_value = (
            torch.arange(240, 288, device=device).reshape(8, 6).float().div_(29).add_(6)
        )
        jointly_assigned_storage_placements = (Shard(1), Shard(0))
        fully_replicated_fqn = "layers.0.routed_experts.fully_replicated"
        sparse_values = {
            "layers.0.routed_experts.sharded": (
                torch.arange(60, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(13)
                .add_(2)
            ),
            "layers.0.routed_experts.replicated": (
                torch.arange(60, 120, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(17)
                .add_(3)
            ),
            "layers.0.routed_experts.repeated_shard": (
                torch.arange(120, 180, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(19)
                .add_(4)
            ),
            fully_replicated_fqn: (
                torch.arange(180, 240, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(23)
                .add_(5)
            ),
        }
        sparse_storage_layouts = {
            "layers.0.routed_experts.sharded": (
                sparse_mesh,
                (Shard(1), Shard(0)),
            ),
            "layers.0.routed_experts.replicated": (
                sparse_mesh,
                (Shard(1), Shard(0)),
            ),
            "layers.0.routed_experts.repeated_shard": (
                repeated_shard_mesh,
                (Shard(0), Shard(0)),
            ),
            fully_replicated_fqn: (
                sparse_mesh,
                (Shard(1), Shard(0)),
            ),
        }
        dense = torch.nn.Parameter(
            distribute_tensor(dense_value.clone(), dense_mesh, (Shard(0),))
        )
        jointly_assigned = torch.nn.Parameter(
            distribute_tensor(
                jointly_assigned_value.clone(),
                sparse_mesh,
                jointly_assigned_storage_placements,
            )
        )
        sparse = {
            fqn: torch.nn.Parameter(
                distribute_tensor(
                    value.clone(),
                    *sparse_storage_layouts[fqn],
                )
            )
            for fqn, value in sparse_values.items()
        }
        dense_fqn = "layers.0.dense"
        optimizer = build_flex_shard_muon(
            [
                {
                    "params": [dense, jointly_assigned, *sparse.values()],
                    "param_names": [dense_fqn, jointly_assigned_fqn, *sparse],
                }
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            compute_sharding_by_fqn={
                dense_fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_shard": Owned(),
                    },
                ),
                jointly_assigned_fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "efsdp": Owned(),
                        "ep": Owned(),
                    },
                ),
                "layers.0.routed_experts.sharded": ComputeLayout(
                    shardings_by_mesh_axis={"efsdp": Shard(0)},
                ),
                "layers.0.routed_experts.replicated": ComputeLayout(
                    shardings_by_mesh_axis={
                        "efsdp": Replicate(),
                        "ep": Shard(0),
                    },
                ),
                "layers.0.routed_experts.repeated_shard": ComputeLayout(
                    shardings_by_mesh_axis={"efsdp": Replicate()},
                ),
                fully_replicated_fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "efsdp": Replicate(),
                        "ep": Replicate(),
                    },
                ),
            },
            bucket_configs=[
                BucketConfig(
                    patterns=(dense_fqn,),
                    redistribution_mesh_axis_names=("dp_shard",),
                ),
                BucketConfig(
                    patterns=(jointly_assigned_fqn,),
                    redistribution_mesh_axis_names=("efsdp", "ep"),
                ),
                BucketConfig(
                    patterns=("layers.0.routed_experts.sharded",),
                    redistribution_mesh_axis_names=("efsdp",),
                ),
                BucketConfig(
                    patterns=("layers.0.routed_experts.replicated",),
                    redistribution_mesh_axis_names=("efsdp",),
                ),
                BucketConfig(
                    patterns=("layers.0.routed_experts.repeated_shard",),
                    redistribution_mesh_axis_names=("efsdp",),
                ),
                BucketConfig(
                    patterns=(fully_replicated_fqn,),
                    redistribution_mesh_axis_names=("ep", "efsdp"),
                ),
            ],
        )
        integration = _get_muon_reshard_integration(optimizer)
        self.assertEqual(
            tuple(spec.redistribution_mesh_axis_names for spec in integration._specs),
            (
                ("dp_shard",),
                ("efsdp", "ep"),
                ("efsdp",),
                ("efsdp",),
                ("efsdp",),
                ("ep", "efsdp"),
            ),
        )
        redistribution_buckets = []
        for bucket_plan in integration._bucket_plans:
            self.assertIsInstance(bucket_plan, _RedistributionBucketPlan)
            redistribution_buckets.append(bucket_plan)

        def redistribution_plan_for(fqn: str):
            return next(
                redistribution_plan
                for bucket_plan in redistribution_buckets
                for item, redistribution_plan in zip(
                    bucket_plan.redistributed_items,
                    bucket_plan.redistribution_plans,
                    strict=True,
                )
                if item.fqn == fqn
            )

        transport_groups = {
            frozenset(plan.group.participants) for plan in redistribution_buckets
        }
        self.assertEqual(
            transport_groups,
            {
                frozenset(range(self.world_size)),
                frozenset(sparse_mesh["efsdp"].mesh.flatten().tolist()),
                frozenset(repeated_shard_mesh["efsdp"].mesh.flatten().tolist()),
            },
        )
        fully_replicated_plan = redistribution_plan_for(fully_replicated_fqn)
        fully_replicated_bucket = next(
            bucket_plan
            for bucket_plan in redistribution_buckets
            if fully_replicated_fqn
            in {item.fqn for item in bucket_plan.redistributed_items}
        )
        self.assertEqual(
            fully_replicated_bucket.group.participant_by_shard_index,
            tuple(sparse_mesh.mesh.permute(1, 0).flatten().tolist()),
        )
        self.assertEqual(
            frozenset(fully_replicated_plan.participants),
            frozenset(range(self.world_size)),
        )
        self.assertEqual(
            tuple(
                partition.tensor_shape
                for partition in fully_replicated_plan.compute_partitions
            ),
            (tuple(sparse_values[fully_replicated_fqn].shape),) * self.world_size,
        )
        jointly_assigned_plan = redistribution_plan_for(jointly_assigned_fqn)
        self.assertEqual(
            frozenset(jointly_assigned_plan.participants),
            frozenset(range(self.world_size)),
        )
        jointly_assigned_compute_shapes = tuple(
            partition.tensor_shape
            for partition in jointly_assigned_plan.compute_partitions
        )
        self.assertEqual(
            jointly_assigned_compute_shapes.count(tuple(jointly_assigned_value.shape)),
            1,
        )
        self.assertEqual(
            jointly_assigned_compute_shapes.count((0,)),
            self.world_size - 1,
        )

        dense_grad = torch.arange(1, 25, device=device).reshape(8, 3).float().div_(19)
        jointly_assigned_grad = (
            torch.arange(1, 49, device=device)
            .reshape_as(jointly_assigned_value)
            .float()
            .div_(23)
            .sin_()
        )
        sparse_grads = {
            fqn: torch.arange(60, device=device)
            .reshape_as(value)
            .float()
            .mul_(0.37 + 0.11 * index)
            .add_(0.2 + index)
            .sin_()
            for index, (fqn, value) in enumerate(sparse_values.items())
        }
        dense.grad = distribute_tensor(dense_grad.clone(), dense_mesh, (Shard(0),))
        jointly_assigned.grad = distribute_tensor(
            jointly_assigned_grad.clone(),
            sparse_mesh,
            jointly_assigned_storage_placements,
        )
        for fqn, parameter in sparse.items():
            storage_mesh, placements = sparse_storage_layouts[fqn]
            parameter.grad = distribute_tensor(
                sparse_grads[fqn].clone(),
                storage_mesh,
                placements,
            )

        reference_dense = torch.nn.Parameter(dense_value.clone())
        reference_jointly_assigned = torch.nn.Parameter(jointly_assigned_value.clone())
        reference_sparse = {
            fqn: tuple(torch.nn.Parameter(matrix.clone()) for matrix in value)
            for fqn, value in sparse_values.items()
        }
        reference_optimizer = torch.optim.Muon(
            [
                reference_dense,
                reference_jointly_assigned,
                *(
                    parameter
                    for matrices in reference_sparse.values()
                    for parameter in matrices
                ),
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        reference_dense.grad = dense_grad.clone()
        reference_jointly_assigned.grad = jointly_assigned_grad.clone()
        for fqn, matrices in reference_sparse.items():
            for parameter, grad in zip(
                matrices,
                sparse_grads[fqn],
                strict=True,
            ):
                parameter.grad = grad.clone()

        optimizer.step()
        reference_optimizer.step()

        decay = 1 - lr * weight_decay
        dense_adjusted_lr = _adjust_muon_learning_rate(lr, None, dense_value.shape)
        torch.testing.assert_close(
            (dense_value * decay - dense.full_tensor()) / dense_adjusted_lr,
            (dense_value * decay - reference_dense) / dense_adjusted_lr,
            rtol=0,
            atol=2e-2,
        )
        jointly_assigned_adjusted_lr = _adjust_muon_learning_rate(
            lr,
            None,
            jointly_assigned_value.shape,
        )
        torch.testing.assert_close(
            (jointly_assigned_value * decay - jointly_assigned.full_tensor())
            / jointly_assigned_adjusted_lr,
            (jointly_assigned_value * decay - reference_jointly_assigned)
            / jointly_assigned_adjusted_lr,
            rtol=0,
            atol=2e-2,
        )
        self.assertEqual(
            jointly_assigned.placements,
            jointly_assigned_storage_placements,
        )
        for fqn, parameter in sparse.items():
            expected = torch.stack(
                [reference.detach() for reference in reference_sparse[fqn]]
            )
            adjusted_lr = _adjust_muon_learning_rate(
                lr,
                None,
                reference_sparse[fqn][0].shape,
            )
            actual_update = (
                sparse_values[fqn] * decay - parameter.full_tensor()
            ) / adjusted_lr
            expected_update = (sparse_values[fqn] * decay - expected) / adjusted_lr
            torch.testing.assert_close(
                actual_update,
                expected_update,
                rtol=0,
                atol=2e-2,
            )
            self.assertEqual(parameter.placements, sparse_storage_layouts[fqn][1])


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires eight CUDA devices")
class TestFlexShardMuonJointOwnedValidation(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_builds_stage_local_cartesian_groups_in_different_axis_orders(self):
        root_mesh = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("pp", "efsdp", "ep"),
        )
        stage_mesh = root_mesh["efsdp", "ep"]
        coordinate = root_mesh.get_coordinate()
        assert coordinate is not None
        pp_rank = coordinate[0]
        fqn = f"stages.{pp_rank}.weight"
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(48, device=device).reshape(8, 6).float()
        parameter = torch.nn.Parameter(
            distribute_tensor(value, stage_mesh, (Shard(1), Shard(0)))
        )
        axis_names_by_stage = (
            ("efsdp", "ep"),
            ("ep", "efsdp"),
        )

        optimizer = build_flex_shard_muon(
            [{"params": (parameter,), "param_names": (fqn,)}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "efsdp": Owned(),
                        "ep": Owned(),
                    }
                )
            },
            bucket_configs=tuple(
                BucketConfig(
                    patterns=(f"stages.{stage}.*",),
                    redistribution_mesh_axis_names=axis_names,
                )
                for stage, axis_names in enumerate(axis_names_by_stage)
            ),
        )

        integration = _get_muon_reshard_integration(optimizer)
        self.assertEqual(len(integration._bucket_plans), 1)
        plan = integration._bucket_plans[0]
        self.assertIsInstance(plan, _RedistributionBucketPlan)
        expected_participant_by_shard_index = stage_mesh.mesh
        if pp_rank == 1:
            expected_participant_by_shard_index = (
                expected_participant_by_shard_index.permute(1, 0)
            )
        self.assertEqual(
            plan.group.participant_by_shard_index,
            tuple(expected_participant_by_shard_index.reshape(-1).tolist()),
        )

    @with_comms
    def test_rejects_nonreplicated_axis_outside_joint_owned(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("efsdp", "ep", "dp_replicate"),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(64, device=device).reshape(8, 8).float()
        parameter = torch.nn.Parameter(
            distribute_tensor(
                value,
                mesh,
                (Shard(1), Shard(0), Shard(0)),
            )
        )
        fqn = "layers.0.jointly_assigned"

        with self.assertRaisesRegex(
            NotImplementedError,
            "cannot preserve non-replicated mesh axis 'dp_replicate'",
        ):
            build_flex_shard_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "efsdp": Owned(),
                            "ep": Owned(),
                        },
                    )
                },
                bucket_configs=[
                    BucketConfig(
                        patterns=(fqn,),
                        redistribution_mesh_axis_names=("efsdp", "ep"),
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
