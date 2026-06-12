# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.flex_shard.example.shard import per_param_placements
from torchtitan.experiments.flex_shard.tests.common import (
    check_flex_shard_parity,
    expected_shard,
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _average_reference_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def _prefer_recompute_context_fn():
    def prefer_recompute_policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(prefer_recompute_policy)


def _checkpoint_transformer_execution_units(model: torch.nn.Module) -> None:
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


def _layer_buckets_with_grouped_root_rest(
    num_layers: int,
    mesh,
    *,
    reshard_after_forward: bool,
) -> list[BucketSpec]:
    return [
        *[
            BucketSpec(
                [f"layers.{idx}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
            for idx in range(num_layers)
        ],
        BucketSpec(
            ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"],
            placement_fn=per_param_placements,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        ),
    ]


class TestFlexShardTraining(FSDPTest):
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
        flex_shard(
            model,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=False,
            ),
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
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

        optim.step()
        ref_optim.step()

        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

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

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["tok_embeddings.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                    ),
                    reshard_after_forward=False,
                ),
                *transformer_bucket_specs(
                    args.n_layers, mesh, reshard_after_forward=False
                )[1:],
            ],
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
        expected_grad = expected_shard(
            ref_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
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

        flex_shard(
            model,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=True,
            ),
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
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ):
            loss = model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()

        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

        optim.step()
        ref_optim.step()
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_grouped_root_rest_bucket_unsupported(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type, n_layers=2)
        _checkpoint_transformer_execution_units(model)

        with self.assertRaisesRegex(RuntimeError, "recomputation-safe"):
            flex_shard(
                model,
                buckets=_layer_buckets_with_grouped_root_rest(
                    args.n_layers,
                    mesh,
                    reshard_after_forward=True,
                ),
            )


if __name__ == "__main__":
    run_tests()
