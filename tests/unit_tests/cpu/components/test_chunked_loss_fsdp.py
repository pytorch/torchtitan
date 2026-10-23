# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.loss import ChunkedLossWrapper


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 16)

    def forward(self, inputs):
        return self.norm(self.projection(inputs))


def _policies(head):
    state_ctx = head._get_fsdp_state()._state_ctx
    return state_ctx.is_last_backward, tuple(
        (
            state._auto_reshard_after_forward,
            tuple(
                (
                    group.post_forward_mesh_info,
                    group.reshard_after_backward,
                    group.reduce_grads,
                    group.all_reduce_grads,
                )
                for group in state._fsdp_param_groups
            ),
        )
        for state in state_ctx.all_states
    )


def _compare(actual, expected, label):
    if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6):
        difference = (actual - expected).abs().max().item()
        raise RuntimeError(f"{label}: maximum absolute difference {difference}")


class TestChunkedLossFSDP(DTensorTestBase):
    device_type = "cpu"

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_accumulation_and_recovery(self):
        torch.set_num_threads(1)
        device = torch.device(self.device_type, self.rank)
        for replicated, grouped, deferred, frozen in itertools.product(
            (False, True), repeat=4
        ):
            with self.subTest(
                replicated=replicated, grouped=grouped, deferred=deferred, frozen=frozen
            ):
                mesh = init_device_mesh(
                    self.device_type,
                    (2, 2) if replicated else (self.world_size,),
                    mesh_dim_names=("dp_replicate", "dp_shard")
                    if replicated
                    else ("dp_shard",),
                )
                torch.manual_seed(42)
                model = _Decoder().to(device)
                if frozen:
                    model.projection.requires_grad_(False)
                    model.norm.requires_grad_(False)
                reference = copy.deepcopy(model)
                fully_shard(model.projection, mesh=mesh)
                if grouped:
                    fully_shard(
                        [model.norm, model.lm_head],
                        mesh=mesh,
                        reshard_after_forward=False,
                    )
                else:
                    fully_shard(model.norm, mesh=mesh)
                    fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=False)
                fully_shard(model, mesh=mesh)
                loss_fn = ChunkedLossWrapper.Config(num_chunks=4).build()
                loss_fn.set_lm_head(model.lm_head)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                reference_optimizer = torch.optim.SGD(
                    reference.parameters(), lr=0.01, momentum=0.9
                )
                count = torch.tensor(24, device=device)

                def set_policy(final):
                    model.set_requires_gradient_sync(final or not deferred)
                    model.set_reshard_after_backward(final or not deferred)
                    model.set_is_last_backward(final if deferred else True)
                    if replicated:
                        model.set_requires_all_reduce(final)

                # Fail with earlier microbatches' full or HSDP partial gradients live.
                for _ in range(2):
                    set_policy(False)
                    inputs = torch.randn(8, 8, device=device)
                    labels = torch.randint(16, (8,), device=device)
                    loss, _ = loss_fn(model(inputs), labels, count)
                    loss.backward()
                set_policy(True)
                inputs = torch.randn(8, 8, device=device)
                labels = torch.randint(16, (8,), device=device)
                hidden = model(inputs)
                expected_policies = _policies(model.lm_head)
                real_loss = loss_fn.loss_fn
                chunks = 0

                def fail_second_chunk(*args, **kwargs):
                    nonlocal chunks
                    chunks += 1
                    if chunks == 2:
                        raise RuntimeError("controlled chunk failure")
                    return real_loss(*args, **kwargs)

                loss_fn.loss_fn = fail_second_chunk
                try:
                    loss_fn(hidden, labels, count)
                except RuntimeError as error:
                    if str(error) != "controlled chunk failure":
                        raise
                else:
                    raise RuntimeError("The controlled chunk failure did not execute")
                if _policies(model.lm_head) != expected_policies:
                    raise RuntimeError(
                        "A failed chunk changed the caller's FSDP policy"
                    )
                del hidden
                model.reset_iter_state()
                if _policies(model.lm_head) != expected_policies:
                    raise RuntimeError("FSDP reset changed the caller's policy")
                loss_fn.loss_fn = real_loss

                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    with patch.object(
                        dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
                    ) as reduce_scatter:
                        for microbatch in range(3):
                            final = microbatch == 2
                            set_policy(final)
                            torch.manual_seed(
                                1000 + 100 * step + 10 * self.rank + microbatch
                            )
                            inputs = torch.randn(8, 8, device=device)
                            labels = torch.randint(16, (8,), device=device)
                            hidden = model(inputs)
                            expected_policies = _policies(model.lm_head)
                            loss, _ = loss_fn(hidden, labels, count)
                            if _policies(model.lm_head) != expected_policies:
                                raise RuntimeError(
                                    "Chunked loss changed the caller's FSDP policy"
                                )
                            loss.backward()
                            expected_loss = (
                                torch.nn.functional.cross_entropy(
                                    reference.lm_head(reference(inputs)),
                                    labels,
                                    reduction="sum",
                                )
                                / count
                            )
                            expected_loss.backward()
                            _compare(loss.detach(), expected_loss.detach(), "loss")
                    if deferred:
                        expected_reductions = 1 if frozen else (2 if grouped else 3)
                        if reduce_scatter.call_count != expected_reductions:
                            raise RuntimeError(
                                f"Expected {expected_reductions} reductions after accumulation, "
                                f"got {reduce_scatter.call_count}"
                            )
                    for (name, parameter), (_, expected) in zip(
                        model.named_parameters(),
                        reference.named_parameters(),
                        strict=True,
                    ):
                        if parameter.requires_grad:
                            dist.all_reduce(expected.grad)
                            expected.grad.div_(self.world_size)
                            _compare(
                                parameter.grad.full_tensor(),
                                expected.grad,
                                f"{name}/gradient",
                            )
                    optimizer.step()
                    reference_optimizer.step()
                    for (name, parameter), (_, expected) in zip(
                        model.named_parameters(),
                        reference.named_parameters(),
                        strict=True,
                    ):
                        _compare(parameter.full_tensor(), expected, f"{name}/updated")
