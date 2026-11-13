# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)


class TestRowShardedEmbedding(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def optimizer_kwargs(self) -> dict:
        return {"foreach": False}

    @with_comms
    def test_lookup_and_adamw_match_dense(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("replica", "row"),
        )
        row_mesh = mesh["row"]
        owner = row_mesh.get_local_rank()
        replica = mesh["replica"].get_local_rank()

        for case in (
            "empty_requester",
            "empty",
            "int32",
            "bfloat16",
            "bfloat16_cancel",
            "scalar",
            "recompute",
            "remat",
        ):
            with self.subTest(case=case):
                num_rows = 4
                num_features = 3
                # Distinct values in independent row groups catch world-group
                # collectives; binary-exact gradients catch unwanted averaging.
                initial = (
                    torch.arange(
                        num_rows * num_features, device=self.device_type
                    ).reshape(num_rows, num_features)
                    + 100 * replica
                ).float() / 16
                reference = torch.nn.Parameter(initial.clone())
                rows_per_owner = num_rows // row_mesh.size()
                row_start = owner * rows_per_owner
                row_end = row_start + rows_per_owner
                weight = torch.nn.Parameter(
                    DTensor.from_local(
                        initial[row_start:row_end].clone(),
                        row_mesh,
                        (Shard(0),),
                        shape=initial.shape,
                        stride=initial.stride(),
                        run_check=False,
                    )
                )
                optimizer = torch.optim.AdamW(
                    [weight], lr=0.125, **self.optimizer_kwargs
                )
                reference_optimizer = torch.optim.AdamW(
                    [reference], lr=0.125, **self.optimizer_kwargs
                )

                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    before = weight.detach().clone()
                    for microbatch in range(2):
                        if case == "empty" or (
                            case == "empty_requester" and owner == 0
                        ):
                            ids = torch.empty(
                                (0, 3), dtype=torch.int64, device=self.device_type
                            )
                        else:
                            ids = torch.tensor(
                                [[4, 0, 4], [1 + owner, 2 + owner, 0]],
                                device=self.device_type,
                            )
                            ids = (ids + step + microbatch) % num_rows
                        if case == "int32":
                            if owner == 0:
                                ids = ids.to(torch.int32)
                            ids = ids.T
                        if case == "scalar":
                            ids = ids[0, 0]
                        if case == "bfloat16_cancel":
                            ids = torch.zeros(
                                2 if owner == 0 else 1,
                                dtype=torch.int64,
                                device=self.device_type,
                            )
                        metadata = prepare_row_sharded_embedding(
                            ids, num_embeddings=num_rows, row_mesh=row_mesh
                        )
                        if case == "recompute":
                            # All owners replay the response collective, even if
                            # a rank could finish its recomputation earlier.
                            with set_checkpoint_early_stop(False):
                                output = checkpoint(
                                    row_sharded_embedding,
                                    weight,
                                    metadata,
                                    use_reentrant=False,
                                )
                        elif case == "remat":
                            lookup = remat.region(
                                row_sharded_embedding, "lookup", recompute=True
                            )
                            output = remat.checkpoint()(lookup)(weight, metadata)
                        else:
                            output = row_sharded_embedding(
                                weight,
                                metadata,
                                dtype=torch.bfloat16
                                if case.startswith("bfloat16")
                                else None,
                            )
                        expected = F.embedding(ids, reference.to(output.dtype))
                        self.assertEqual(output, expected, atol=0, rtol=0)
                        coefficients = (
                            torch.arange(output.numel(), device=self.device_type)
                            .remainder(9)
                            .sub(4)
                            .reshape(output.shape)
                        ).float() / 4
                        if case == "bfloat16_cancel":
                            coefficients = torch.tensor(
                                [256.0, 1.0] if owner == 0 else [-256.0],
                                device=self.device_type,
                            )[:, None].expand(-1, num_features)
                        (output * coefficients).sum().backward()
                        (expected * coefficients).sum().backward()
                        self.assertEqual(weight, before, atol=0, rtol=0)

                    dist.all_reduce(reference.grad, group=row_mesh.get_group())
                    self.assertIsNotNone(weight.grad)
                    self.assertEqual(
                        weight.grad.to_local(),
                        reference.grad[row_start:row_end],
                        atol=0,
                        rtol=0,
                    )
                    optimizer.step()
                    reference_optimizer.step()
                    self.assertEqual(
                        weight.to_local(), reference[row_start:row_end], atol=0, rtol=0
                    )
                    for state_name in ("exp_avg", "exp_avg_sq"):
                        self.assertEqual(
                            optimizer.state[weight][state_name].to_local(),
                            reference_optimizer.state[reference][state_name][
                                row_start:row_end
                            ],
                            atol=0,
                            rtol=0,
                        )

    @with_comms
    def test_invalid_ids_fail_on_entire_row_group(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("replica", "row"),
        )
        row_mesh = mesh["row"]
        for invalid_id in (-1, 4):
            ids = torch.tensor(
                [invalid_id if row_mesh.get_local_rank() == 0 else 0],
                device=self.device_type,
            )
            with self.assertRaisesRegex(ValueError, "on every owner"):
                prepare_row_sharded_embedding(ids, num_embeddings=4, row_mesh=row_mesh)
