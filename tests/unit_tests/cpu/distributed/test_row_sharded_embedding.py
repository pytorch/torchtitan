# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from torchtitan.components.checkpointer.base import export_state_dict
from torchtitan.components.optim.utils import (
    get_flat_optim_state_dict,
    load_flat_optim_state_dict,
)

from torchtitan.distributed.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)
from torchtitan.distributed.utils import clip_grad_norm_


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
            mesh_dim_names=("column", "row"),
        )
        row_mesh = mesh["row"]
        owner = row_mesh.get_local_rank()
        column = self.rank // 2

        for case in (
            "uneven",
            "empty_requester",
            "empty_owner",
            "empty",
            "int32",
            "bfloat16",
            "bfloat16_cancel",
            "scalar",
            "recompute",
            "remat",
        ):
            with self.subTest(case=case):
                num_rows = 1 if case == "empty_owner" else 5
                num_features = 3
                # Distinct values in independent row groups catch world-group
                # collectives; binary-exact gradients catch unwanted averaging.
                initial = (
                    torch.arange(
                        num_rows * num_features, device=self.device_type
                    ).reshape(num_rows, num_features)
                    + 100 * column
                ).float() / 16
                reference = torch.nn.Parameter(initial.clone())
                rows_per_owner = (num_rows + row_mesh.size() - 1) // row_mesh.size()
                row_start = owner * rows_per_owner
                row_end = min(row_start + rows_per_owner, num_rows)
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
            mesh_dim_names=("column", "row"),
        )
        row_mesh = mesh["row"]
        for invalid_id in (-1, 5):
            ids = torch.tensor(
                [invalid_id if row_mesh.get_local_rank() == 0 else 0],
                device=self.device_type,
            )
            with self.assertRaisesRegex(ValueError, "on every owner"):
                prepare_row_sharded_embedding(ids, num_embeddings=5, row_mesh=row_mesh)

    @with_comms
    def test_global_clipping_and_native_checkpoint_resharding(self):
        """Clip owned rows globally and resume weights/moments on fewer owners."""
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        row_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("row",)
        )
        initial = torch.arange(15, device=self.device_type).reshape(5, 3).float() / 16
        reference = torch.nn.Parameter(initial.clone())
        reference_optimizer = torch.optim.AdamW(
            [{"params": [reference], "param_names": ["lookup"]}],
            lr=0.125,
            **self.optimizer_kwargs,
        )

        def build_owner(mesh):
            rows_per_owner = (5 + mesh.size() - 1) // mesh.size()
            begin = min(mesh.get_local_rank() * rows_per_owner, 5)
            end = min(begin + rows_per_owner, 5)
            weight = torch.nn.Parameter(
                DTensor.from_local(
                    initial[begin:end].clone(),
                    mesh,
                    (Shard(0),),
                    shape=initial.shape,
                    stride=initial.stride(),
                    run_check=False,
                )
            )
            optimizer = torch.optim.AdamW(
                [{"params": [weight], "param_names": ["lookup"]}],
                lr=0.125,
                **self.optimizer_kwargs,
            )
            return weight, optimizer, begin, end

        weight, optimizer, begin, end = build_owner(row_mesh)
        # Include a replicated parameter and clipping that is actually active.
        ordinary = torch.nn.Parameter(torch.zeros(1, device=self.device_type))
        ordinary_reference = torch.nn.Parameter(torch.zeros_like(ordinary))
        # Native DTensor L2 reduction includes owners with no local rows.
        reference.grad = torch.zeros_like(reference)
        reference.grad[0, 0] = 3
        reference.grad[2, 0] = 4
        weight.grad = DTensor.from_local(
            reference.grad[begin:end].clone(),
            row_mesh,
            (Shard(0),),
            shape=initial.shape,
            stride=initial.stride(),
            run_check=False,
        )
        ordinary.grad = torch.tensor([12.0], device=self.device_type)
        ordinary_reference.grad = ordinary.grad.clone()
        actual = clip_grad_norm_([ordinary, weight], 0.5, foreach=True)
        expected = torch.nn.utils.clip_grad_norm_(
            [ordinary_reference, reference], 0.5, foreach=True
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(
            weight.grad.to_local(), reference.grad[begin:end], atol=0, rtol=0
        )
        self.assertEqual(ordinary.grad, ordinary_reference.grad, atol=0, rtol=0)

        optimizer.step()
        reference_optimizer.step()
        state = {"weight": weight.detach(), **get_flat_optim_state_dict(optimizer)}
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path, src=0)
        checkpoint_pg = dist.new_group(backend="gloo")
        try:
            for asynchronous in (False, True):
                checkpoint_path = path[0] + ("/async" if asynchronous else "/sync")
                if asynchronous:
                    dcp.async_save(
                        state,
                        checkpoint_id=checkpoint_path,
                        process_group=checkpoint_pg,
                    ).result()
                else:
                    dcp.save(
                        state,
                        checkpoint_id=checkpoint_path,
                        process_group=checkpoint_pg,
                    )
                saved = dcp.FileSystemReader(checkpoint_path).read_metadata()
                self.assertEqual(
                    saved.state_dict_metadata["weight"].size, torch.Size((5, 3))
                )
            exported = export_state_dict(
                {"weight": state["weight"]}, dtype=torch.bfloat16, model_parts=[]
            )
            dcp.save(
                exported, checkpoint_id=path[0] + "/export", process_group=checkpoint_pg
            )
            saved = dcp.FileSystemReader(path[0] + "/export").read_metadata()
            self.assertEqual(
                saved.state_dict_metadata["weight"].size, torch.Size((5, 3))
            )
            self.assertEqual(
                saved.state_dict_metadata["weight"].properties.dtype, torch.bfloat16
            )
            # Two independent groups restore the same global table with a
            # different row partition, including the Adam moments.
            smaller = init_device_mesh(
                self.device_type,
                (2, self.world_size // 2),
                mesh_dim_names=("replica", "row"),
            )["row"]
            loaded, loaded_optimizer, begin, end = build_owner(smaller)
            loaded.grad = torch.zeros_like(loaded)
            loaded_optimizer.step()
            loaded_state = {
                "weight": loaded.detach(),
                **get_flat_optim_state_dict(loaded_optimizer),
            }
            dcp.load(
                loaded_state, checkpoint_id=checkpoint_path, process_group=checkpoint_pg
            )
            load_flat_optim_state_dict(loaded_optimizer, loaded_state)
            self.assertEqual(loaded.to_local(), reference[begin:end], atol=0, rtol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(
                    loaded_optimizer.state[loaded][key].to_local(),
                    reference_optimizer.state[reference][key][begin:end],
                    atol=0,
                    rtol=0,
                )
            # A resumed update must include decay/momentum for unvisited rows.
            reference.grad = torch.zeros_like(reference)
            reference.grad[0] = 1
            loaded.grad = DTensor.from_local(
                reference.grad[begin:end].clone(),
                smaller,
                (Shard(0),),
                shape=initial.shape,
                stride=initial.stride(),
                run_check=False,
            )
            loaded_optimizer.step()
            reference_optimizer.step()
            self.assertEqual(loaded.to_local(), reference[begin:end], atol=0, rtol=0)
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path[0])
