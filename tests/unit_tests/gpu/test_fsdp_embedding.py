# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from spmd_types import SpmdType
from spmd_types.checker import typecheck
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.fsdp import resolve_fsdp_mesh
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.embedding import Embedding
from torchtitan.protocols.sharding import ShardingConfig

pytestmark = pytest.mark.multi_gpu


class TestFSDPEmbedding(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_padding_lifecycle(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=4,
            enable_sequence_parallel=False,
        )
        fsdp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        runtime_mesh = parallel_dims.spmd_dense_mesh()
        dp_mesh = parallel_dims.get_mesh("dp_shard")
        tp_mesh = parallel_dims.get_mesh("tp")
        dp_rank = dp_mesh.get_local_rank()
        tp_rank = tp_mesh.get_local_rank()
        dp, cp, tp = MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP
        input_layout = SpmdType({dp: spmd.S(0), cp: spmd.S(1), tp: spmd.R})
        output_layout = SpmdType(
            {dp: spmd.V, cp: spmd.V, tp: spmd.V},
            partition_spec=spmd.PartitionSpec(dp, (cp, tp), None),
        )

        for padding_idx, reshard_after_forward in product(
            (None, 5, 69, 127, -1), (False, True)
        ):
            with self.subTest(
                padding_idx=padding_idx,
                reshard_after_forward=reshard_after_forward,
            ):
                torch.manual_seed(42)
                reference = nn.Embedding(
                    128,
                    16,
                    padding_idx=padding_idx,
                    device=self.device_type,
                )
                with torch.no_grad():
                    reference.weight.normal_()
                embedding = (
                    Embedding.Config(
                        num_embeddings=128,
                        embedding_dim=16,
                        padding_idx=padding_idx,
                        sharding_config=ShardingConfig(
                            state_shardings={
                                "weight": dense_param_placement(tp=spmd.S(0))
                            },
                            in_src_shardings={"input": input_layout},
                            in_dst_shardings={"input": input_layout},
                            out_src_shardings=SpmdType(
                                {dp: spmd.S(0), cp: spmd.S(1), tp: spmd.P}
                            ),
                            out_dst_shardings=output_layout,
                            local_spmd=True,
                        ),
                    )
                    .build()
                    .to(self.device_type)
                )
                embedding.load_state_dict(reference.state_dict())
                embedding._parallelize(parallel_dims)
                fully_shard(
                    embedding,
                    mesh=fsdp_mesh,
                    dp_mesh_dims=dp_mesh_dims,
                    reshard_after_forward=reshard_after_forward,
                )
                # Nest the embedding as in a decoder, so it is not the FSDP root.
                model = nn.Sequential(embedding)
                fully_shard(model, mesh=fsdp_mesh, dp_mesh_dims=dp_mesh_dims)
                self.assertIsInstance(embedding, FSDPModule)
                parameter = embedding.weight
                self.assertIsInstance(parameter, DTensor)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.125)

                for step in range(3):
                    # Different DP batches exercise gradient averaging; repeated
                    # IDs straddle the TP boundary and include both padding owners.
                    tokens = torch.tensor(
                        [[0, 5, 64, 69, 100 + dp_rank + step, 127, 64, 5]],
                        device=self.device_type,
                    )
                    reference_optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    expected = reference(tokens)
                    expected.sum().backward()
                    dist.all_reduce(reference.weight.grad, group=dp_mesh.get_group())
                    reference.weight.grad.div_(dp_mesh.size())

                    with set_current_spmd_mesh(runtime_mesh), typecheck(local=False):
                        spmd.assert_type(tokens, input_layout)
                        output = model(tokens)
                        if reshard_after_forward:
                            self.assertIs(embedding.weight, parameter)
                        else:
                            self.assertNotIsInstance(embedding.weight, DTensor)
                            self.assertEqual(embedding.weight.shape, (64, 16))
                        output.sum().backward()

                    self.assertEqual(
                        output,
                        expected.chunk(tp_mesh.size(), dim=1)[tp_rank],
                        atol=0,
                        rtol=0,
                    )
                    # FSDP must restore the optimizer's persistent sharded parameter
                    # after backward, including the padding-masked gradient.
                    self.assertIs(embedding.weight, parameter)
                    self.assertEqual(
                        parameter.grad.full_tensor(),
                        reference.weight.grad,
                        atol=0,
                        rtol=0,
                    )
                    optimizer.step()
                    reference_optimizer.step()
                    self.assertEqual(
                        model.state_dict()["0.weight"].full_tensor(),
                        reference.weight,
                        atol=0,
                        rtol=0,
                    )
