# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import shutil
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.checkpointer.base import export_state_dict
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.embedding import NGramTable
from torchtitan.models.qwen3_8_flash_next.sharding import ngram_table_sharding
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)


class TestOwnedTable(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def backend(self) -> str:
        return "gloo"

    @with_comms
    def test_hf_fragments_round_trip_without_gathering(self):
        """PLE fragments avoid gathers; fused projections retain their row shards."""
        key = "layers.1.ple.ple_embedding.ngram_embedding.weight"
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path, src=0)
        try:
            for column_shards, num_rows, num_parts, hsdp in (
                (1, 60, 5, False),
                (2, 60, 5, False),
                (1, 60, 5, True),
            ):
                mesh = init_device_mesh(
                    "cpu",
                    (2, 2)
                    if hsdp
                    else (self.world_size // column_shards, column_shards),
                    mesh_dim_names=("dp_replicate", "dp_shard")
                    if hsdp
                    else ("dp_shard", "tp"),
                )
                config = build_model_config(
                    "debugmodel", parallelism=ParallelismConfig()
                )
                embedding = config.layers[1].ple.ple_embedding
                embedding.ngram_embedding.num_embeddings = num_rows
                embedding.split_ngram_parts = num_parts
                adapter = Qwen38FlashNextStateDictAdapter(config, None)
                full = torch.arange(num_rows * 6).reshape(num_rows, 6).float()
                placements = (Shard(0), Shard(0) if hsdp else Shard(1))
                weight = distribute_tensor(full, mesh, placements)
                # Four DP owners split GDN's two head groups. Projection/gate
                # conversion must still preserve each child's native placement.
                linear_state = {}
                for layer_id, field in ((0, "in_proj"), (3, "wq")):
                    linear = getattr(config.layers[layer_id].attention, field)
                    for child in ("projection", "gates"):
                        projection = getattr(linear, child)
                        shape = (projection.out_features, projection.in_features)
                        full_projection = torch.arange(
                            shape[0] * shape[1], dtype=torch.float32
                        ).reshape(shape)
                        linear_state[
                            f"layers.{layer_id}.attention.{field}.{child}.weight"
                        ] = distribute_tensor(full_projection, mesh, placements)
                linear_hf = adapter.to_hf(
                    {"lm_head.weight": torch.ones(1, 6), **linear_state}
                )
                linear_restored = adapter.from_hf(linear_hf)
                del linear_restored["lm_head.weight"]
                self.assertEqual(linear_restored.keys(), linear_state.keys())
                for name, original in linear_state.items():
                    restored_projection = linear_restored[name]
                    self.assertEqual(restored_projection.device_mesh, mesh)
                    self.assertEqual(restored_projection.placements, placements)
                    self.assertEqual(
                        restored_projection.to_local(),
                        original.to_local(),
                        atol=0,
                        rtol=0,
                    )
                row_owner, feature_owner = (
                    (self.rank, 0) if hsdp else mesh.get_coordinate()
                )
                num_owners = self.world_size if hsdp else mesh.size(0)
                rows_per_owner = num_rows // num_owners
                begin = row_owner * rows_per_owner
                end = begin + rows_per_owner
                feature_begin = feature_owner * (6 // column_shards)
                feature_end = feature_begin + 6 // column_shards
                expected = full[begin:end, feature_begin:feature_end].bfloat16()
                with CommDebugMode() as communication:
                    exported = export_state_dict(
                        {key: weight, "lm_head.weight": torch.ones(1, 6)},
                        dtype=torch.bfloat16,
                        model_parts=[],
                    )
                    hf = adapter.to_hf(exported)
                    restored = adapter.from_hf(dict(reversed(list(hf.items()))))
                self.assertEqual(communication.get_total_counts(), 0)
                with self.assertRaisesRegex(ValueError, "Rank-local HF"):
                    Qwen38FlashNextStateDictAdapter(config, None).from_hf(hf)
                self.assertIsInstance(restored[key], DTensor)
                self.assertEqual(restored[key].device_mesh, mesh)
                self.assertEqual(restored[key].placements, placements)
                self.assertEqual(
                    restored[key].to_local(grad_placements=placements),
                    expected,
                    atol=0,
                    rtol=0,
                )
                rows_per_part = num_rows // num_parts
                hf_shards = {
                    name: value for name, value in hf.items() if ".shard_" in name
                }
                expected_parts = [
                    part
                    for part in range(num_parts)
                    if max(begin, rows_per_part * part)
                    < min(end, rows_per_part * (part + 1))
                ]
                self.assertEqual(len(hf_shards), len(expected_parts))
                for part, (name, fragment) in zip(
                    expected_parts, hf_shards.items(), strict=True
                ):
                    self.assertTrue(name.endswith(f".shard_{part}.weight"))
                    self.assertNotIsInstance(fragment, DTensor)
                    part_begin = max(begin, rows_per_part * part)
                    part_end = min(end, rows_per_part * (part + 1))
                    self.assertEqual(fragment.global_shape, (rows_per_part, 6))
                    self.assertEqual(
                        fragment.global_offsets,
                        ((part_begin - rows_per_part * part, feature_begin),),
                    )
                    self.assertEqual(
                        fragment,
                        full[part_begin:part_end, feature_begin:feature_end].bfloat16(),
                        atol=0,
                        rtol=0,
                    )
                checkpoint = f"{path[0]}/{column_shards}-{num_rows}-{hsdp}"
                dcp.save(
                    hf,
                    storage_writer=dcp.HuggingFaceStorageWriter(
                        checkpoint, save_distributed=True, enable_consolidation=True
                    ),
                )
                destination = adapter.to_hf(
                    {
                        key: torch.empty_like(exported[key]),
                        "lm_head.weight": torch.empty_like(exported["lm_head.weight"]),
                    }
                )
                dcp.load(
                    destination,
                    storage_reader=adapter.get_hf_storage_reader(checkpoint),
                )
                restored = adapter.from_hf(destination)
                self.assertEqual(
                    restored[key].to_local(grad_placements=placements),
                    expected,
                    atol=0,
                    rtol=0,
                )
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path[0])

    @with_comms
    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_storage_mesh_and_initialization_match_row_owners(self):
        """DTensor storage must agree with SPMD allocation and owner-seeded initialization."""
        for num_replicas, num_rows in ((1, 8), (2, 8), (4, 8), (1, 5), (2, 5)):
            pc = ParallelismContext(
                dp_replicate=num_replicas,
                dp_shard=self.world_size // num_replicas,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=self.world_size,
                enable_sequence_parallel=False,
            )
            config = NGramTable.Config(
                num_embeddings=num_rows,
                embedding_dim=3,
                param_init={
                    "weight": functools.partial(torch.nn.init.normal_, std=0.02)
                },
                sharding_config=ngram_table_sharding(),
            )
            with torch.device("meta"):
                table = config.build()
            if num_rows == 5:
                with self.assertRaisesRegex(
                    ValueError, "PLE requires num_embeddings divisible"
                ):
                    table._parallelize(pc)
                continue
            table._parallelize(pc)
            table.to_empty(device="cpu")
            torch.manual_seed(42)
            table.init_states()
            self.assertIsInstance(table.weight, DTensor)
            placements = table.weight.placements
            local = table.weight.to_local(grad_placements=placements)
            with torch.random.fork_rng():
                torch.manual_seed(42 + table.row_mesh.get_local_rank())
                expected = torch.empty_like(local).normal_(std=0.02)
            self.assertEqual(local, expected, atol=0, rtol=0)
            (chunk,) = table.weight.__create_chunk_list__()
            rows_per_owner = (num_rows + self.world_size - 1) // self.world_size
            self.assertEqual(
                chunk.offsets[0], min(self.rank * rows_per_owner, num_rows)
            )
            self.assertEqual(tuple(chunk.sizes), tuple(local.shape))
