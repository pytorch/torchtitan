# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp


class TestAsyncTP(unittest.TestCase):
    def test_get_dense_tp_mesh_uses_backend_mesh(self) -> None:
        tp_mesh = MagicMock()
        for spmd_backend in ("default", "full_dtensor", "spmd_types"):
            with self.subTest(spmd_backend=spmd_backend):
                parallel_dims = ParallelDims(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=2,
                    pp=1,
                    ep=1,
                    world_size=2,
                    spmd_backend=spmd_backend,
                )
                parent_mesh = MagicMock()
                parent_mesh.__getitem__.return_value = tp_mesh

                if spmd_backend == "default":
                    with patch.object(
                        parallel_dims, "get_mesh", return_value=tp_mesh
                    ) as get_mesh:
                        actual = parallel_dims.get_dense_tp_mesh()
                    get_mesh.assert_called_once_with("tp")
                elif spmd_backend == "full_dtensor":
                    with patch.object(
                        parallel_dims, "spmd_meshes", return_value=[parent_mesh]
                    ):
                        actual = parallel_dims.get_dense_tp_mesh()
                else:
                    with patch.object(
                        parallel_dims, "spmd_dense_mesh", return_value=parent_mesh
                    ):
                        actual = parallel_dims.get_dense_tp_mesh()

                self.assertIs(actual, tp_mesh)

    def test_uses_backend_tp_group(self) -> None:
        tp_group = SimpleNamespace(group_name="tp_group")
        tp_mesh = MagicMock()
        tp_mesh.get_group.return_value = tp_group
        parallel_dims = MagicMock()
        parallel_dims.get_dense_tp_mesh.return_value = tp_mesh

        parallelism = ParallelismConfig(enable_async_tensor_parallel=True)
        compile_config = CompileConfig(enable=True, components=["model"])
        original_micro_pipeline_tp = torch._inductor.config._micro_pipeline_tp
        try:
            with patch(
                "torch.distributed._symmetric_memory.enable_symm_mem_for_group"
            ) as enable_symm_mem:
                maybe_enable_async_tp(parallelism, compile_config, parallel_dims)
        finally:
            torch._inductor.config._micro_pipeline_tp = original_micro_pipeline_tp

        parallel_dims.get_dense_tp_mesh.assert_called_once_with()
        enable_symm_mem.assert_called_once_with("tp_group")


if __name__ == "__main__":
    unittest.main()
