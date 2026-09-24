# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.utils import is_sm_at_least_90

from pytorch.flex_shard import BucketSpec, flex_shard
from pytorch.flex_shard.custom_placements.fp8_bucketed_block_shard import (
    _align_up,
    _quantize_dense_weight_to_blockwise_fp8,
    _VEC_ALIGN_BYTES,
    BlockwiseFp8WeightFactory,
    Fp8BucketedBlockShard,
    make_fp8_bucketed_block_placement_fn,
)
from pytorch.flex_shard.custom_placements.shard import per_param_placements
from pytorch.flex_shard.custom_placements.utils import foreach_copy_
from pytorch.flex_shard.flex_shard.bucket_storage import ParamInfo, PlacementFn
from pytorch.flex_shard.flex_shard.placement_contract import (
    Placement,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from pytorch.flex_shard.flex_shard.unshard_op import mark_unshard_bucket
from pytorch.flex_shard.flex_shard.utils import (
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)
from pytorch.flex_shard.tests.common import make_test_sgd, make_transformer_model
from torchtitan.quantization.blockwise_fp8 import (
    BlockwiseFp8Weight,
    convert_to_flex_shard_float8_blockwise_linear,
)


device_type = torch.device(get_devtype())


def _make_dense_all_gather_reference_placement_fn(
    *,
    weight_factory: BlockwiseFp8WeightFactory,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    def dense_all_gather_reference_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = _DenseAllGatherReferencePlacement(
            world_size=mesh.size(),
            weight_factory=weight_factory,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return dense_all_gather_reference_placements


class _DenseAllGatherReferencePlacement(Fp8BucketedBlockShard):
    """All-gather dense shards using the FP8 placement's shard layout."""

    def _dense_send_numel_per_rank(
        self,
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata,
    ) -> int:
        return _align_up(
            max(metadata.rank_fp8_numels, default=0),
            _VEC_ALIGN_BYTES,
        )

    def _pack_local_dense_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata,
    ) -> torch.Tensor:
        dtype = infos[0].unsharded_dtype
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        local_dense = torch.empty(
            self._dense_send_numel_per_rank(metadata),
            dtype=dtype,
            device=device,
        )
        payload_end = metadata.rank_fp8_numels[rank]
        if payload_end < local_dense.numel():
            local_dense[payload_end:].zero_()
        tensors_by_fqn = {
            info.fqn: tensor.reshape(info.local_shape)
            for tensor, info in zip(tensors, infos, strict=True)
        }
        infos_by_fqn = {info.fqn: info for info in infos}
        for chunk in metadata.chunks:
            if chunk.rank != rank:
                continue
            info = infos_by_fqn[chunk.fqn]
            tensor = tensors_by_fqn[chunk.fqn]
            expected_numel = self._chunk_numel(chunk, info.global_shape[1])
            if tensor.numel() != expected_numel:
                raise AssertionError(
                    "Dense reference local shard does not match planned chunk for "
                    f"{chunk.fqn!r}: expected {expected_numel} elements, got "
                    f"{tensor.numel()}."
                )
            local_dense[chunk.fp8_offset : chunk.fp8_offset + tensor.numel()].copy_(
                tensor.reshape(-1)
            )
        return local_dense

    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        metadata = self._fp8_metadata_from_infos(infos)
        with _record_copy_in_if_eager():
            local_dense = self._pack_local_dense_chunks(
                tensors,
                infos,
                rank,
                metadata,
            )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_dense],
            placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        expected_shape = (
            len(state.metadata.rank_fp8_numels),
            self._dense_send_numel_per_rank(state.metadata),
        )
        if tuple(rank_rows.shape) != expected_shape:
            raise ValueError(
                f"Expected gathered rank rows with shape {expected_shape}, "
                f"got {tuple(rank_rows.shape)}."
            )

        infos_by_fqn = {info.fqn: info for info in state.infos}
        dense_by_fqn = {
            info.fqn: torch.empty(
                info.global_shape,
                dtype=rank_rows.dtype,
                device=rank_rows.device,
            )
            for info in state.infos
        }
        copy_dsts: list[torch.Tensor] = []
        copy_srcs: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for chunk in state.metadata.chunks:
                info = infos_by_fqn[chunk.fqn]
                in_dim = info.global_shape[1]
                numel = self._chunk_numel(chunk, in_dim)
                copy_dsts.append(
                    dense_by_fqn[chunk.fqn][chunk.row_start : chunk.row_end, :]
                )
                copy_srcs.append(
                    rank_rows[
                        chunk.rank,
                        chunk.fp8_offset : chunk.fp8_offset + numel,
                    ].view(chunk.row_end - chunk.row_start, in_dim)
                )
            foreach_copy_(copy_dsts, copy_srcs)
        full_params = []
        for info in state.infos:
            dense_weight = dense_by_fqn[info.fqn]
            dense_weight.requires_grad_(info.requires_grad)
            fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
                dense_weight,
                self.block_size,
                self.fp8_dtype,
            )
            full_params.append(
                self._make_blockwise_fp8_weight(
                    fp8_data,
                    recip_scale,
                    orig_dtype=dense_weight.dtype,
                    requires_grad=dense_weight.requires_grad,
                )
            )
        return PlacementUnshardResult(full_params=full_params)


def _make_blockwise_fp8_weight() -> BlockwiseFp8Weight:
    block_size = 4
    fp8_data = (
        torch.arange(128, dtype=torch.bfloat16).reshape(16, 8).to(torch.float8_e4m3fn)
    )
    recip_scale = torch.full((4, 2), 0.5, dtype=torch.float32)
    return BlockwiseFp8Weight(
        fp8_data,
        recip_scale,
        block_size,
        orig_dtype=torch.bfloat16,
        requires_grad=True,
    )


class TestBlockwiseFp8Weight(TestCase):
    def test_blockwise_fp8_weight_protocol(self) -> None:
        wrapped = _make_blockwise_fp8_weight()

        self.assertEqual(wrapped.dtype, torch.bfloat16)
        self.assertTrue(wrapped.requires_grad)
        self.assertEqual(wrapped.block_size, 4)
        self.assertEqual(wrapped.orig_dtype, torch.bfloat16)
        self.assertEqual(
            wrapped.dequantize(torch.float32),
            wrapped.fp8_data.float() * 0.5,
        )

    def test_mark_unshard_bucket_preserves_blockwise_fp8_weight(self) -> None:
        wrapped = _make_blockwise_fp8_weight()

        (marked,) = mark_unshard_bucket([wrapped])

        self.assertIsInstance(marked, BlockwiseFp8Weight)
        self.assertIsNot(marked, wrapped)
        self.assertTrue(marked.requires_grad)
        self.assertEqual(marked.block_size, wrapped.block_size)
        self.assertEqual(marked.orig_dtype, wrapped.orig_dtype)
        self.assertEqual(
            marked.fp8_data.untyped_storage().data_ptr(),
            wrapped.fp8_data.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            marked.recip_scale.untyped_storage().data_ptr(),
            wrapped.recip_scale.untyped_storage().data_ptr(),
        )


@pytest.mark.multi_gpu
class TestBlockwiseFp8FlexShardTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    def _skip_unless_blockwise_linear_supported(self) -> None:
        if not is_sm_at_least_90():
            self.skipTest("Float8BlockwiseLinear requires CUDA SM90+")

    def _assert_sharded_linear_state_is_bf16(self, model: nn.Module) -> None:
        for storage in model.sharded_bucket_storages:
            for info in storage.param_infos.values():
                self.assertEqual(info.dtype, torch.bfloat16)
                self.assertEqual(
                    storage.get_local_view(info.fqn).dtype,
                    torch.bfloat16,
                )

    def _make_transformer_model_pair(self, *, mesh):
        block_size = 128
        torch.manual_seed(0)
        model_args, fp8_model = make_transformer_model(
            device=device_type,
            n_layers=1,
            vocab_size=256,
            max_seq_len=8,
            dim=256,
            n_heads=4,
        )
        fp8_model.to(dtype=torch.bfloat16)
        bf16_model = copy.deepcopy(fp8_model)

        linear_weight_fqns = [
            f"{fqn}.weight"
            for fqn, module in fp8_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        linear_weight_fqn_set = set(linear_weight_fqns)
        other_param_fqns = [
            fqn
            for fqn, _ in fp8_model.named_parameters()
            if fqn not in linear_weight_fqn_set
        ]
        convert_to_flex_shard_float8_blockwise_linear(fp8_model, use_triton=False)
        convert_to_flex_shard_float8_blockwise_linear(bf16_model, use_triton=False)

        def make_buckets(placement_fn) -> list[BucketSpec]:
            return [
                BucketSpec(
                    linear_weight_fqns,
                    placement_fn=placement_fn,
                    mesh=mesh,
                    reshard_after_forward=True,
                ),
                BucketSpec(
                    other_param_fqns,
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=True,
                ),
            ]

        flex_shard(
            bf16_model,
            buckets=make_buckets(
                _make_dense_all_gather_reference_placement_fn(
                    weight_factory=BlockwiseFp8Weight,
                    block_size=block_size,
                )
            ),
        )
        flex_shard(
            fp8_model,
            buckets=make_buckets(
                make_fp8_bucketed_block_placement_fn(
                    weight_factory=BlockwiseFp8Weight,
                    block_size=block_size,
                )
            ),
        )
        return model_args, fp8_model, bf16_model

    def _assert_named_parameter_data_equal(
        self,
        fp8_model: nn.Module,
        bf16_model: nn.Module,
    ) -> None:
        for (fp8_name, fp8_param), (bf16_name, bf16_param) in zip(
            fp8_model.named_parameters(),
            bf16_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(fp8_name, bf16_name)
            self.assertTrue(torch.equal(fp8_param, bf16_param), fp8_name)

    def _assert_named_parameter_grads_equal(
        self,
        fp8_model: nn.Module,
        bf16_model: nn.Module,
    ) -> None:
        for (fp8_name, fp8_param), (bf16_name, bf16_param) in zip(
            fp8_model.named_parameters(),
            bf16_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(fp8_name, bf16_name)
            self.assertIsNotNone(fp8_param.grad)
            self.assertIsNotNone(bf16_param.grad)
            self.assertTrue(torch.equal(fp8_param.grad, bf16_param.grad), fp8_name)

    def _temporary_prepared_unshard_bytes(self, model: nn.Module, mesh) -> int:
        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        tensors = [storage.get_local_view(info.fqn) for info in infos]

        placement = infos[0].placement
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        return prepared.buffers[0].nbytes

    @skip_if_lt_x_gpu(2)
    def test_transformer_training_matches_bf16_allgather_then_quantize(self) -> None:
        """Five training steps match the BF16-all-gather FP8-GEMM reference."""
        self._skip_unless_blockwise_linear_supported()
        mesh = self._mesh()
        model_args, fp8_model, bf16_model = self._make_transformer_model_pair(
            mesh=mesh,
        )
        fp8_unshard_bytes = self._temporary_prepared_unshard_bytes(fp8_model, mesh)
        bf16_unshard_bytes = self._temporary_prepared_unshard_bytes(bf16_model, mesh)
        self.assertLess(fp8_unshard_bytes, bf16_unshard_bytes)
        self._assert_sharded_linear_state_is_bf16(fp8_model)
        self._assert_named_parameter_data_equal(fp8_model, bf16_model)

        fp8_optim = make_test_sgd(fp8_model.parameters(), lr=0.01)
        bf16_optim = make_test_sgd(bf16_model.parameters(), lr=0.01)

        for step in range(5):
            torch.manual_seed(3 + step)
            tokens = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            targets = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            dist.broadcast(tokens, src=0)
            dist.broadcast(targets, src=0)
            fp8_optim.zero_grad(set_to_none=True)
            bf16_optim.zero_grad(set_to_none=True)

            fp8_logits = fp8_model(tokens)
            bf16_logits = bf16_model(tokens)
            self.assertTrue(
                torch.equal(fp8_logits, bf16_logits),
                f"logits diverged at step {step}",
            )

            fp8_loss = nn.functional.cross_entropy(
                fp8_logits.flatten(0, 1),
                targets.flatten(),
            )
            bf16_loss = nn.functional.cross_entropy(
                bf16_logits.flatten(0, 1),
                targets.flatten(),
            )
            fp8_loss.backward()
            bf16_loss.backward()
            self._assert_named_parameter_grads_equal(fp8_model, bf16_model)

            fp8_optim.step()
            bf16_optim.step()
            self._assert_named_parameter_data_equal(fp8_model, bf16_model)


if __name__ == "__main__":
    run_tests()
