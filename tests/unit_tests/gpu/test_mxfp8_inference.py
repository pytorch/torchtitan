# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, DTensor, Shard


pytest.importorskip("torchao")

import torchtitan.quantization.mxfp8.inference as inference  # noqa: E402
from torchtitan.distributed.fsdp import resolve_fsdp_mesh  # noqa: E402
from torchtitan.distributed.parallel_dims import ParallelDims  # noqa: E402
from torchtitan.distributed.utils import get_spmd_context  # noqa: E402
from torchtitan.models.common.decoder_sharding import (  # noqa: E402
    dense_param_placement,
)
from torchtitan.protocols.sharding import ShardingConfig  # noqa: E402
from torchtitan.quantization.mxfp8.linear import _MXFP8LinearFunction  # noqa: E402
from torchtitan.quantization.mxfp8.tensor import _MXFP8LinearOperands  # noqa: E402


pytestmark = [pytest.mark.multi_gpu]


def _reference_quantize(weight):
    """CPU/Hopper oracle for transport tests; real MXFP8 tests use TorchAO."""
    rows, cols = weight.shape
    tiles = weight.float().view(rows // 32, 32, cols // 32, 32)
    amax = tiles.abs().amax(dim=(1, 3))
    exponent = (amax / 448).log2().ceil().clamp(-127, 127)
    scale = torch.exp2(exponent)
    data = (tiles / scale[:, None, :, None]).view(rows, cols).to(torch.float8_e4m3fn)
    return _MXFP8LinearOperands(
        data,
        inference._swizzle_scale(
            scale.repeat_interleave(32, 0).to(torch.float8_e8m0fnu)
        ),
        inference._swizzle_scale(
            scale.t().repeat_interleave(32, 0).to(torch.float8_e8m0fnu)
        ),
    )


def _dequantize(operands):
    data = operands.weight_qdata_dgrad_NK
    scale = inference._unswizzle_scale(
        operands.weight_scale_fprop_swizzled, data.shape[0], data.shape[1] // 32
    )
    return (
        (data.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1))
        .flatten(-2)
        .bfloat16()
    )


class _TransportLinear(inference.MXFP8InferenceLinear):
    """Exercise the storage lifecycle on GPUs without MXFP8 matrix hardware."""

    @dataclass(kw_only=True, slots=True)
    class Config(inference.MXFP8InferenceLinear.Config):
        pass

    def forward(self, input):
        return self._unflatten_output(
            torch.nn.functional.linear(
                input,
                _dequantize(self.weight.operands),
                None if self.bias is None else self.bias.flatten(),
            )
        )


def _run_weight_updates(
    rank, world_size, port, num_linears, tp_degree, tp_dim, real_mxfp8
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    original_quantize = inference._quantize_mxfp8_weight
    if not real_mxfp8:
        inference._quantize_mxfp8_weight = _reference_quantize
    quantize = inference._quantize_mxfp8_weight
    try:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=world_size // tp_degree,
            cp=1,
            tp=tp_degree,
            pp=1,
            ep=1,
            world_size=world_size,
            enable_sequence_parallel=False,
        )
        fsdp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        cls = inference.MXFP8InferenceLinear if real_mxfp8 else _TransportLinear
        with torch.device("meta"):
            linear = (
                cls.Config(
                    in_features=128,
                    out_features=256,
                    num_linears=num_linears,
                    bias=num_linears > 1,
                    sharding_config=ShardingConfig(
                        state_shardings={
                            "weight": dense_param_placement(tp=spmd.S(tp_dim)),
                            "bias": dense_param_placement(tp=spmd.S(-1)),
                        }
                    ),
                )
                .build()
                .bfloat16()
            )
        with get_spmd_context(parallel_dims=parallel_dims):
            linear._parallelize(parallel_dims)
            fully_shard(
                linear,
                mesh=fsdp_mesh,
                dp_mesh_dims=dp_mesh_dims,
                shard_placement_fn=lambda _: Shard(0 if num_linears == 1 else 1),
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                reshard_after_forward=False,
            )
            linear.to_empty(device="cuda")
            with torch.no_grad():
                linear.init_states()

        sharded = linear.weight
        assert isinstance(sharded, DTensor)
        local = sharded.to_local()
        assert isinstance(local, inference._MXFP8StorageTensor)
        assert local._qdata.dtype == torch.float8_e4m3fn
        assert local._scale.dtype == torch.float8_e8m0fnu
        tp_rank = parallel_dims.get_mesh("tp").get_local_rank() if tp_degree > 1 else 0
        inputs = torch.randn(
            32,
            128 // (tp_degree if tp_dim == -1 else 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        # vLLM captures with temporary initial weights, before the first pull.
        with torch.inference_mode(), get_spmd_context(parallel_dims=parallel_dims):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    linear(inputs)
            torch.cuda.current_stream().wait_stream(stream)
            param_group = fully_shard.state(linear)._fsdp_param_group
            param = param_group.fsdp_params[0]
            buffers = param._unsharded_inner_tensors
            pointers = tuple(t.data_ptr() for t in buffers)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = linear(inputs)
        previous_output = None
        for update in range(3):
            with torch.no_grad(), get_spmd_context(parallel_dims=parallel_dims):
                linear.reshard(free_unsharded=False, force=True)
                state_dict = linear.state_dict()
                destination = state_dict["weight"]
                assert isinstance(destination, DTensor)
                assert type(destination.to_local()) is torch.Tensor
                assert destination.dtype == torch.bfloat16
                assert destination.placements == sharded.placements
                assert destination.device_mesh == sharded.device_mesh
                full_weight = torch.randn(
                    sharded.shape, device="cuda", dtype=torch.bfloat16
                ) * (update + 1)
                # Emulate TorchStore filling the discovered BF16 destination layout.
                destination.copy_(
                    distribute_tensor(
                        full_weight, destination.device_mesh, destination.placements
                    )
                )
                if "bias" in state_dict:
                    bias_destination = state_dict["bias"]
                    bias_destination.copy_(
                        distribute_tensor(
                            full_weight.mean(-1),
                            bias_destination.device_mesh,
                            bias_destination.placements,
                        )
                    )
                linear.load_state_dict(state_dict)
                del destination, state_dict

                tp_weight = full_weight.chunk(tp_degree, dim=tp_dim)[
                    tp_rank
                ].contiguous()
                expected_operands = quantize(tp_weight.flatten(0, -2))
                with torch.inference_mode():
                    linear.unshard()
                    current_pointers = tuple(t.data_ptr() for t in buffers)
                    assert current_pointers == pointers
                    for actual, expected in zip(
                        buffers,
                        (
                            expected_operands.weight_qdata_dgrad_NK,
                            expected_operands.weight_scale_fprop_swizzled,
                            expected_operands.weight_scale_dgrad_swizzled,
                        ),
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            actual.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )

                    if real_mxfp8:
                        expected = _MXFP8LinearFunction.apply(
                            inputs,
                            tp_weight.flatten(0, -2),
                            expected_operands.weight_qdata_fprop_KN,
                            expected_operands.weight_scale_fprop_swizzled,
                            expected_operands.weight_qdata_dgrad_NK,
                            expected_operands.weight_scale_dgrad_swizzled,
                            None if linear.bias is None else linear.bias.flatten(),
                            "bf16",
                        )
                    else:
                        expected = torch.nn.functional.linear(
                            inputs,
                            _dequantize(expected_operands),
                            None if linear.bias is None else linear.bias.flatten(),
                        )
                    expected = linear._unflatten_output(expected)
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
                    if previous_output is not None:
                        assert not torch.equal(captured, previous_output)
                    previous_output = captured.clone()
                    assert all(
                        t.untyped_storage().size() == 0
                        for t in param.all_gather_outputs
                    )

        with torch.inference_mode():
            linear.reshard(free_unsharded=False, force=True)
            assert [t.dtype for t in param.all_gather_inputs] == [
                torch.uint8,
            ]
        graph.reset()
    finally:
        inference._quantize_mxfp8_weight = original_quantize
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "num_linears,dp_degree,tp_degree,tp_dim",
    [(1, 2, 1, -2), (2, 2, 1, -2), (2, 2, 2, -2), (1, 2, 2, -1), (2, 1, 2, -2)],
)
def test_mxfp8_storage_collectives_and_graph_updates(
    num_linears, dp_degree, tp_degree, tp_dim
):
    world_size = dp_degree * tp_degree
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    mp.spawn(
        _run_weight_updates,
        args=(world_size, get_free_port(), num_linears, tp_degree, tp_dim, False),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.cuda.get_device_capability() < (10, 0),
    reason="MXFP8 GEMM requires two SM100 GPUs",
)
@pytest.mark.parametrize("num_linears", [1, 2])
def test_mxfp8_inference_matches_bf16_storage(num_linears):
    mp.spawn(
        _run_weight_updates,
        args=(2, get_free_port(), num_linears, 1, -2, True),
        nprocs=2,
        join=True,
    )
