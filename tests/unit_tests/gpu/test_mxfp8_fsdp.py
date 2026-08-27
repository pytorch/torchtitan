# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

import torchtitan.components.quantization.mxfp8.tensor as mxfp8_tensor  # noqa: E402
from torchtitan.components.quantization._fsdp_tensor import (  # noqa: E402
    _UnshardedFSDPTensor,
)
from torchtitan.components.quantization.mxfp8.grouped_experts import (  # noqa: E402
    get_mxfp8_grouped_experts_cls,
)
from torchtitan.components.quantization.mxfp8.linear import MXFP8Linear  # noqa: E402
from torchtitan.components.quantization.mxfp8.tensor import (  # noqa: E402
    _GroupedExpertsShardedTensorWithMXFP8Compute,
    _LinearShardedTensorWithMXFP8Compute,
)
from torchtitan.distributed.cudagraph import (  # noqa: E402
    cudagraph_teardown,
    CUDAGraphWrapper,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (  # noqa: E402
    data_parallel,
    disable_active_parametrization,
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
)
from torchtitan.models.common.moe import GroupedExperts  # noqa: E402


# Every test here spawns a two-rank process group, so the whole module belongs
# to the multi_gpu lane. Without the marker the tests land in the single-GPU
# lane instead, where the device-count guard skips all of them.
pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]


def _get_weight_param(linear):
    state = fully_shard.state(linear)
    param_group = state._fsdp_param_group
    assert param_group is not None
    return next(
        param
        for param in param_group.fsdp_params
        if param._module_info.param_name == "weight"
    )


def _run_reshard_after_forward(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test RAF=true release and refill of FSDP-managed MXFP8 operands.

    Forward and backward use separate unshards. Each reshard must release both
    the temporary BF16 all-gather output and the MXFP8 operand storage, while a
    later unshard must refill the same stable inner tensor objects.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        linear.compile()
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=True,
        )
        assert isinstance(
            linear.weight.to_local(), _LinearShardedTensorWithMXFP8Compute
        )

        input_MK = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output_MN = linear(input_MK)
        weight_param = _get_weight_param(linear)
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert isinstance(
            linear.weight.to_local(), _LinearShardedTensorWithMXFP8Compute
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        output_MN.sum().backward()
        assert isinstance(
            linear.weight.to_local(), _LinearShardedTensorWithMXFP8Compute
        )
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )
    finally:
        dist.destroy_process_group()


def _run_pp_cache_lifecycle(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test RAF=false cache reuse across pipeline-parallel microbatches.

    The first unshard quantizes the weight once. Multiple forwards and
    backwards reuse those MXFP8 operands until the last backward requests
    a reshard. The next generation must quantize again while reusing the same
    managed tensor objects.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize_weight = mxfp8_tensor._quantize_mxfp8_weight
    num_quantize_calls = 0

    def counted_quantize_weight(weight_NK: torch.Tensor):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize_weight(weight_NK)

    mxfp8_tensor._quantize_mxfp8_weight = counted_quantize_weight
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=False,
        )
        linear.set_is_last_backward(False)
        linear.set_reshard_after_backward(False)
        linear.set_requires_gradient_sync(False)

        inputs = [
            torch.randn(
                64,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(2)
        ]
        outputs = [linear(input_MK) for input_MK in inputs]
        assert num_quantize_calls == 1
        weight_param = _get_weight_param(linear)
        assert isinstance(linear.weight, _UnshardedFSDPTensor)
        assert linear.weight.operands is not None
        assert len(weight_param._unsharded_inner_tensors) == 3
        operands = linear.weight.operands
        assert operands is not None
        assert (
            operands.weight_qdata_dgrad_NK.data_ptr()
            == operands.weight_qdata_fprop_KN.data_ptr()
        )
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        outputs[0].sum().backward()
        assert num_quantize_calls == 1
        assert isinstance(linear.weight, _UnshardedFSDPTensor)
        assert linear.weight.operands is not None
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        linear.set_is_last_backward(True)
        linear.set_reshard_after_backward(True)
        linear.set_requires_gradient_sync(True)
        outputs[1].sum().backward()
        assert num_quantize_calls == 1
        assert isinstance(
            linear.weight.to_local(), _LinearShardedTensorWithMXFP8Compute
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        output_MN = linear(inputs[0].detach())
        assert num_quantize_calls == 2
        assert isinstance(linear.weight, _UnshardedFSDPTensor)
        assert linear.weight.operands is not None
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )
        output_MN.sum().backward()
    finally:
        mxfp8_tensor._quantize_mxfp8_weight = original_quantize_weight
        dist.destroy_process_group()


def _run_cuda_graph_cache_lifecycle(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test that the RAF=false MXFP8 cache is safe for CUDA graph replay.

    Warmup, capture, and replay must not re-quantize the weight or change the
    cached operand addresses. An explicit reshard after graph teardown must
    release the FSDP-managed operand storage.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize_weight = mxfp8_tensor._quantize_mxfp8_weight
    num_quantize_calls = 0

    def counted_quantize_weight(weight_NK: torch.Tensor):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize_weight(weight_NK)

    mxfp8_tensor._quantize_mxfp8_weight = counted_quantize_weight
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=False,
        )
        linear.set_is_last_backward(False)
        linear.set_reshard_after_backward(False)
        linear.set_requires_gradient_sync(False)

        def forward_backward(
            input_MK: torch.Tensor,
        ) -> torch.Tensor:
            output_MN = linear(input_MK)
            output_MN.sum().backward()
            return output_MN

        input_MK = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Establish the FSDP unsharded generation and its prepared weights on
        # the current stream before CUDA-graph warmup moves to its side stream.
        forward_backward(input_MK)
        torch.cuda.synchronize()
        assert num_quantize_calls == 1
        weight_param = _get_weight_param(linear)
        cache_addresses = tuple(
            tensor.data_ptr() for tensor in weight_param._unsharded_inner_tensors
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        graphed_step = CUDAGraphWrapper(
            forward_backward,
            (input_MK,),
            static_input_indices=(0,),
            should_check_address=True,
        )

        # RAF=false keeps the prepared weights alive, so CUDA-graph warmup,
        # capture, and replay reuse the same tensor objects and addresses.
        graphed_step(input_MK)
        assert num_quantize_calls == 1

        captured_output_MN = graphed_step(input_MK).clone()
        with torch.no_grad():
            input_MK.copy_(torch.randn_like(input_MK))
        replay_output_MN = graphed_step(input_MK).clone()
        torch.cuda.synchronize()

        assert graphed_step._graph is not None
        assert num_quantize_calls == 1
        assert (
            tuple(tensor.data_ptr() for tensor in weight_param._unsharded_inner_tensors)
            == cache_addresses
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert not torch.equal(captured_output_MN, replay_output_MN)

        graphed_step.teardown()
        linear.reshard()
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )
    finally:
        mxfp8_tensor._quantize_mxfp8_weight = original_quantize_weight
        cudagraph_teardown()
        dist.destroy_process_group()


def _run_simple_fsdp(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test GraphTrainer SimpleFSDP unsharded tensors and gradient propagation."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize_weight = mxfp8_tensor._quantize_mxfp8_weight
    num_quantize_calls = 0

    def counted_quantize_weight(weight_NK: torch.Tensor):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize_weight(weight_NK)

    mxfp8_tensor._quantize_mxfp8_weight = counted_quantize_weight
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        linear = data_parallel(
            linear,
            mesh,
            mode="fully_shard",
            mp_policy=SimpleFSDPMixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            # apply_simple_fsdp() composes this for real GraphTrainer runs.
        )
        sharded_weight = linear._parameters["weight"]
        assert isinstance(
            sharded_weight._local_tensor, _LinearShardedTensorWithMXFP8Compute
        )

        input_MK = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output_MN = linear(input_MK)
        output_MN.sum().backward()

        assert output_MN.shape == (64, 128)
        assert num_quantize_calls == 1
        assert input_MK.grad is not None
        assert sharded_weight.grad is not None
    finally:
        mxfp8_tensor._quantize_mxfp8_weight = original_quantize_weight
        dist.destroy_process_group()


def _make_grouped_experts(num_experts: int, dim: int, hidden_dim: int):
    experts_cls = get_mxfp8_grouped_experts_cls(GroupedExperts)
    experts = (
        experts_cls.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )
        .build()
        .cuda()
        .bfloat16()
    )
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    return experts


def _get_grouped_weight_param(experts, param_name: str):
    state = fully_shard.state(experts)
    param_group = state._fsdp_param_group
    assert param_group is not None
    return next(
        param
        for param in param_group.fsdp_params
        if param._module_info.param_name == param_name
    )


def _run_grouped_experts_reshard_after_forward(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test RAF=true release and refill of FSDP-managed grouped MXFP8 operands.

    Grouped experts manage four inner tensors per weight rather than the dense
    linear's three, because FPROP and DGRAD need separate qdata layouts. Both
    the temporary BF16 all-gather output and all four operands must be released
    on reshard, and a later unshard must refill the same objects.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        num_experts, dim, hidden_dim = 4, 128, 256
        experts = _make_grouped_experts(num_experts, dim, hidden_dim)
        fully_shard(
            experts,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=True,
        )
        assert isinstance(
            experts.w1_EFD.to_local(), _GroupedExpertsShardedTensorWithMXFP8Compute
        )

        tokens_per_expert = 128
        x_RD = torch.randn(
            num_experts * tokens_per_expert,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        num_tokens_per_expert_E = torch.full(
            (num_experts,), tokens_per_expert, device="cuda", dtype=torch.int32
        )

        out_RD = experts(x_RD, num_tokens_per_expert_E)
        weight_param = _get_grouped_weight_param(experts, "w1_EFD")
        # One shared qdata is impossible here, so FSDP owns two qdata tensors
        # and two blocked scale tensors.
        assert len(weight_param._unsharded_inner_tensors) == 4
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        out_RD.sum().backward()
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )
        assert x_RD.grad is not None
        for parameter in experts.parameters():
            assert parameter.grad is not None
    finally:
        dist.destroy_process_group()


def _run_grouped_experts_pp_cache_lifecycle(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test RAF=false grouped-expert cache reuse across microbatches.

    The first unshard quantizes each expert weight once and every microbatch
    reuses those operands until the last backward reshards. The next
    generation re-quantizes into the same managed tensor objects.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize = mxfp8_tensor._quantize_mxfp8_grouped_weight
    num_quantize_calls = 0

    def counted_quantize(weight_ENK: torch.Tensor):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize(weight_ENK)

    mxfp8_tensor._quantize_mxfp8_grouped_weight = counted_quantize
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        num_experts, dim, hidden_dim = 4, 128, 256
        experts = _make_grouped_experts(num_experts, dim, hidden_dim)
        fully_shard(
            experts,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=False,
        )
        experts.set_is_last_backward(False)
        experts.set_reshard_after_backward(False)
        experts.set_requires_gradient_sync(False)

        tokens_per_expert = 128
        num_tokens_per_expert_E = torch.full(
            (num_experts,), tokens_per_expert, device="cuda", dtype=torch.int32
        )
        inputs = [
            torch.randn(
                num_experts * tokens_per_expert,
                dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(2)
        ]
        outputs = [experts(x_RD, num_tokens_per_expert_E) for x_RD in inputs]
        # Three expert weights, quantized once each on the first unshard.
        assert num_quantize_calls == 3
        weight_param = _get_grouped_weight_param(experts, "w1_EFD")
        assert isinstance(experts.w1_EFD, _UnshardedFSDPTensor)
        operands = experts.w1_EFD.operands
        assert operands is not None
        # The two qdata layouts hold identical values in distinct storages.
        assert (
            operands.weight_qdata_fprop_EKN.untyped_storage().data_ptr()
            != operands.weight_qdata_dgrad_ENK.untyped_storage().data_ptr()
        )
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        outputs[0].sum().backward()
        assert num_quantize_calls == 3
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        experts.set_is_last_backward(True)
        experts.set_reshard_after_backward(True)
        experts.set_requires_gradient_sync(True)
        outputs[1].sum().backward()
        assert num_quantize_calls == 3
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        out_RD = experts(inputs[0].detach(), num_tokens_per_expert_E)
        assert num_quantize_calls == 6
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )
        out_RD.sum().backward()
    finally:
        mxfp8_tensor._quantize_mxfp8_grouped_weight = original_quantize
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "target",
    [
        _run_reshard_after_forward,
        _run_pp_cache_lifecycle,
        _run_cuda_graph_cache_lifecycle,
        _run_simple_fsdp,
        _run_grouped_experts_reshard_after_forward,
        _run_grouped_experts_pp_cache_lifecycle,
    ],
    ids=[
        "reshard-after-forward",
        "pp-cache",
        "cuda-graph-cache",
        "simple-fsdp",
        "grouped-experts-reshard-after-forward",
        "grouped-experts-pp-cache",
    ],
)
def test_mxfp8_fsdp_tensor_lifecycle(target):
    mp.spawn(
        target,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )


def _run_simple_fsdp_disabled_parametrization(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Test that disable_active_parametrization() yields the raw parameter.

    Models call it around ``init_states()`` to inspect and initialize weights.
    Building an unsharded tensor there would quantize the still-sharded shard as
    if it were the logical weight, so the disable has to cover that step too.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        linear = (
            MXFP8Linear.Config(in_features=128, out_features=128, bias=False)
            .build()
            .cuda()
            .bfloat16()
        )
        linear = data_parallel(
            linear,
            mesh,
            mode="fully_shard",
            mp_policy=SimpleFSDPMixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
        )

        # Reading the parametrized weight all-gathers, so every rank has to
        # reach both of these.
        active_weight = linear.weight
        with disable_active_parametrization():
            disabled_weight = linear.weight

        assert isinstance(active_weight, _UnshardedFSDPTensor)
        assert isinstance(disabled_weight, DTensor)
        assert isinstance(
            disabled_weight._local_tensor, _LinearShardedTensorWithMXFP8Compute
        )
        assert not isinstance(disabled_weight._local_tensor, _UnshardedFSDPTensor)
    finally:
        dist.destroy_process_group()


def test_simple_fsdp_disable_active_parametrization():
    mp.spawn(
        _run_simple_fsdp_disabled_parametrization,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )


def _run_grouped_experts_uneven_shard_dim0(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Expert count that does not divide the FSDP degree.

    FSDP pads the last shard, so ``fsdp_pre_all_gather`` returns the padded
    shard and carries the logical size in metadata; ``fsdp_post_all_gather``
    narrows the padding away before quantizing. Were the padding to reach the
    quantizer it would occupy real 32x32 scale tiles and show up as extra
    experts, so this checks the gradients too rather than just completion.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        # 1 expert over 2 ranks: dim 0 does not divide, so FSDP pads. The
        # expert count has to stay a power of two because TorchAO's scale
        # rearrange kernel does tl.arange over the token groups.
        num_experts, dim, hidden_dim = 1, 128, 256
        experts = _make_grouped_experts(num_experts, dim, hidden_dim)
        fully_shard(
            experts,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
            reshard_after_forward=True,
        )
        tokens_per_expert = 128
        x_RD = torch.randn(
            num_experts * tokens_per_expert,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        counts = torch.full(
            (num_experts,), tokens_per_expert, device="cuda", dtype=torch.int32
        )
        experts(x_RD, counts).sum().backward()

        assert x_RD.grad is not None
        assert torch.isfinite(x_RD.grad).all()
        for name, parameter in experts.named_parameters():
            assert parameter.grad is not None, name
            assert parameter.grad.shape == parameter.shape, name
            assert torch.isfinite(parameter.grad.to_local()).all(), name
    finally:
        dist.destroy_process_group()


def _run_grouped_experts_shard_dim1(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """FSDP degree above the expert count, so torchtitan shards the hidden dim.

    ``apply_fsdp_to_decoder`` selects ``Shard(1)`` when ``efsdp * ep`` exceeds
    the expert count, which any job with more ranks than experts hits. The
    all-gather then concatenates dim-1 shards along dim 0, so the hook has to
    rebuild the logical weight before quantizing. It assumes dim 0 instead.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        num_experts, dim, hidden_dim = 1, 128, 256
        experts = _make_grouped_experts(num_experts, dim, hidden_dim)
        fully_shard(
            experts,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
            shard_placement_fn=lambda _param: Shard(1),
            reshard_after_forward=True,
        )
        tokens_per_expert = 256
        x_RD = torch.randn(
            num_experts * tokens_per_expert,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        counts = torch.full(
            (num_experts,), tokens_per_expert, device="cuda", dtype=torch.int32
        )
        experts(x_RD, counts).sum().backward()
    finally:
        dist.destroy_process_group()


def test_mxfp8_grouped_experts_uneven_shard_dim0():
    """An expert count that does not divide the FSDP degree still trains."""
    mp.spawn(
        _run_grouped_experts_uneven_shard_dim0,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.xfail(
    strict=True,
    reason="unsharded tensors support sharding dimension 0 only",
)
def test_mxfp8_grouped_experts_shard_dim1():
    """Shard(1), which torchtitan picks when the FSDP degree exceeds experts.

    The all-gather concatenates dim-1 shards along dim 0, so the hook would
    have to rebuild the logical weight before quantizing. It rejects this
    explicitly instead; remove the xfail when it is supported.
    """
    mp.spawn(
        _run_grouped_experts_shard_dim1, args=(2, get_free_port()), nprocs=2, join=True
    )
