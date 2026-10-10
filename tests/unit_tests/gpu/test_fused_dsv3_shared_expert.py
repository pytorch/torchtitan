# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch


pytest.importorskip("torchao")

from torchtitan.quantization.mxfp8.tensor import _quantize_mxfp8_weight  # noqa: E402
from torchtitan_recipes.overrides._dsv3_shared_expert import autograd  # noqa: E402
from torchtitan_recipes.overrides._dsv3_shared_expert.kernels.backward import (  # noqa: E402
    launch_swiglu_backward,
)
from torchtitan_recipes.overrides._dsv3_shared_expert.kernels.quantize import (  # noqa: E402
    quantize_values,
)
from torchtitan_recipes.overrides._dsv3_shared_expert.ops import (  # noqa: E402
    shared_expert_backward_op,
    shared_expert_forward_op,
)
from torchtitan_recipes.overrides.fused_swiglu import (  # noqa: E402
    silu_and_mul_backward_op,
    silu_and_mul_op,
)

from tests.unit_tests.gpu.dsv3_shared_expert_utils import (  # noqa: E402
    assert_bits_equal,
    make_case,
    MODES,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="GB300 shared-expert specialization",
)


@pytest.mark.parametrize("seed", [42, 73, 919])
def test_primitive_bf16_boundaries_and_all_fp8_bytes(seed):
    torch.manual_seed(seed)
    x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(7168, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    dy = torch.randn_like(x)
    row, _, scale, _ = autograd.quantize(x)
    weight13 = _quantize_mxfp8_weight(w13)
    packed = autograd.scaled_mm(
        row, weight13.weight_qdata_fprop_KN, scale, weight13.weight_scale_fprop_swizzled
    )
    hidden = silu_and_mul_op(packed.unflatten(-1, (2, 2048)))
    expected_forward = (hidden, packed, *autograd.quantize(hidden))
    actual_forward = shared_expert_forward_op(
        row, weight13.weight_qdata_dgrad_NK, scale, weight13.weight_scale_fprop_swizzled
    )
    for expected, actual in zip(expected_forward, actual_forward):
        assert_bits_equal(expected, actual)
        assert expected.stride() == actual.stride()

    row, _, scale, _ = autograd.quantize(dy)
    weight2 = _quantize_mxfp8_weight(w2)
    d_hidden = autograd.scaled_mm(
        row, weight2.weight_qdata_dgrad_NK, scale, weight2.weight_scale_dgrad_swizzled
    )
    d_packed = silu_and_mul_backward_op(
        d_hidden, packed.unflatten(-1, (2, 2048))
    ).flatten(-2)
    expected_backward = (*autograd.quantize(d_packed), d_hidden, d_packed)
    actual_backward = launch_swiglu_backward(
        row,
        weight2.weight_qdata_dgrad_NK,
        scale,
        weight2.weight_scale_dgrad_swizzled,
        packed,
        save_intermediates=True,
    )
    for expected, actual in zip(expected_backward, actual_backward):
        assert_bits_equal(expected, actual)
        assert expected.stride() == actual.stride()
    for expected, actual in zip(
        expected_backward,
        shared_expert_backward_op(
            row,
            weight2.weight_qdata_dgrad_NK,
            scale,
            weight2.weight_scale_dgrad_swizzled,
            packed,
        ),
    ):
        assert_bits_equal(expected, actual)


@pytest.mark.parametrize("mode", list(MODES)[1:])
@pytest.mark.parametrize(
    "input_format,hidden_format", [("bf16", "mxfp8"), ("mxfp8", "bf16")]
)
def test_complete_ffn_output_and_gradients(mode, input_format, hidden_format):
    results = []
    for variant in ("native", mode):
        module, x, dy = make_case(
            42,
            variant,
            input_save_format=input_format,
            hidden_save_format=hidden_format,
        )
        with (
            patch.object(
                autograd, "shared_expert_forward_op", wraps=shared_expert_forward_op
            ) as forward,
            patch.object(
                autograd, "shared_expert_backward_op", wraps=shared_expert_backward_op
            ) as backward,
        ):
            output = module(x)
            gradients = torch.autograd.grad(
                output, (x, module.w13.weight, module.w2.weight), dy
            )
            assert (forward.call_count, backward.call_count) == tuple(
                map(int, MODES[variant])
            )
        results.append((output.detach(), *gradients))
    for expected, actual in zip(*results):
        assert_bits_equal(expected, actual)


@pytest.mark.parametrize(
    "case", ["all_bf16_patterns", "mixed_bf16_patterns", "zero", "tiny"]
)
def test_quantization_raw_patterns_include_nonfinite_and_signed_zero(case):
    if case in ("all_bf16_patterns", "mixed_bf16_patterns"):
        values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        if case == "mixed_bf16_patterns":
            torch.manual_seed(42)
            values = values[torch.randperm(65536, device="cuda")]
        values = values.view(torch.bfloat16).reshape(128, 512)
    elif case == "zero":
        values = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
        values[:, 1::2] = -0.0
    else:
        values = torch.arange(128 * 128, device="cuda", dtype=torch.int32).to(
            torch.int16
        )
        values = (values % 256).view(torch.bfloat16).reshape(128, 128)
    for expected, actual in zip(autograd.quantize(values), quantize_values(values)):
        assert_bits_equal(expected, actual)


@pytest.mark.parametrize(
    "requires_grad", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_frozen_operands(requires_grad):
    results = []
    for variant in ("native", "both_fused"):
        module, x, dy = make_case(73, variant, requires_grad=requires_grad)
        output = module(x)
        output.backward(dy)
        results.append(
            (output.detach(), x.grad, module.w13.weight.grad, module.w2.weight.grad)
        )
    for expected, actual in zip(*results):
        assert_bits_equal(expected, actual)


@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_preseeded_accumulation_retained_backward_and_physical_weight_grad_dtype(
    grad_dtype,
):
    results = []
    for variant in ("native", "both_fused"):
        module, x, dy = make_case(919, variant, grad_dtype=grad_dtype)
        pointers = []
        for parameter in (module.w13.weight, module.w2.weight):
            parameter.grad = torch.randn(
                parameter.shape, device="cuda", dtype=grad_dtype
            )
            pointers.append(parameter.grad.data_ptr())
        output = module(x)
        output.backward(dy, retain_graph=True)
        output.backward(dy)
        for parameter, pointer in zip((module.w13.weight, module.w2.weight), pointers):
            assert parameter.grad.dtype == grad_dtype
            assert parameter.grad.data_ptr() == pointer
        results.append((x.grad, module.w13.weight.grad, module.w2.weight.grad))
    for expected, actual in zip(*results):
        assert_bits_equal(expected, actual)


@pytest.mark.parametrize("layout", ["strided", "expanded", "negative"])
def test_noncontiguous_upstream(layout):
    results = []
    for variant in ("native", "both_fused"):
        module, x, dy = make_case(42, variant)
        if layout == "strided":
            dy = dy.transpose(-1, -2).contiguous().transpose(-1, -2)
        elif layout == "expanded":
            dy = dy[:, :1, :1].expand_as(x)
        else:
            dy = torch._neg_view(dy)
        output = module(x)
        results.append(
            torch.autograd.grad(output, (x, module.w13.weight, module.w2.weight), dy)
        )
    for expected, actual in zip(*results):
        assert_bits_equal(expected, actual)


def test_prepared_input_is_distinct_from_hidden_and_is_not_requantized():
    module, x, dy = make_case(42, "both_fused", input_save_format="mxfp8")
    expected_output = module(x)
    expected_gradients = torch.autograd.grad(
        expected_output, (x, module.w13.weight, module.w2.weight), dy
    )
    with torch.no_grad():
        prepared_input = autograd.quantize(x)
    with patch.object(autograd, "quantize", wraps=autograd.quantize) as quantize:
        output = module(x, prepared_input=prepared_input)
        assert quantize.call_count == 0
        gradients = torch.autograd.grad(
            output, (x, module.w13.weight, module.w2.weight), dy
        )
        assert quantize.call_count == 1  # Only dy; hidden operands came from W13.
    for expected, actual in zip(
        (expected_output, *expected_gradients), (output, *gradients)
    ):
        assert_bits_equal(expected, actual)
    with pytest.raises(ValueError, match="Prepared input"):
        module(x, prepared_input=prepared_input[:2])


@pytest.mark.parametrize("target", ["w13", "w2", "root"])
def test_observable_forward_hooks_use_native_path(target):
    module, x, dy = make_case(42, "both_fused")
    expected = module(x)
    seen = []
    child = module if target == "root" else getattr(module, target)
    with child.register_forward_hook(
        lambda _module, _args, output: seen.append(output.shape)
    ):
        with patch.object(
            autograd, "shared_expert_forward_op", wraps=shared_expert_forward_op
        ) as forward:
            output = module(x)
            output.backward(dy)
            assert forward.call_count == 0
    assert len(seen) == 1
    assert_bits_equal(expected, output)


def test_selective_remat_preserves_native_region_boundaries():
    module, x, _ = make_case(42, "both_fused")
    expected = module(x)
    module.configure_remat_regions(["w13.linear"])
    with patch.object(
        autograd, "shared_expert_forward_op", wraps=shared_expert_forward_op
    ) as forward:
        output = module(x)
        assert forward.call_count == 0
    assert_bits_equal(expected, output)


@pytest.mark.parametrize("checkpoint_impl", ["pytorch", "torch_remat"])
def test_whole_ffn_checkpoint_output_and_gradients(checkpoint_impl):
    import torch_remat as remat
    from torch.utils.checkpoint import checkpoint

    module, x, dy = make_case(73, "both_fused")
    expected = module(x)
    expected_gradients = torch.autograd.grad(
        expected, (x, module.w13.weight, module.w2.weight), dy
    )
    output = (
        checkpoint(module, x, use_reentrant=False)
        if checkpoint_impl == "pytorch"
        else remat.checkpoint()(module)(x)
    )
    gradients = torch.autograd.grad(
        output, (x, module.w13.weight, module.w2.weight), dy
    )
    for expected_value, actual_value in zip(
        (expected, *expected_gradients), (output, *gradients)
    ):
        assert_bits_equal(expected_value, actual_value)


def test_complete_override_compiles_without_graph_breaks():
    module, x, dy = make_case(73, "both_fused")
    expected = module(x)
    expected_gradients = torch.autograd.grad(
        expected, (x, module.w13.weight, module.w2.weight), dy
    )
    output = torch.compile(module, fullgraph=True)(x)
    gradients = torch.autograd.grad(
        output, (x, module.w13.weight, module.w2.weight), dy
    )
    for expected_value, actual_value in zip(
        (expected, *expected_gradients), (output, *gradients)
    ):
        assert_bits_equal(expected_value, actual_value)


def test_cuda_graph_replay_with_updated_inputs_and_upstream_gradients():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference, reference_x, reference_dy = make_case(42, "native")
        module, x, dy = make_case(42, "both_fused")
        operands = (x, module.w13.weight, module.w2.weight)
        for _ in range(3):
            output = module(x)
            torch.autograd.grad(output, operands, dy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = module(x)
            gradients = torch.autograd.grad(output, operands, dy)
        for seed in (42, 73, 919):
            torch.manual_seed(seed)
            with torch.no_grad():
                x.normal_()
                dy.normal_()
                reference_x.copy_(x)
                reference_dy.copy_(dy)
            graph.replay()
            expected = reference(reference_x)
            expected_gradients = torch.autograd.grad(
                expected,
                (reference_x, reference.w13.weight, reference.w2.weight),
                reference_dy,
            )
            for expected_value, actual_value in zip(
                (expected, *expected_gradients), (output, *gradients)
            ):
                assert_bits_equal(expected_value, actual_value)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("case", ["zero", "tiny", "extreme", "nonfinite"])
def test_primitive_boundaries_at_special_values(case):
    torch.manual_seed(42)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    if case == "zero":
        x.zero_()
    elif case == "tiny":
        x.mul_(1e-38)
    elif case == "extreme":
        x.mul_(1e20)
    else:
        x[0, 0], x[1, 0], x[2, 0] = float("nan"), float("inf"), -float("inf")
    row, _, scale, _ = autograd.quantize(x)
    weight13 = _quantize_mxfp8_weight(w13)
    packed = autograd.scaled_mm(
        row, weight13.weight_qdata_fprop_KN, scale, weight13.weight_scale_fprop_swizzled
    )
    hidden = silu_and_mul_op(packed.unflatten(-1, (2, 128)))
    actual_forward = shared_expert_forward_op(
        row, weight13.weight_qdata_dgrad_NK, scale, weight13.weight_scale_fprop_swizzled
    )
    for expected, actual in zip(
        (hidden, packed, *autograd.quantize(hidden)), actual_forward
    ):
        assert_bits_equal(expected, actual)
    weight2 = _quantize_mxfp8_weight(w2)
    d_hidden = autograd.scaled_mm(
        row, weight2.weight_qdata_dgrad_NK, scale, weight2.weight_scale_dgrad_swizzled
    )
    d_packed = silu_and_mul_backward_op(
        d_hidden, packed.unflatten(-1, (2, 128))
    ).flatten(-2)
    actual_backward = launch_swiglu_backward(
        row,
        weight2.weight_qdata_dgrad_NK,
        scale,
        weight2.weight_scale_dgrad_swizzled,
        packed,
        save_intermediates=True,
    )
    for expected, actual in zip(
        (*autograd.quantize(d_packed), d_hidden, d_packed), actual_backward
    ):
        assert_bits_equal(expected, actual)


def test_unprepared_weights_and_outer_routed_residual_gradients():
    results = []
    for variant in ("native", "both_fused"):
        module, x, dy = make_case(42, variant, prepared_weights=False)
        routed = torch.randn_like(x, requires_grad=True)
        residual = torch.randn_like(x, requires_grad=True)
        output = (routed + module(x)) + residual
        gradients = torch.autograd.grad(
            output, (x, module.w13.weight, module.w2.weight, routed, residual), dy
        )
        results.append((output.detach(), *gradients))
    for expected, actual in zip(*results):
        assert_bits_equal(expected, actual)


def test_spmd_local_types_and_values_match_native():
    import spmd_types as spmd
    import torch.distributed as dist
    from spmd_types.checker import typecheck
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from torchtitan.distributed.spmd_types import set_current_spmd_mesh

    if dist.is_initialized():
        pytest.skip("Standalone FakePG typecheck requires an unused default group")
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2, 1, 1), mesh_dim_names=("dp", "cp", "tp"))
        local_type = {"dp": spmd.V, "cp": spmd.R, "tp": spmd.R}
        results, layouts = [], []
        for variant in ("native", "both_fused"):
            module, x, dy = make_case(73, variant)
            x = x.detach().view(4096, 7168).requires_grad_()
            dy = dy.view_as(x)
            with set_current_spmd_mesh(mesh), typecheck(local=True):
                spmd.assert_type(x, local_type)
                spmd.assert_type(dy, local_type)
                for parameter in module.parameters():
                    spmd.assert_type(parameter, spmd.R)
                    operands = parameter.operands
                    for tensor in (
                        operands.weight_qdata_dgrad_NK,
                        operands.weight_scale_fprop_swizzled,
                        operands.weight_scale_dgrad_swizzled,
                    ):
                        spmd.assert_type(tensor, spmd.R)
                with patch.object(
                    autograd, "shared_expert_forward_op", wraps=shared_expert_forward_op
                ) as forward:
                    output = module(x)
                    assert forward.call_count == int(variant == "both_fused")
                spmd.assert_type(output, local_type)
                gradients = torch.autograd.grad(
                    output, (x, module.w13.weight, module.w2.weight), dy
                )
                layouts.append(
                    [spmd.get_local_type(tensor) for tensor in (output, *gradients)]
                )
            results.append((output.detach(), *gradients))
        assert layouts[0] == layouts[1]
        for expected, actual in zip(*results):
            assert_bits_equal(expected, actual)
    finally:
        dist.destroy_process_group()
