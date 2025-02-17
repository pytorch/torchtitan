# AOT ID: ['0_forward']
import math
import os
import random
import tempfile
from ctypes import c_int, c_long, c_void_p
from math import inf, nan

import torch
import triton
import triton.language as tl
from torch import device, empty_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codegen.memory_planning import _align as align
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.runtime.triton_heuristics import (
    cooperative_reduction_grid,
    end_graph,
    grid,
    grid_combo_kernels,
    split_scan_grid,
    start_graph,
)
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.utils import maybe_profile

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_danvm/j3/cj3aaa4oa3cxacjerv7baomyivvg5jvg7usxhp2tlidhqirw7j2s.py
# Topologically Sorted Source Nodes: [output, output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
# Source node to ATen node mapping:
#   output => _scaled_mm, abs_1, amax, clamp_max_1, clamp_min_2, clamp_min_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, mul_2, mul_3, reciprocal_1, reciprocal_2, reciprocal_3
#   output_1 => _scaled_mm_1, clamp_max_3, clamp_min_6, clamp_min_7, convert_element_type_14, convert_element_type_15, convert_element_type_16, convert_element_type_17, mul_7, mul_8, reciprocal_5, reciprocal_7
# Graph fragment:
#   %abs_1 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%primals_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [-1], True), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_1, torch.float64), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_4, 1e-12), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 448.0), kwargs = {})
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float32), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %convert_element_type_5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, -448.0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 448.0), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_2 : [num_users=3] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_1,), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_5,), kwargs = {})
#   %_scaled_mm : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view, %convert_element_type_7, %reciprocal_2, %reciprocal_3, None, None, torch.bfloat16, True), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_3, torch.float64), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_14, 1e-12), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 448.0), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.float32), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_1, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, %convert_element_type_15), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_8, -448.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 448.0), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_3, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_7 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_15,), kwargs = {})
#   %_scaled_mm_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view, %convert_element_type_17, %reciprocal_2, %reciprocal_7, None, None, torch.bfloat16, True), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_0 = async_compile.triton(
    "triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_0",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp8e4nv', 'out_ptr4': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_0(in_ptr0, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp1, r0_mask & xmask) <------ full input abs?
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tmp5 = tmp3.to(tl.float64)
    tmp6 = tl.full([1, 1], 1e-12, tl.float64)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tl.full([1, 1], 448.0, tl.float64)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (x0), tmp13, xmask) <------ scale
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp12
        tmp17 = -448.0
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = 448.0
        tmp20 = triton_helpers.minimum(tmp18, tmp19)
        tmp21 = tmp20.to(tl.float8e4nv)
        tl.store(out_ptr3 + (r0_1 + 4096*x0), tmp21, r0_mask & xmask)    <---- quantized values
        tl.store(out_ptr4 + (r0_1 + 4096*x0), tmp21, r0_mask & xmask)    <---- quantized values
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/ii/ciixp3lqrhj65bjje7eugwloffecuwrhde43psb7fse27wrv3bex.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
# Source node to ATen node mapping:
#   output => _scaled_mm, abs_2, amax_1, clamp_max_1, clamp_min_2, clamp_min_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, mul_2, mul_3, reciprocal_1, reciprocal_3
# Graph fragment:
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute,), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_2, [0], True), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_1, torch.float64), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_4, 1e-12), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 448.0), kwargs = {})
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float32), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %convert_element_type_5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, -448.0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 448.0), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_5,), kwargs = {})
#   %_scaled_mm : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view, %convert_element_type_7, %reciprocal_2, %reciprocal_3, None, None, torch.bfloat16, True), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_1 = async_compile.triton(
    "triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_1",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_1(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]   # <--- max along dim 1
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp3.to(tl.float64)
        tmp8 = tl.full([1, 1], 1e-12, tl.float64)
        tmp9 = triton_helpers.maximum(tmp7, tmp8)
        tmp10 = tl.full([1, 1], 1, tl.int32)
        tmp11 = tmp10 / tmp9                        <------ scale
        tmp12 = tl.full([1, 1], 448.0, tl.float64)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)                <--- apply scale
        tmp15 = tmp6 * tmp14
        tmp16 = -448.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16) <-- clamp
        tmp18 = 448.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18) <-- clamp
        tmp20 = tmp19.to(tl.float8e4nv)             <--- convert dtype  
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp20, r0_mask)   <----- quantized values
    tmp21 = tmp3.to(tl.float64)
    tmp22 = tl.full([1, 1], 1e-12, tl.float64)   <--- EPS
    tmp23 = triton_helpers.maximum(tmp21, tmp22) <--- apply min EPS
    tmp24 = tl.full([1, 1], 1, tl.int32)
    tmp25 = tmp24 / tmp23                        <---- reciprocal of clamped w/ min EPS
    tmp26 = tl.full([1, 1], 448.0, tl.float64)   <--- max for fp8 dtype (448) 
    tmp27 = tmp25 * tmp26                        <--- dtype max / clamped w/ EPS
    tmp28 = tmp27.to(tl.float32)                
    tmp29 = tmp24 / tmp28                        <--- reciprocal of scale
    tl.store(out_ptr2 + (x0), tmp29, None)       <--- return reciprocal of scale
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/iv/civvg45odupbusgmig3rlca5gn6ph5bbbcnun7xax2e344vikhe7.py
# Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_1 => _scaled_mm_1, abs_4, amax_3, clamp_max_3, clamp_min_6, clamp_min_7, convert_element_type_14, convert_element_type_15, convert_element_type_16, convert_element_type_17, mul_7, mul_8, reciprocal_5, reciprocal_7
# Graph fragment:
#   %abs_4 : [num_users=2] = call_function[target=torch.ops.aten.abs.default](args = (%permute_1,), kwargs = {})
#   %amax_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_4, [0], True), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_3, torch.float64), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_14, 1e-12), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 448.0), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.float32), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_1, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, %convert_element_type_15), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_8, -448.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 448.0), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_3, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_7 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_15,), kwargs = {})
#   %_scaled_mm_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view, %convert_element_type_17, %reciprocal_2, %reciprocal_7, None, None, torch.bfloat16, True), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_2 = async_compile.triton(
    "triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_2",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_2(in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp1, r0_mask) <---- full tensor abs values?
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp3.to(tl.float64)
        tmp8 = tl.full([1, 1], 1e-12, tl.float64)
        tmp9 = triton_helpers.maximum(tmp7, tmp8)
        tmp10 = tl.full([1, 1], 1, tl.int32)
        tmp11 = tmp10 / tmp9
        tmp12 = tl.full([1, 1], 448.0, tl.float64)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp6 * tmp14
        tmp16 = -448.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 448.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tmp20 = tmp19.to(tl.float8e4nv)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp20, r0_mask) <--- quantized values
    tmp21 = tmp3.to(tl.float64)
    tmp22 = tl.full([1, 1], 1e-12, tl.float64)
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = tl.full([1, 1], 1, tl.int32)
    tmp25 = tmp24 / tmp23
    tmp26 = tl.full([1, 1], 448.0, tl.float64)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp24 / tmp28
    tl.store(out_ptr3 + (x0), tmp29, None) <---- scale
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/26/c26iyqqjm3qargz5uwszhy2bsjvfs7pieaclnhfferte4nt6cje4.py
# Topologically Sorted Source Nodes: [silu, mul, output_2], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   mul => mul_9
#   output_2 => abs_5, amax_4
#   silu => convert_element_type_8, convert_element_type_9, mul_4, sigmoid
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %view_5), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_9,), kwargs = {})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_5, [-1], True), kwargs = {})
triton_red_fused_abs_amax_mul_silu_3 = async_compile.triton(
    "triton_red_fused_abs_amax_mul_silu_3",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_mul_silu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_amax_mul_silu_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tl_math.abs(tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/tg/ctgfrwoni6kr2t4tajwdeqfskectpmqoe27bprocl3iryjmtdroa.py
# Topologically Sorted Source Nodes: [silu, mul, output_2], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten._scaled_mm]
# Source node to ATen node mapping:
#   mul => mul_9
#   output_2 => _scaled_mm_2, abs_5, amax_4, clamp_max_5, clamp_min_10, clamp_min_11, convert_element_type_22, convert_element_type_23, convert_element_type_24, convert_element_type_25, mul_12, mul_13, reciprocal_10, reciprocal_11, reciprocal_9
#   silu => convert_element_type_8, convert_element_type_9, mul_4, sigmoid
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %view_5), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_9,), kwargs = {})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_5, [-1], True), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_5, torch.float64), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_22, 1e-12), kwargs = {})
#   %reciprocal_9 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_10,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 448.0), kwargs = {})
#   %convert_element_type_23 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.float32), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %convert_element_type_23), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_13, -448.0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 448.0), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_5, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_10 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_7,), kwargs = {})
#   %reciprocal_11 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_23,), kwargs = {})
#   %_scaled_mm_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_6, %convert_element_type_25, %reciprocal_10, %reciprocal_11, None, None, torch.bfloat16, True), kwargs = {})
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_silu_4 = async_compile.triton(
    "triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_silu_4",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_silu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_silu_4(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float64)
    tmp6 = tl.full([1, 1], 1e-12, tl.float64)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tl.full([1, 1], 448.0, tl.float64)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/j7/cj7bfkg5gexx2hvk4lidnvmdtyno47zcq6r5spgx32odg5wququs.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_2 => _scaled_mm_2, clamp_max_5, clamp_min_10, clamp_min_11, convert_element_type_22, convert_element_type_23, convert_element_type_24, convert_element_type_25, mul_12, mul_13, reciprocal_10, reciprocal_11, reciprocal_9
# Graph fragment:
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_5, torch.float64), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_22, 1e-12), kwargs = {})
#   %reciprocal_9 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_10,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 448.0), kwargs = {})
#   %convert_element_type_23 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.float32), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %convert_element_type_23), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_13, -448.0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 448.0), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_5, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_10 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_7,), kwargs = {})
#   %reciprocal_11 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_23,), kwargs = {})
#   %_scaled_mm_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_6, %convert_element_type_25, %reciprocal_10, %reciprocal_11, None, None, torch.bfloat16, True), kwargs = {})
triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_5 = async_compile.triton(
    "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_5",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float64)
    tmp10 = tl.full([1], 1e-12, tl.float64)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tl.full([1], 448.0, tl.float64)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp7 * tmp16
    tmp18 = -448.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 448.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp21.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x2), tmp22, None)
""",
    device_str="cuda",
)


# kernel path: /tmp/torchinductor_danvm/iu/ciu56dezu4d3prdcizpttizlavmyqmxxi5zt7zpfgpsyibjjb7su.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
# Source node to ATen node mapping:
#   output_2 => _scaled_mm_2, abs_6, amax_5, clamp_max_5, clamp_min_10, clamp_min_11, convert_element_type_22, convert_element_type_23, convert_element_type_24, convert_element_type_25, mul_12, mul_13, reciprocal_10, reciprocal_11, reciprocal_9
# Graph fragment:
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute_2,), kwargs = {})
#   %amax_5 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_6, [0], True), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax_5, torch.float64), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_22, 1e-12), kwargs = {})
#   %reciprocal_9 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_10,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 448.0), kwargs = {})
#   %convert_element_type_23 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.float32), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %convert_element_type_23), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_13, -448.0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 448.0), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_5, torch.float8_e4m3fn), kwargs = {})
#   %reciprocal_10 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_7,), kwargs = {})
#   %reciprocal_11 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_23,), kwargs = {})
#   %_scaled_mm_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_6, %convert_element_type_25, %reciprocal_10, %reciprocal_11, None, None, torch.bfloat16, True), kwargs = {})
triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_6 = async_compile.triton(
    "triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_6",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_6(in_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp3.to(tl.float64)
        tmp8 = tl.full([1, 1], 1e-12, tl.float64)
        tmp9 = triton_helpers.maximum(tmp7, tmp8)
        tmp10 = tl.full([1, 1], 1, tl.int32)
        tmp11 = tmp10 / tmp9
        tmp12 = tl.full([1, 1], 448.0, tl.float64)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp6 * tmp14
        tmp16 = -448.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 448.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tmp20 = tmp19.to(tl.float8e4nv)
        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp20, r0_mask)
    tmp21 = tmp3.to(tl.float64)
    tmp22 = tl.full([1, 1], 1e-12, tl.float64)
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = tl.full([1, 1], 1, tl.int32)
    tmp25 = tmp24 / tmp23
    tmp26 = tl.full([1, 1], 448.0, tl.float64)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp24 / tmp28
    tl.store(out_ptr2 + (x0), tmp29, None)
""",
    device_str="cuda",
)


async_compile.wait(globals())
del async_compile


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16, 4096), (65536, 4096, 1))  # input
    assert_size_stride(primals_2, (16384, 4096), (4096, 1))  # w1
    assert_size_stride(primals_3, (16384, 4096), (4096, 1))  # w3
    assert_size_stride(primals_4, (4096, 16384), (16384, 1))  # w2
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 16, 4096), (65536, 4096, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 4096), (4096, 1), torch.float8_e4m3fn)
        buf10 = empty_strided_cuda((16, 4096), (4096, 1), torch.float8_e4m3fn)
        import pdb

        pdb.set_trace()

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf4:fp8=(16, 4096) + buf10:fp8=(16, 4096)
        # total bytes: 262,208

        # Topologically Sorted Source Nodes: [output, output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        # INPUTS -> get amax, scales, and quantized values
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_0.run(
            primals_1,
            buf0,  # amaxes
            buf3,  # scales
            buf4,  # quantized values
            buf10,  # copy of quantized values
            16,
            4096,
            grid=grid(16),
            stream=stream0,
        )
        buf5 = empty_strided_cuda((4096, 16384), (1, 4096), torch.float8_e4m3fn)
        buf6 = empty_strided_cuda((1, 16384), (16384, 1), torch.float32)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf4:fp8=(16, 4096) + buf10:fp8=(16, 4096) + buf5:fp8=(4096, 16384) + buf6:fp32=(1, 16384)
        # total bytes: 67,436,608

        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        # W1 -> get quantized values and scale (or reciprocal of scale?)
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_1.run(
            primals_2,
            buf5,  # output quantized values
            buf6,  # output scales
            16384,
            4096,
            grid=grid(16384),
            stream=stream0,
        )
        del primals_2
        buf7 = empty_strided_cuda((16, 16384), (16384, 1), torch.bfloat16)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf4:fp8=(16, 4096) + buf10:fp8=(16, 4096) + buf5:fp8=(4096, 16384) + buf6:fp32=(1, 16384) \
        #          + buf7:bf16=(16, 16384)
        # total bytes: 67,960,896

        # Topologically Sorted Source Nodes: [output], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        # SCALED_MM WITH FP8 INPUTS AND W1
        extern_kernels._scaled_mm(
            buf4,  # quantized inputs
            buf5,  # quantized W1
            buf3,  # input scales
            buf6,  # w1 scales
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
            out=buf7,  # w1(x)
        )
        del buf4

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf5:fp8=(4096, 16384) + buf6:fp32=(1, 16384) + buf7:bf16=(16, 16384)
        # total bytes: 67,895,360

        buf8 = empty_strided_cuda((4096, 16384), (1, 4096), torch.bfloat16)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf5:fp8=(4096, 16384) + buf6:fp32=(1, 16384) + buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384)
        # total bytes: 202,113,088

        buf11 = buf5

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf5:fp8=(4096, 16384) +  buf11:fp8=(4096, 16384) + buf6:fp32=(1, 16384) + buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384)
        # total bytes: 269,221,952

        del buf5  # reuse

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf11:fp8=(4096, 16384) + buf6:fp32=(1, 16384) + buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384)
        # total bytes: 202,113,088

        buf12 = buf6
        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf11:fp8=(4096, 16384) + buf6:fp32=(1, 16384) + buf12:fp32=(1, 16384)+  buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384)
        # total bytes: 202,178,624

        del buf6  # reuse

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf11:fp8=(4096, 16384) + buf12:fp32=(1, 16384) + buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384)
        # total bytes: 202,113,088

        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        # W3 -> get quantized values and scale
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_2.run(
            primals_3,  # W3
            buf8,  # full W3 abs values?!
            buf11,  # quantized values
            buf12,  # scales
            16384,
            4096,
            grid=grid(16384),
            stream=stream0,
        )
        buf13 = empty_strided_cuda((16, 16384), (16384, 1), torch.bfloat16)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf10:fp8=(16, 4096) + buf11:fp8=(4096, 16384) + buf12:fp32=(1, 16384) + buf7:bf16=(16, 16384) \
        #          + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384)
        # total bytes: 202,637,376

        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        extern_kernels._scaled_mm(
            buf10,  # quantized inputs
            buf11,  # quantized W3
            buf3,  # input scales
            buf12,  # W3 scales
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
            out=buf13,  # W3(x)
        )
        del buf10
        del buf12

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384)
        # total bytes: 202506304

        buf14 = empty_strided_cuda((1, 16, 1, 2), (32, 2, 32, 1), torch.float32)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384) \
        #          + buf14:fp32=(1, 16, 1, 2)
        # total bytes: 202506432

        # Topologically Sorted Source Nodes: [silu, mul, output_2], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_mul_silu_3.run(
            buf7, buf13, buf14, 32, 8192, grid=grid(32), stream=stream0
        )
        buf15 = empty_strided_cuda((1, 16, 1), (16, 1, 16), torch.bfloat16)
        buf19 = empty_strided_cuda((16, 1), (1, 1), torch.float32)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384) \
        #          + buf14:fp32=(1, 16, 1, 2) + buf15:bf16=(1, 16, 1) + buf19:fp32=(16, 1)
        # total bytes: 202506528

        # Topologically Sorted Source Nodes: [silu, mul, output_2], Original ATen: [aten.silu, aten.mul, aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_silu_4.run(
            buf14, buf15, buf19, 16, 2, grid=grid(16), stream=stream0
        )
        del buf14

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384) \
        #          + buf15:bf16=(1, 16, 1) + buf19:fp32=(16, 1)
        # total bytes: 202506400

        buf17 = empty_strided_cuda((16, 16384), (16384, 1), torch.float8_e4m3fn)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) + buf13:bf16=(16, 16384) \
        #          + buf15:bf16=(1, 16, 1) + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384)
        # total bytes: 202768544

        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_5.run(
            buf7, buf13, buf15, buf17, 262144, grid=grid(262144), stream=stream0
        )
        del buf13
        del buf15

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #          + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384)
        # total bytes: 202244224

        buf18 = reinterpret_tensor(buf11, (16384, 4096), (1, 16384), 0)
        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf11:fp8=(4096, 16384) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #          + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384) + buf18:fp8=(16384, 4096)
        # total bytes: 269353088

        del buf11  # reuse

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #          + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384) + buf18:fp8=(16384, 4096)
        # total bytes: 202244224

        buf20 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #          + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384) + buf18:fp8=(16384, 4096) + buf20:fp32=(1, 4096)
        # total bytes: 202260608

        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.abs, aten.amax, aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        stream0 = get_raw_stream(0)
        triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_mul_reciprocal_6.run(
            primals_4, buf18, buf20, 4096, 16384, grid=grid(4096), stream=stream0
        )
        buf21 = empty_strided_cuda((16, 4096), (4096, 1), torch.bfloat16)

        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #          + buf19:fp32=(16, 1) + buf17:fp8=(16, 16384) + buf18:fp8=(16384, 4096) + buf20:fp32=(1, 4096) + buf21:bf16=(16, 4096)
        # total bytes: 202391680

        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten._to_copy, aten.clamp, aten.reciprocal, aten.mul, aten._scaled_mm]
        extern_kernels._scaled_mm(
            buf17,
            buf18,
            buf19,
            buf20,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
            out=buf21,
        )
        del buf17
        del buf18
        del buf19
        del buf20
        # EXTRA MEM: buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) \
        #           + buf21:bf16=(16, 4096)
        # total bytes: 135004224
    return (
        reinterpret_tensor(
            buf21, (1, 16, 4096), (65536, 4096, 1), 0
        ),  # buf21:bf16=(16, 4096) => small (FFN output)
        primals_1,
        primals_3,
        primals_4,
        buf0,  # buf0:bf16=(1, 16, 4096)    => abs(input) => small
        buf3,  # buf3:fp32=(16, 1)          => rowwise scales for inputs
        buf7,  # buf7:bf16=(16, 16384)      => W1(x) => small
        buf8,  # buf8:bf16=(4096, 16384)    => abs(W3) =>  huge
    )

    # RETURNS (save for backward): buf0:bf16=(1, 16, 4096) + buf3:fp32=(16, 1) + buf7:bf16=(16, 16384) + buf8:bf16=(4096, 16384) = 134,873,152 bytes


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (1, 16, 4096), (65536, 4096, 1), device="cuda:0", dtype=torch.bfloat16
    )
    primals_2 = rand_strided(
        (16384, 4096), (4096, 1), device="cuda:0", dtype=torch.bfloat16
    )
    primals_3 = rand_strided(
        (16384, 4096), (4096, 1), device="cuda:0", dtype=torch.bfloat16
    )
    primals_4 = rand_strided(
        (4096, 16384), (16384, 1), device="cuda:0", dtype=torch.bfloat16
    )
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


def total_bytes(tensors: list[torch.Tensor]) -> int:
    total = 0
    for tensor in tensors:
        total += tensor.element_size() * tensor.numel()
    return total


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
