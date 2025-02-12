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


# kernel path: /tmp/torchinductor_danvm/gj/cgjugimza7cnhbqbxv4oyj66lp24pgam7flmze4edtdeubwbldku.py
# Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_1
#   silu => convert_element_type_2, convert_element_type_3, mul, sigmoid
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %sigmoid), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %view_3), kwargs = {})
triton_poi_fused_mul_silu_0 = async_compile.triton(
    "triton_poi_fused_mul_silu_0",
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'DB4EC0BC06A1FCBFCDA04BA16907EC3B1E867E352F9777F2A8CBA8D490D26C32', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
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
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        buf0 = empty_strided_cuda((16, 16384), (16384, 1), torch.bfloat16)
        # EXTRA MEM: buf0=(16*16384)

        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(
            reinterpret_tensor(primals_1, (16, 4096), (4096, 1), 0),
            reinterpret_tensor(primals_2, (4096, 16384), (1, 4096), 0),
            out=buf0,
        )
        del primals_2
        buf1 = empty_strided_cuda((16, 16384), (16384, 1), torch.bfloat16)
        # EXTRA MEM: buf0=(16*16384) + buf1=(16*16384)

        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(
            reinterpret_tensor(primals_1, (16, 4096), (4096, 1), 0),
            reinterpret_tensor(primals_3, (4096, 16384), (1, 4096), 0),
            out=buf1,
        )
        buf2 = reinterpret_tensor(buf1, (1, 16, 16384), (262144, 16384, 1), 0)
        del buf1  # reuse
        # EXTRA MEM: buf0=(16*16384) + buf2=(1*16*16384)

        # Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
        stream1 = get_raw_stream(1)
        triton_poi_fused_mul_silu_0.run(
            buf2, buf0, 262144, grid=grid(262144), stream=stream1
        )
        buf3 = empty_strided_cuda((16, 4096), (4096, 1), torch.bfloat16)
        # EXTRA MEM: buf0=(16*16384) + buf2=(1*16*16384) + buf3=(16*4096)

        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(
            reinterpret_tensor(buf2, (16, 16384), (16384, 1), 0),
            reinterpret_tensor(primals_4, (16384, 4096), (1, 16384), 0),
            out=buf3,
        )
        del buf2
        # EXTRA MEM: buf0=(16*16384) + buf3=(16*4096)

        # PEAK EXTRA MEM was line 134: (16*16384) + (1*16*16384) + (16*4096) = 589824 * 2 bytes for bf16 = 655,360 bytes
    return (
        reinterpret_tensor(
            buf3, (1, 16, 4096), (65536, 4096, 1), 0
        ),  # FFN output => (1,16,4096) in bf16 = 131,072 bytes
        primals_1,
        primals_3,
        primals_4,
        buf0,  # w1(x) => (16,16384) in bf16 = 524,288 bytes
    )

    # RETURNS (save for backward?) only buf0 and buf3  => (buf0=(16*16384) + buf3=(1*16*4096)) * 2 bytes for bf16 = 655,360 bytes


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (1, 16, 4096), (65536, 4096, 1), device="cuda:1", dtype=torch.bfloat16
    )
    primals_2 = rand_strided(
        (16384, 4096), (4096, 1), device="cuda:1", dtype=torch.bfloat16
    )
    primals_3 = rand_strided(
        (16384, 4096), (4096, 1), device="cuda:1", dtype=torch.bfloat16
    )
    primals_4 = rand_strided(
        (4096, 16384), (16384, 1), device="cuda:1", dtype=torch.bfloat16
    )
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
