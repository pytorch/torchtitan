# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


@triton.jit
def triton_poi_fused__softmax__to_copy_add_mul_slice_sum_unsqueeze_3(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 117440512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 7168
    x0 = xindex % 7168
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8 * x1), None, eviction_policy="evict_last")
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy="evict_last")
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy="evict_last")
    tmp6 = tl.load(in_ptr3 + (x0 + 50176 * x1), None).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (1 + 8 * x1), None, eviction_policy="evict_last")
    tmp13 = tl.load(in_ptr3 + (7168 + x0 + 50176 * x1), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (2 + 8 * x1), None, eviction_policy="evict_last")
    tmp21 = tl.load(in_ptr3 + (14336 + x0 + 50176 * x1), None).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (3 + 8 * x1), None, eviction_policy="evict_last")
    tmp29 = tl.load(in_ptr3 + (21504 + x0 + 50176 * x1), None).to(tl.float32)
    tmp33 = tl.load(in_ptr0 + (4 + 8 * x1), None, eviction_policy="evict_last")
    tmp37 = tl.load(in_ptr3 + (28672 + x0 + 50176 * x1), None).to(tl.float32)
    tmp41 = tl.load(in_ptr0 + (5 + 8 * x1), None, eviction_policy="evict_last")
    tmp45 = tl.load(in_ptr3 + (35840 + x0 + 50176 * x1), None).to(tl.float32)
    tmp49 = tl.load(in_ptr0 + (6 + 8 * x1), None, eviction_policy="evict_last")
    tmp53 = tl.load(in_ptr3 + (43008 + x0 + 50176 * x1), None).to(tl.float32)
    tmp57 = tl.load(in_ptr0 + (7 + 8 * x1), None, eviction_policy="evict_last")
    tmp61 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp2 = tmp0 - tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp10 = tmp9 - tmp1
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tmp11 / tmp4
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp8 + tmp15
    tmp18 = tmp17 - tmp1
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tmp19 / tmp4
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp24 = tmp16 + tmp23
    tmp26 = tmp25 - tmp1
    tmp27 = libdevice.exp(tmp26)
    tmp28 = tmp27 / tmp4
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp32 = tmp24 + tmp31
    tmp34 = tmp33 - tmp1
    tmp35 = libdevice.exp(tmp34)
    tmp36 = tmp35 / tmp4
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 * tmp38
    tmp40 = tmp32 + tmp39
    tmp42 = tmp41 - tmp1
    tmp43 = libdevice.exp(tmp42)
    tmp44 = tmp43 / tmp4
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp48 = tmp40 + tmp47
    tmp50 = tmp49 - tmp1
    tmp51 = libdevice.exp(tmp50)
    tmp52 = tmp51 / tmp4
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp52 * tmp54
    tmp56 = tmp48 + tmp55
    tmp58 = tmp57 - tmp1
    tmp59 = libdevice.exp(tmp58)
    tmp60 = tmp59 / tmp4
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp60 * tmp62
    tmp64 = tmp56 + tmp63
    tmp65 = tmp64.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp65, None)
