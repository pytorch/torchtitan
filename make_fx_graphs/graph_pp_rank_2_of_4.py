


def forward(self, forward_mb_index, arg_mb, kwargs_mb):
    forward_mb_index_1, = fx_pytree.tree_flatten_spec([forward_mb_index, arg_mb, kwargs_mb], self._in_spec)
    _tensor_constant0 = self._tensor_constant0
    view = torch.ops.aten.view.default(_tensor_constant0, [1, 16, 32]);  _tensor_constant0 = None
    _to_copy = torch.ops.aten._to_copy.default(view, dtype = torch.float32)
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
    add_ = torch.ops.aten.add_.Scalar(mean, 9.999999747378752e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_);  add_ = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = None
    _param_constant0 = self._param_constant0
    mul_1 = torch.ops.aten.mul.Tensor(mul, _param_constant0);  mul = _param_constant0 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_1 = None
    detach = torch.ops.aten.detach.default(rsqrt);  rsqrt = None
    _param_constant1 = self._param_constant1
    t = torch.ops.aten.t.default(_param_constant1);  _param_constant1 = None
    view_1 = torch.ops.aten.view.default(_to_copy_1, [16, 32])
    mm = torch.ops.aten.mm.default(view_1, t);  view_1 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [1, 16, 3072]);  mm = None
    view_2 = torch.ops.aten.view.default(_unsafe_view, [1, 16, -1, 192]);  _unsafe_view = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_2, [128, 64], -1);  view_2 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1];  split_with_sizes = None
    _to_copy_2 = torch.ops.aten._to_copy.default(getitem_1, dtype = torch.float32);  getitem_1 = None
    view_3 = torch.ops.aten.view.default(_to_copy_2, [1, 16, 16, -1, 2]);  _to_copy_2 = None
    view_as_complex = torch.ops.aten.view_as_complex.default(view_3);  view_3 = None
    _tensor_constant1 = self._tensor_constant1
    view_4 = torch.ops.aten.view.default(_tensor_constant1, [1, 16, 1, 32]);  _tensor_constant1 = None
    mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_4);  view_as_complex = None
    view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
    view_5 = torch.ops.aten.view.default(view_as_real, [1, 16, 16, 64]);  view_as_real = None
    _to_copy_3 = torch.ops.aten._to_copy.default(view_5, dtype = torch.bfloat16);  view_5 = None
    cat = torch.ops.aten.cat.default([getitem, _to_copy_3], -1);  getitem = _to_copy_3 = None
    _param_constant2 = self._param_constant2
    t_1 = torch.ops.aten.t.default(_param_constant2);  _param_constant2 = None
    view_6 = torch.ops.aten.view.default(_to_copy_1, [16, 32]);  _to_copy_1 = None
    mm_1 = torch.ops.aten.mm.default(view_6, t_1);  view_6 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(mm_1, [1, 16, 576]);  mm_1 = None
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(_unsafe_view_1, [512, 64], -1);  _unsafe_view_1 = None
    getitem_2 = split_with_sizes_1[0]
    getitem_3 = split_with_sizes_1[1];  split_with_sizes_1 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(unsqueeze, dtype = torch.float32);  unsqueeze = None
    view_7 = torch.ops.aten.view.default(_to_copy_4, [1, 16, 1, -1, 2]);  _to_copy_4 = None
    view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_7);  view_7 = None
    _tensor_constant1_1 = self._tensor_constant1
    view_8 = torch.ops.aten.view.default(_tensor_constant1_1, [1, 16, 1, 32]);  _tensor_constant1_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_8);  view_as_complex_1 = None
    view_as_real_1 = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
    view_9 = torch.ops.aten.view.default(view_as_real_1, [1, 16, 1, 64]);  view_as_real_1 = None
    _to_copy_5 = torch.ops.aten._to_copy.default(view_9, dtype = torch.bfloat16);  view_9 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.float32)
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_6, 2)
    mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
    add__1 = torch.ops.aten.add_.Scalar(mean_1, 9.999999747378752e-06);  mean_1 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add__1);  add__1 = None
    mul_4 = torch.ops.aten.mul.Tensor(_to_copy_6, rsqrt_1);  _to_copy_6 = None
    _param_constant3 = self._param_constant3
    mul_5 = torch.ops.aten.mul.Tensor(mul_4, _param_constant3);  mul_4 = _param_constant3 = None
    _to_copy_7 = torch.ops.aten._to_copy.default(mul_5, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_5 = None
    detach_1 = torch.ops.aten.detach.default(rsqrt_1);  rsqrt_1 = None
    _param_constant4 = self._param_constant4
    t_2 = torch.ops.aten.t.default(_param_constant4);  _param_constant4 = None
    view_10 = torch.ops.aten.view.default(_to_copy_7, [16, 512]);  _to_copy_7 = None
    mm_2 = torch.ops.aten.mm.default(view_10, t_2);  view_10 = None
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(mm_2, [1, 16, 4096]);  mm_2 = None
    view_11 = torch.ops.aten.view.default(_unsafe_view_2, [1, 16, -1, 256]);  _unsafe_view_2 = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_11, [128, 128], -1);  view_11 = None
    getitem_4 = split_with_sizes_2[0]
    getitem_5 = split_with_sizes_2[1];  split_with_sizes_2 = None
    expand = torch.ops.aten.expand.default(_to_copy_5, [-1, -1, 16, -1]);  _to_copy_5 = None
    cat_1 = torch.ops.aten.cat.default([getitem_4, expand], -1);  getitem_4 = expand = None
    transpose = torch.ops.aten.transpose.int(cat, 1, 2);  cat = None
    transpose_1 = torch.ops.aten.transpose.int(cat_1, 1, 2);  cat_1 = None
    transpose_2 = torch.ops.aten.transpose.int(getitem_5, 1, 2);  getitem_5 = None
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(transpose, transpose_1, transpose_2, None, True, 0.0, True, scale = 0.07216878364870322)
    getitem_6 = _scaled_dot_product_efficient_attention[0]
    getitem_7 = _scaled_dot_product_efficient_attention[1]
    getitem_8 = _scaled_dot_product_efficient_attention[2]
    getitem_9 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    detach_2 = torch.ops.aten.detach.default(getitem_6)
    transpose_3 = torch.ops.aten.transpose.int(getitem_6, 1, 2);  getitem_6 = None
    view_12 = torch.ops.aten.view.default(transpose_3, [1, 16, -1]);  transpose_3 = None
    _param_constant5 = self._param_constant5
    t_3 = torch.ops.aten.t.default(_param_constant5);  _param_constant5 = None
    view_13 = torch.ops.aten.view.default(view_12, [16, 2048]);  view_12 = None
    mm_3 = torch.ops.aten.mm.default(view_13, t_3);  view_13 = None
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(mm_3, [1, 16, 32]);  mm_3 = None
    add = torch.ops.aten.add.Tensor(view, _unsafe_view_3);  _unsafe_view_3 = None
    _to_copy_8 = torch.ops.aten._to_copy.default(add, dtype = torch.float32)
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_8, 2)
    mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
    add__2 = torch.ops.aten.add_.Scalar(mean_2, 9.999999747378752e-06);  mean_2 = None
    rsqrt_2 = torch.ops.aten.rsqrt.default(add__2);  add__2 = None
    mul_6 = torch.ops.aten.mul.Tensor(_to_copy_8, rsqrt_2);  _to_copy_8 = None
    _param_constant6 = self._param_constant6
    mul_7 = torch.ops.aten.mul.Tensor(mul_6, _param_constant6);  mul_6 = _param_constant6 = None
    _to_copy_9 = torch.ops.aten._to_copy.default(mul_7, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_7 = None
    detach_3 = torch.ops.aten.detach.default(rsqrt_2);  rsqrt_2 = None
    view_14 = torch.ops.aten.view.default(_to_copy_9, [-1, 32]);  _to_copy_9 = None
    _param_constant7 = self._param_constant7
    t_4 = torch.ops.aten.t.default(_param_constant7);  _param_constant7 = None
    mm_4 = torch.ops.aten.mm.default(view_14, t_4)
    _to_copy_10 = torch.ops.aten._to_copy.default(mm_4, dtype = torch.float32);  mm_4 = None
    _softmax = torch.ops.aten._softmax.default(_to_copy_10, 1, False);  _to_copy_10 = None
    detach_4 = torch.ops.aten.detach.default(_softmax)
    _tensor_constant2 = self._tensor_constant2
    add_1 = torch.ops.aten.add.Tensor(_softmax, _tensor_constant2);  _tensor_constant2 = None
    topk = torch.ops.aten.topk.default(add_1, 3, 1);  add_1 = None
    getitem_10 = topk[0];  getitem_10 = None
    getitem_11 = topk[1];  topk = None
    gather = torch.ops.aten.gather.default(_softmax, 1, getitem_11);  _softmax = None
    mul_8 = torch.ops.aten.mul.Tensor(gather, 1.0);  gather = None
    view_15 = torch.ops.aten.view.default(getitem_11, [-1])
    histc = torch.ops.aten.histc.default(view_15, 8, 0, 8);  view_15 = None
    _tensor_constant3 = self._tensor_constant3
    add__3 = torch.ops.aten.add_.Tensor(_tensor_constant3, histc);  _tensor_constant3 = histc = add__3 = None
    view_16 = torch.ops.aten.view.default(getitem_11, [-1])
    histc_1 = torch.ops.aten.histc.default(view_16, 8, 0, 8);  view_16 = None
    view_17 = torch.ops.aten.view.default(getitem_11, [-1])
    sort = torch.ops.aten.sort.stable(view_17, stable = True);  view_17 = None
    getitem_12 = sort[0];  getitem_12 = None
    getitem_13 = sort[1];  sort = None
    view_18 = torch.ops.aten.view.default(mul_8, [-1]);  mul_8 = None
    index = torch.ops.aten.index.Tensor(view_18, [getitem_13]);  view_18 = None
    floor_divide = torch.ops.aten.floor_divide.default(getitem_13, 3)
    view_19 = torch.ops.aten.view.default(floor_divide, [-1, 1]);  floor_divide = None
    expand_1 = torch.ops.aten.expand.default(view_19, [-1, 32]);  view_19 = None
    gather_1 = torch.ops.aten.gather.default(view_14, 0, expand_1)
    view_20 = torch.ops.aten.view.default(gather_1, [48, 32]);  gather_1 = None
    all_to_all_single = torch.ops._c10d_functional.all_to_all_single.default(histc_1, [4, 4], [4, 4], '8')
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
    wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(wait_tensor);  wait_tensor = None
    view_21 = torch.ops.aten.view.default(histc_1, [2, -1]);  histc_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(view_21, [1]);  view_21 = None
    _to_copy_11 = torch.ops.aten._to_copy.default(sum_1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), non_blocking = True);  sum_1 = _to_copy_11 = None
    view_22 = torch.ops.aten.view.default(wait_tensor_1, [2, -1])
    sum_2 = torch.ops.aten.sum.dim_IntList(view_22, [1]);  view_22 = None
    _to_copy_12 = torch.ops.aten._to_copy.default(sum_2, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'));  sum_2 = _to_copy_12 = None
    all_to_all_single_1 = torch.ops._c10d_functional.all_to_all_single.default(view_20, [22, 23], [22, 26], '8');  view_20 = None
    wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
    view_23 = torch.ops.aten.view.default(wait_tensor_2, [45, 32]);  wait_tensor_2 = None
    _tensor_constant4 = self._tensor_constant4
    view_24 = torch.ops.aten.view.default(_tensor_constant4, [4, 256, 32]);  _tensor_constant4 = None
    _tensor_constant5 = self._tensor_constant5
    view_25 = torch.ops.aten.view.default(_tensor_constant5, [4, 32, 256]);  _tensor_constant5 = None
    _tensor_constant6 = self._tensor_constant6
    view_26 = torch.ops.aten.view.default(_tensor_constant6, [4, 256, 32]);  _tensor_constant6 = None
    cumsum = torch.ops.aten.cumsum.default(wait_tensor_1, 0)
    sub = torch.ops.aten.sub.Tensor(cumsum, wait_tensor_1);  cumsum = sub = None
    view_27 = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
    sum_3 = torch.ops.aten.sum.dim_IntList(view_27, [0]);  view_27 = None
    clamp_min = torch.ops.aten.clamp_min.default(sum_3, 8);  sum_3 = None
    add_2 = torch.ops.aten.add.Tensor(clamp_min, 8);  clamp_min = None
    sub_1 = torch.ops.aten.sub.Tensor(add_2, 1);  add_2 = None
    floor_divide_1 = torch.ops.aten.floor_divide.default(sub_1, 8);  sub_1 = None
    mul_9 = torch.ops.aten.mul.Tensor(floor_divide_1, 8);  floor_divide_1 = None
    _to_copy_13 = torch.ops.aten._to_copy.default(mul_9, dtype = torch.int32);  mul_9 = None
    cumsum_1 = torch.ops.aten.cumsum.default(_to_copy_13, 0)
    sub_2 = torch.ops.aten.sub.Tensor(cumsum_1, _to_copy_13);  sub_2 = None
    full = torch.ops.aten.full.default([77], -1, dtype = torch.int32, device = device(type='cuda', index=2), pin_memory = False)
    _to_copy_14 = torch.ops.aten._to_copy.default(cumsum_1, dtype = torch.int32);  cumsum_1 = _to_copy_14 = None
    new_zeros = torch.ops.aten.new_zeros.default(view_23, [32], pin_memory = False)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(new_zeros, 0);  new_zeros = None
    cat_2 = torch.ops.aten.cat.default([view_23, unsqueeze_1]);  view_23 = unsqueeze_1 = None
    index_1 = torch.ops.aten.index.Tensor(cat_2, [full]);  cat_2 = None
    cumsum_2 = torch.ops.aten.cumsum.default(_to_copy_13, 0, dtype = torch.int32);  _to_copy_13 = None
    transpose_4 = torch.ops.aten.transpose.int(view_24, -2, -1);  view_24 = None
    _grouped_mm = torch.ops.aten._grouped_mm.default(index_1, transpose_4, cumsum_2)
    silu = torch.ops.aten.silu.default(_grouped_mm)
    transpose_5 = torch.ops.aten.transpose.int(view_26, -2, -1);  view_26 = None
    _grouped_mm_1 = torch.ops.aten._grouped_mm.default(index_1, transpose_5, cumsum_2);  index_1 = None
    mul_10 = torch.ops.aten.mul.Tensor(silu, _grouped_mm_1)
    transpose_6 = torch.ops.aten.transpose.int(view_25, -2, -1);  view_25 = None
    _grouped_mm_2 = torch.ops.aten._grouped_mm.default(mul_10, transpose_6, cumsum_2);  mul_10 = None
    new_empty = torch.ops.aten.new_empty.default(_grouped_mm_2, [46, 32], pin_memory = False)
    index_put_ = torch.ops.aten.index_put_.default(new_empty, [full], _grouped_mm_2);  new_empty = _grouped_mm_2 = None
    slice_1 = torch.ops.aten.slice.Tensor(index_put_, 0, 0, -1);  index_put_ = None
    all_to_all_single_2 = torch.ops._c10d_functional.all_to_all_single.default(slice_1, [22, 26], [22, 23], '8');  slice_1 = None
    wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_2);  all_to_all_single_2 = None
    view_28 = torch.ops.aten.view.default(wait_tensor_3, [48, 32]);  wait_tensor_3 = None
    _to_copy_15 = torch.ops.aten._to_copy.default(view_28, dtype = torch.float32);  view_28 = None
    view_29 = torch.ops.aten.view.default(index, [-1, 1]);  index = None
    mul_11 = torch.ops.aten.mul.Tensor(_to_copy_15, view_29)
    _to_copy_16 = torch.ops.aten._to_copy.default(mul_11, dtype = torch.bfloat16);  mul_11 = None
    _param_constant8 = self._param_constant8
    t_5 = torch.ops.aten.t.default(_param_constant8);  _param_constant8 = None
    mm_5 = torch.ops.aten.mm.default(view_14, t_5)
    silu_1 = torch.ops.aten.silu.default(mm_5)
    _param_constant9 = self._param_constant9
    t_6 = torch.ops.aten.t.default(_param_constant9);  _param_constant9 = None
    mm_6 = torch.ops.aten.mm.default(view_14, t_6);  view_14 = None
    mul_12 = torch.ops.aten.mul.Tensor(silu_1, mm_6)
    _param_constant10 = self._param_constant10
    t_7 = torch.ops.aten.t.default(_param_constant10);  _param_constant10 = None
    mm_7 = torch.ops.aten.mm.default(mul_12, t_7);  mul_12 = None
    scatter_add = torch.ops.aten.scatter_add.default(mm_7, 0, expand_1, _to_copy_16);  mm_7 = _to_copy_16 = None
    view_30 = torch.ops.aten.view.default(scatter_add, [1, 16, 32]);  scatter_add = None
    add_3 = torch.ops.aten.add.Tensor(add, view_30);  view_30 = None
    view_31 = torch.ops.aten.view.default(add_3, [1, 16, 32]);  add_3 = None
    _to_copy_17 = torch.ops.aten._to_copy.default(view_31, dtype = torch.float32)
    pow_4 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_17, 2)
    mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
    add__4 = torch.ops.aten.add_.Scalar(mean_3, 9.999999747378752e-06);  mean_3 = None
    rsqrt_3 = torch.ops.aten.rsqrt.default(add__4);  add__4 = None
    mul_13 = torch.ops.aten.mul.Tensor(_to_copy_17, rsqrt_3);  _to_copy_17 = None
    _param_constant11 = self._param_constant11
    mul_14 = torch.ops.aten.mul.Tensor(mul_13, _param_constant11);  mul_13 = _param_constant11 = None
    _to_copy_18 = torch.ops.aten._to_copy.default(mul_14, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_14 = None
    detach_5 = torch.ops.aten.detach.default(rsqrt_3);  rsqrt_3 = None
    _param_constant12 = self._param_constant12
    t_8 = torch.ops.aten.t.default(_param_constant12);  _param_constant12 = None
    view_32 = torch.ops.aten.view.default(_to_copy_18, [16, 32])
    mm_8 = torch.ops.aten.mm.default(view_32, t_8);  view_32 = None
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(mm_8, [1, 16, 3072]);  mm_8 = None
    view_33 = torch.ops.aten.view.default(_unsafe_view_4, [1, 16, -1, 192]);  _unsafe_view_4 = None
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_33, [128, 64], -1);  view_33 = None
    getitem_14 = split_with_sizes_3[0]
    getitem_15 = split_with_sizes_3[1];  split_with_sizes_3 = None
    _to_copy_19 = torch.ops.aten._to_copy.default(getitem_15, dtype = torch.float32);  getitem_15 = None
    view_34 = torch.ops.aten.view.default(_to_copy_19, [1, 16, 16, -1, 2]);  _to_copy_19 = None
    view_as_complex_2 = torch.ops.aten.view_as_complex.default(view_34);  view_34 = None
    _tensor_constant1_2 = self._tensor_constant1
    view_35 = torch.ops.aten.view.default(_tensor_constant1_2, [1, 16, 1, 32]);  _tensor_constant1_2 = None
    mul_15 = torch.ops.aten.mul.Tensor(view_as_complex_2, view_35);  view_as_complex_2 = None
    view_as_real_2 = torch.ops.aten.view_as_real.default(mul_15);  mul_15 = None
    view_36 = torch.ops.aten.view.default(view_as_real_2, [1, 16, 16, 64]);  view_as_real_2 = None
    _to_copy_20 = torch.ops.aten._to_copy.default(view_36, dtype = torch.bfloat16);  view_36 = None
    cat_3 = torch.ops.aten.cat.default([getitem_14, _to_copy_20], -1);  getitem_14 = _to_copy_20 = None
    _param_constant13 = self._param_constant13
    t_9 = torch.ops.aten.t.default(_param_constant13);  _param_constant13 = None
    view_37 = torch.ops.aten.view.default(_to_copy_18, [16, 32]);  _to_copy_18 = None
    mm_9 = torch.ops.aten.mm.default(view_37, t_9);  view_37 = None
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(mm_9, [1, 16, 576]);  mm_9 = None
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(_unsafe_view_5, [512, 64], -1);  _unsafe_view_5 = None
    getitem_16 = split_with_sizes_4[0]
    getitem_17 = split_with_sizes_4[1];  split_with_sizes_4 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_17, 2);  getitem_17 = None
    _to_copy_21 = torch.ops.aten._to_copy.default(unsqueeze_2, dtype = torch.float32);  unsqueeze_2 = None
    view_38 = torch.ops.aten.view.default(_to_copy_21, [1, 16, 1, -1, 2]);  _to_copy_21 = None
    view_as_complex_3 = torch.ops.aten.view_as_complex.default(view_38);  view_38 = None
    _tensor_constant1_3 = self._tensor_constant1
    view_39 = torch.ops.aten.view.default(_tensor_constant1_3, [1, 16, 1, 32]);  _tensor_constant1_3 = None
    mul_16 = torch.ops.aten.mul.Tensor(view_as_complex_3, view_39);  view_as_complex_3 = None
    view_as_real_3 = torch.ops.aten.view_as_real.default(mul_16);  mul_16 = None
    view_40 = torch.ops.aten.view.default(view_as_real_3, [1, 16, 1, 64]);  view_as_real_3 = None
    _to_copy_22 = torch.ops.aten._to_copy.default(view_40, dtype = torch.bfloat16);  view_40 = None
    _to_copy_23 = torch.ops.aten._to_copy.default(getitem_16, dtype = torch.float32)
    pow_5 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_23, 2)
    mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
    add__5 = torch.ops.aten.add_.Scalar(mean_4, 9.999999747378752e-06);  mean_4 = None
    rsqrt_4 = torch.ops.aten.rsqrt.default(add__5);  add__5 = None
    mul_17 = torch.ops.aten.mul.Tensor(_to_copy_23, rsqrt_4);  _to_copy_23 = None
    _param_constant14 = self._param_constant14
    mul_18 = torch.ops.aten.mul.Tensor(mul_17, _param_constant14);  mul_17 = _param_constant14 = None
    _to_copy_24 = torch.ops.aten._to_copy.default(mul_18, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_18 = None
    detach_6 = torch.ops.aten.detach.default(rsqrt_4);  rsqrt_4 = None
    _param_constant15 = self._param_constant15
    t_10 = torch.ops.aten.t.default(_param_constant15);  _param_constant15 = None
    view_41 = torch.ops.aten.view.default(_to_copy_24, [16, 512]);  _to_copy_24 = None
    mm_10 = torch.ops.aten.mm.default(view_41, t_10);  view_41 = None
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(mm_10, [1, 16, 4096]);  mm_10 = None
    view_42 = torch.ops.aten.view.default(_unsafe_view_6, [1, 16, -1, 256]);  _unsafe_view_6 = None
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_42, [128, 128], -1);  view_42 = None
    getitem_18 = split_with_sizes_5[0]
    getitem_19 = split_with_sizes_5[1];  split_with_sizes_5 = None
    expand_2 = torch.ops.aten.expand.default(_to_copy_22, [-1, -1, 16, -1]);  _to_copy_22 = None
    cat_4 = torch.ops.aten.cat.default([getitem_18, expand_2], -1);  getitem_18 = expand_2 = None
    transpose_7 = torch.ops.aten.transpose.int(cat_3, 1, 2);  cat_3 = None
    transpose_8 = torch.ops.aten.transpose.int(cat_4, 1, 2);  cat_4 = None
    transpose_9 = torch.ops.aten.transpose.int(getitem_19, 1, 2);  getitem_19 = None
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(transpose_7, transpose_8, transpose_9, None, True, 0.0, True, scale = 0.07216878364870322)
    getitem_20 = _scaled_dot_product_efficient_attention_1[0]
    getitem_21 = _scaled_dot_product_efficient_attention_1[1]
    getitem_22 = _scaled_dot_product_efficient_attention_1[2]
    getitem_23 = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    detach_7 = torch.ops.aten.detach.default(getitem_20)
    transpose_10 = torch.ops.aten.transpose.int(getitem_20, 1, 2);  getitem_20 = None
    view_43 = torch.ops.aten.view.default(transpose_10, [1, 16, -1]);  transpose_10 = None
    _param_constant16 = self._param_constant16
    t_11 = torch.ops.aten.t.default(_param_constant16);  _param_constant16 = None
    view_44 = torch.ops.aten.view.default(view_43, [16, 2048]);  view_43 = None
    mm_11 = torch.ops.aten.mm.default(view_44, t_11);  view_44 = None
    _unsafe_view_7 = torch.ops.aten._unsafe_view.default(mm_11, [1, 16, 32]);  mm_11 = None
    add_4 = torch.ops.aten.add.Tensor(view_31, _unsafe_view_7);  _unsafe_view_7 = None
    _to_copy_25 = torch.ops.aten._to_copy.default(add_4, dtype = torch.float32)
    pow_6 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_25, 2)
    mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
    add__6 = torch.ops.aten.add_.Scalar(mean_5, 9.999999747378752e-06);  mean_5 = None
    rsqrt_5 = torch.ops.aten.rsqrt.default(add__6);  add__6 = None
    mul_19 = torch.ops.aten.mul.Tensor(_to_copy_25, rsqrt_5);  _to_copy_25 = None
    _param_constant17 = self._param_constant17
    mul_20 = torch.ops.aten.mul.Tensor(mul_19, _param_constant17);  mul_19 = _param_constant17 = None
    _to_copy_26 = torch.ops.aten._to_copy.default(mul_20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_20 = None
    detach_8 = torch.ops.aten.detach.default(rsqrt_5);  rsqrt_5 = None
    view_45 = torch.ops.aten.view.default(_to_copy_26, [-1, 32]);  _to_copy_26 = None
    _param_constant18 = self._param_constant18
    t_12 = torch.ops.aten.t.default(_param_constant18);  _param_constant18 = None
    mm_12 = torch.ops.aten.mm.default(view_45, t_12)
    _to_copy_27 = torch.ops.aten._to_copy.default(mm_12, dtype = torch.float32);  mm_12 = None
    _softmax_1 = torch.ops.aten._softmax.default(_to_copy_27, 1, False);  _to_copy_27 = None
    detach_9 = torch.ops.aten.detach.default(_softmax_1)
    _tensor_constant7 = self._tensor_constant7
    add_5 = torch.ops.aten.add.Tensor(_softmax_1, _tensor_constant7);  _tensor_constant7 = None
    topk_1 = torch.ops.aten.topk.default(add_5, 3, 1);  add_5 = None
    getitem_24 = topk_1[0];  getitem_24 = None
    getitem_25 = topk_1[1];  topk_1 = None
    gather_2 = torch.ops.aten.gather.default(_softmax_1, 1, getitem_25);  _softmax_1 = None
    mul_21 = torch.ops.aten.mul.Tensor(gather_2, 1.0);  gather_2 = None
    view_46 = torch.ops.aten.view.default(getitem_25, [-1])
    histc_2 = torch.ops.aten.histc.default(view_46, 8, 0, 8);  view_46 = None
    _tensor_constant8 = self._tensor_constant8
    add__7 = torch.ops.aten.add_.Tensor(_tensor_constant8, histc_2);  _tensor_constant8 = histc_2 = add__7 = None
    view_47 = torch.ops.aten.view.default(getitem_25, [-1])
    histc_3 = torch.ops.aten.histc.default(view_47, 8, 0, 8);  view_47 = None
    view_48 = torch.ops.aten.view.default(getitem_25, [-1])
    sort_1 = torch.ops.aten.sort.stable(view_48, stable = True);  view_48 = None
    getitem_26 = sort_1[0];  getitem_26 = None
    getitem_27 = sort_1[1];  sort_1 = None
    view_49 = torch.ops.aten.view.default(mul_21, [-1]);  mul_21 = None
    index_2 = torch.ops.aten.index.Tensor(view_49, [getitem_27]);  view_49 = None
    floor_divide_2 = torch.ops.aten.floor_divide.default(getitem_27, 3)
    view_50 = torch.ops.aten.view.default(floor_divide_2, [-1, 1]);  floor_divide_2 = None
    expand_3 = torch.ops.aten.expand.default(view_50, [-1, 32]);  view_50 = None
    gather_3 = torch.ops.aten.gather.default(view_45, 0, expand_3)
    view_51 = torch.ops.aten.view.default(gather_3, [48, 32]);  gather_3 = None
    all_to_all_single_3 = torch.ops._c10d_functional.all_to_all_single.default(histc_3, [4, 4], [4, 4], '8')
    wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_3);  all_to_all_single_3 = None
    wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_4);  wait_tensor_4 = None
    view_52 = torch.ops.aten.view.default(histc_3, [2, -1]);  histc_3 = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_52, [1]);  view_52 = None
    _to_copy_28 = torch.ops.aten._to_copy.default(sum_4, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), non_blocking = True);  sum_4 = _to_copy_28 = None
    view_53 = torch.ops.aten.view.default(wait_tensor_5, [2, -1])
    sum_5 = torch.ops.aten.sum.dim_IntList(view_53, [1]);  view_53 = None
    _to_copy_29 = torch.ops.aten._to_copy.default(sum_5, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'));  sum_5 = _to_copy_29 = None
    all_to_all_single_4 = torch.ops._c10d_functional.all_to_all_single.default(view_51, [25, 19], [25, 23], '8');  view_51 = None
    wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_4);  all_to_all_single_4 = None
    view_54 = torch.ops.aten.view.default(wait_tensor_6, [44, 32]);  wait_tensor_6 = None
    _tensor_constant9 = self._tensor_constant9
    view_55 = torch.ops.aten.view.default(_tensor_constant9, [4, 256, 32]);  _tensor_constant9 = None
    _tensor_constant10 = self._tensor_constant10
    view_56 = torch.ops.aten.view.default(_tensor_constant10, [4, 32, 256]);  _tensor_constant10 = None
    _tensor_constant11 = self._tensor_constant11
    view_57 = torch.ops.aten.view.default(_tensor_constant11, [4, 256, 32]);  _tensor_constant11 = None
    cumsum_3 = torch.ops.aten.cumsum.default(wait_tensor_5, 0)
    sub_3 = torch.ops.aten.sub.Tensor(cumsum_3, wait_tensor_5);  cumsum_3 = sub_3 = None
    view_58 = torch.ops.aten.view.default(wait_tensor_5, [2, -1]);  wait_tensor_5 = None
    sum_6 = torch.ops.aten.sum.dim_IntList(view_58, [0]);  view_58 = None
    clamp_min_1 = torch.ops.aten.clamp_min.default(sum_6, 8);  sum_6 = None
    add_6 = torch.ops.aten.add.Tensor(clamp_min_1, 8);  clamp_min_1 = None
    sub_4 = torch.ops.aten.sub.Tensor(add_6, 1);  add_6 = None
    floor_divide_3 = torch.ops.aten.floor_divide.default(sub_4, 8);  sub_4 = None
    mul_22 = torch.ops.aten.mul.Tensor(floor_divide_3, 8);  floor_divide_3 = None
    _to_copy_30 = torch.ops.aten._to_copy.default(mul_22, dtype = torch.int32);  mul_22 = None
    cumsum_4 = torch.ops.aten.cumsum.default(_to_copy_30, 0)
    sub_5 = torch.ops.aten.sub.Tensor(cumsum_4, _to_copy_30);  sub_5 = None
    full_1 = torch.ops.aten.full.default([76], -1, dtype = torch.int32, device = device(type='cuda', index=2), pin_memory = False)
    _to_copy_31 = torch.ops.aten._to_copy.default(cumsum_4, dtype = torch.int32);  cumsum_4 = _to_copy_31 = None
    new_zeros_1 = torch.ops.aten.new_zeros.default(view_54, [32], pin_memory = False)
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(new_zeros_1, 0);  new_zeros_1 = None
    cat_5 = torch.ops.aten.cat.default([view_54, unsqueeze_3]);  view_54 = unsqueeze_3 = None
    index_3 = torch.ops.aten.index.Tensor(cat_5, [full_1]);  cat_5 = None
    cumsum_5 = torch.ops.aten.cumsum.default(_to_copy_30, 0, dtype = torch.int32);  _to_copy_30 = None
    transpose_11 = torch.ops.aten.transpose.int(view_55, -2, -1);  view_55 = None
    _grouped_mm_3 = torch.ops.aten._grouped_mm.default(index_3, transpose_11, cumsum_5)
    silu_2 = torch.ops.aten.silu.default(_grouped_mm_3)
    transpose_12 = torch.ops.aten.transpose.int(view_57, -2, -1);  view_57 = None
    _grouped_mm_4 = torch.ops.aten._grouped_mm.default(index_3, transpose_12, cumsum_5);  index_3 = None
    mul_23 = torch.ops.aten.mul.Tensor(silu_2, _grouped_mm_4)
    transpose_13 = torch.ops.aten.transpose.int(view_56, -2, -1);  view_56 = None
    _grouped_mm_5 = torch.ops.aten._grouped_mm.default(mul_23, transpose_13, cumsum_5);  mul_23 = None
    new_empty_1 = torch.ops.aten.new_empty.default(_grouped_mm_5, [45, 32], pin_memory = False)
    index_put__1 = torch.ops.aten.index_put_.default(new_empty_1, [full_1], _grouped_mm_5);  new_empty_1 = _grouped_mm_5 = None
    slice_2 = torch.ops.aten.slice.Tensor(index_put__1, 0, 0, -1);  index_put__1 = None
    all_to_all_single_5 = torch.ops._c10d_functional.all_to_all_single.default(slice_2, [25, 23], [25, 19], '8');  slice_2 = None
    wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_5);  all_to_all_single_5 = None
    view_59 = torch.ops.aten.view.default(wait_tensor_7, [48, 32]);  wait_tensor_7 = None
    _to_copy_32 = torch.ops.aten._to_copy.default(view_59, dtype = torch.float32);  view_59 = None
    view_60 = torch.ops.aten.view.default(index_2, [-1, 1]);  index_2 = None
    mul_24 = torch.ops.aten.mul.Tensor(_to_copy_32, view_60)
    _to_copy_33 = torch.ops.aten._to_copy.default(mul_24, dtype = torch.bfloat16);  mul_24 = None
    _param_constant19 = self._param_constant19
    t_13 = torch.ops.aten.t.default(_param_constant19);  _param_constant19 = None
    mm_13 = torch.ops.aten.mm.default(view_45, t_13)
    silu_3 = torch.ops.aten.silu.default(mm_13)
    _param_constant20 = self._param_constant20
    t_14 = torch.ops.aten.t.default(_param_constant20);  _param_constant20 = None
    mm_14 = torch.ops.aten.mm.default(view_45, t_14);  view_45 = None
    mul_25 = torch.ops.aten.mul.Tensor(silu_3, mm_14)
    _param_constant21 = self._param_constant21
    t_15 = torch.ops.aten.t.default(_param_constant21);  _param_constant21 = None
    mm_15 = torch.ops.aten.mm.default(mul_25, t_15);  mul_25 = None
    scatter_add_1 = torch.ops.aten.scatter_add.default(mm_15, 0, expand_3, _to_copy_33);  mm_15 = _to_copy_33 = None
    view_61 = torch.ops.aten.view.default(scatter_add_1, [1, 16, 32]);  scatter_add_1 = None
    add_7 = torch.ops.aten.add.Tensor(add_4, view_61);  view_61 = None
    view_62 = torch.ops.aten.view.default(add_7, [1, 16, 32]);  add_7 = None
    _to_copy_34 = torch.ops.aten._to_copy.default(view_62, dtype = torch.float32)
    pow_7 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_34, 2)
    mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
    add__8 = torch.ops.aten.add_.Scalar(mean_6, 9.999999747378752e-06);  mean_6 = None
    rsqrt_6 = torch.ops.aten.rsqrt.default(add__8);  add__8 = None
    mul_26 = torch.ops.aten.mul.Tensor(_to_copy_34, rsqrt_6);  _to_copy_34 = None
    _param_constant22 = self._param_constant22
    mul_27 = torch.ops.aten.mul.Tensor(mul_26, _param_constant22);  mul_26 = _param_constant22 = None
    _to_copy_35 = torch.ops.aten._to_copy.default(mul_27, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_27 = None
    detach_10 = torch.ops.aten.detach.default(rsqrt_6);  rsqrt_6 = None
    _param_constant23 = self._param_constant23
    t_16 = torch.ops.aten.t.default(_param_constant23);  _param_constant23 = None
    view_63 = torch.ops.aten.view.default(_to_copy_35, [16, 32])
    mm_16 = torch.ops.aten.mm.default(view_63, t_16);  view_63 = None
    _unsafe_view_8 = torch.ops.aten._unsafe_view.default(mm_16, [1, 16, 3072]);  mm_16 = None
    view_64 = torch.ops.aten.view.default(_unsafe_view_8, [1, 16, -1, 192]);  _unsafe_view_8 = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_64, [128, 64], -1);  view_64 = None
    getitem_28 = split_with_sizes_6[0]
    getitem_29 = split_with_sizes_6[1];  split_with_sizes_6 = None
    _to_copy_36 = torch.ops.aten._to_copy.default(getitem_29, dtype = torch.float32);  getitem_29 = None
    view_65 = torch.ops.aten.view.default(_to_copy_36, [1, 16, 16, -1, 2]);  _to_copy_36 = None
    view_as_complex_4 = torch.ops.aten.view_as_complex.default(view_65);  view_65 = None
    _tensor_constant1_4 = self._tensor_constant1
    view_66 = torch.ops.aten.view.default(_tensor_constant1_4, [1, 16, 1, 32]);  _tensor_constant1_4 = None
    mul_28 = torch.ops.aten.mul.Tensor(view_as_complex_4, view_66);  view_as_complex_4 = None
    view_as_real_4 = torch.ops.aten.view_as_real.default(mul_28);  mul_28 = None
    view_67 = torch.ops.aten.view.default(view_as_real_4, [1, 16, 16, 64]);  view_as_real_4 = None
    _to_copy_37 = torch.ops.aten._to_copy.default(view_67, dtype = torch.bfloat16);  view_67 = None
    cat_6 = torch.ops.aten.cat.default([getitem_28, _to_copy_37], -1);  getitem_28 = _to_copy_37 = None
    _param_constant24 = self._param_constant24
    t_17 = torch.ops.aten.t.default(_param_constant24);  _param_constant24 = None
    view_68 = torch.ops.aten.view.default(_to_copy_35, [16, 32]);  _to_copy_35 = None
    mm_17 = torch.ops.aten.mm.default(view_68, t_17);  view_68 = None
    _unsafe_view_9 = torch.ops.aten._unsafe_view.default(mm_17, [1, 16, 576]);  mm_17 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(_unsafe_view_9, [512, 64], -1);  _unsafe_view_9 = None
    getitem_30 = split_with_sizes_7[0]
    getitem_31 = split_with_sizes_7[1];  split_with_sizes_7 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
    _to_copy_38 = torch.ops.aten._to_copy.default(unsqueeze_4, dtype = torch.float32);  unsqueeze_4 = None
    view_69 = torch.ops.aten.view.default(_to_copy_38, [1, 16, 1, -1, 2]);  _to_copy_38 = None
    view_as_complex_5 = torch.ops.aten.view_as_complex.default(view_69);  view_69 = None
    _tensor_constant1_5 = self._tensor_constant1
    view_70 = torch.ops.aten.view.default(_tensor_constant1_5, [1, 16, 1, 32]);  _tensor_constant1_5 = None
    mul_29 = torch.ops.aten.mul.Tensor(view_as_complex_5, view_70);  view_as_complex_5 = None
    view_as_real_5 = torch.ops.aten.view_as_real.default(mul_29);  mul_29 = None
    view_71 = torch.ops.aten.view.default(view_as_real_5, [1, 16, 1, 64]);  view_as_real_5 = None
    _to_copy_39 = torch.ops.aten._to_copy.default(view_71, dtype = torch.bfloat16);  view_71 = None
    _to_copy_40 = torch.ops.aten._to_copy.default(getitem_30, dtype = torch.float32)
    pow_8 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_40, 2)
    mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
    add__9 = torch.ops.aten.add_.Scalar(mean_7, 9.999999747378752e-06);  mean_7 = None
    rsqrt_7 = torch.ops.aten.rsqrt.default(add__9);  add__9 = None
    mul_30 = torch.ops.aten.mul.Tensor(_to_copy_40, rsqrt_7);  _to_copy_40 = None
    _param_constant25 = self._param_constant25
    mul_31 = torch.ops.aten.mul.Tensor(mul_30, _param_constant25);  mul_30 = _param_constant25 = None
    _to_copy_41 = torch.ops.aten._to_copy.default(mul_31, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_31 = None
    detach_11 = torch.ops.aten.detach.default(rsqrt_7);  rsqrt_7 = None
    _param_constant26 = self._param_constant26
    t_18 = torch.ops.aten.t.default(_param_constant26);  _param_constant26 = None
    view_72 = torch.ops.aten.view.default(_to_copy_41, [16, 512]);  _to_copy_41 = None
    mm_18 = torch.ops.aten.mm.default(view_72, t_18);  view_72 = None
    _unsafe_view_10 = torch.ops.aten._unsafe_view.default(mm_18, [1, 16, 4096]);  mm_18 = None
    view_73 = torch.ops.aten.view.default(_unsafe_view_10, [1, 16, -1, 256]);  _unsafe_view_10 = None
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_73, [128, 128], -1);  view_73 = None
    getitem_32 = split_with_sizes_8[0]
    getitem_33 = split_with_sizes_8[1];  split_with_sizes_8 = None
    expand_4 = torch.ops.aten.expand.default(_to_copy_39, [-1, -1, 16, -1]);  _to_copy_39 = None
    cat_7 = torch.ops.aten.cat.default([getitem_32, expand_4], -1);  getitem_32 = expand_4 = None
    transpose_14 = torch.ops.aten.transpose.int(cat_6, 1, 2);  cat_6 = None
    transpose_15 = torch.ops.aten.transpose.int(cat_7, 1, 2);  cat_7 = None
    transpose_16 = torch.ops.aten.transpose.int(getitem_33, 1, 2);  getitem_33 = None
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(transpose_14, transpose_15, transpose_16, None, True, 0.0, True, scale = 0.07216878364870322)
    getitem_34 = _scaled_dot_product_efficient_attention_2[0]
    getitem_35 = _scaled_dot_product_efficient_attention_2[1]
    getitem_36 = _scaled_dot_product_efficient_attention_2[2]
    getitem_37 = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    detach_12 = torch.ops.aten.detach.default(getitem_34)
    transpose_17 = torch.ops.aten.transpose.int(getitem_34, 1, 2);  getitem_34 = None
    view_74 = torch.ops.aten.view.default(transpose_17, [1, 16, -1]);  transpose_17 = None
    _param_constant27 = self._param_constant27
    t_19 = torch.ops.aten.t.default(_param_constant27);  _param_constant27 = None
    view_75 = torch.ops.aten.view.default(view_74, [16, 2048]);  view_74 = None
    mm_19 = torch.ops.aten.mm.default(view_75, t_19);  view_75 = None
    _unsafe_view_11 = torch.ops.aten._unsafe_view.default(mm_19, [1, 16, 32]);  mm_19 = None
    add_8 = torch.ops.aten.add.Tensor(view_62, _unsafe_view_11);  _unsafe_view_11 = None
    _to_copy_42 = torch.ops.aten._to_copy.default(add_8, dtype = torch.float32)
    pow_9 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_42, 2)
    mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
    add__10 = torch.ops.aten.add_.Scalar(mean_8, 9.999999747378752e-06);  mean_8 = None
    rsqrt_8 = torch.ops.aten.rsqrt.default(add__10);  add__10 = None
    mul_32 = torch.ops.aten.mul.Tensor(_to_copy_42, rsqrt_8);  _to_copy_42 = None
    _param_constant28 = self._param_constant28
    mul_33 = torch.ops.aten.mul.Tensor(mul_32, _param_constant28);  mul_32 = _param_constant28 = None
    _to_copy_43 = torch.ops.aten._to_copy.default(mul_33, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_33 = None
    detach_13 = torch.ops.aten.detach.default(rsqrt_8);  rsqrt_8 = None
    view_76 = torch.ops.aten.view.default(_to_copy_43, [-1, 32]);  _to_copy_43 = None
    _param_constant29 = self._param_constant29
    t_20 = torch.ops.aten.t.default(_param_constant29);  _param_constant29 = None
    mm_20 = torch.ops.aten.mm.default(view_76, t_20)
    _to_copy_44 = torch.ops.aten._to_copy.default(mm_20, dtype = torch.float32);  mm_20 = None
    _softmax_2 = torch.ops.aten._softmax.default(_to_copy_44, 1, False);  _to_copy_44 = None
    detach_14 = torch.ops.aten.detach.default(_softmax_2)
    _tensor_constant12 = self._tensor_constant12
    add_9 = torch.ops.aten.add.Tensor(_softmax_2, _tensor_constant12);  _tensor_constant12 = None
    topk_2 = torch.ops.aten.topk.default(add_9, 3, 1);  add_9 = None
    getitem_38 = topk_2[0];  getitem_38 = None
    getitem_39 = topk_2[1];  topk_2 = None
    gather_4 = torch.ops.aten.gather.default(_softmax_2, 1, getitem_39);  _softmax_2 = None
    mul_34 = torch.ops.aten.mul.Tensor(gather_4, 1.0);  gather_4 = None
    view_77 = torch.ops.aten.view.default(getitem_39, [-1])
    histc_4 = torch.ops.aten.histc.default(view_77, 8, 0, 8);  view_77 = None
    _tensor_constant13 = self._tensor_constant13
    add__11 = torch.ops.aten.add_.Tensor(_tensor_constant13, histc_4);  _tensor_constant13 = histc_4 = add__11 = None
    view_78 = torch.ops.aten.view.default(getitem_39, [-1])
    histc_5 = torch.ops.aten.histc.default(view_78, 8, 0, 8);  view_78 = None
    view_79 = torch.ops.aten.view.default(getitem_39, [-1])
    sort_2 = torch.ops.aten.sort.stable(view_79, stable = True);  view_79 = None
    getitem_40 = sort_2[0];  getitem_40 = None
    getitem_41 = sort_2[1];  sort_2 = None
    view_80 = torch.ops.aten.view.default(mul_34, [-1]);  mul_34 = None
    index_4 = torch.ops.aten.index.Tensor(view_80, [getitem_41]);  view_80 = None
    floor_divide_4 = torch.ops.aten.floor_divide.default(getitem_41, 3)
    view_81 = torch.ops.aten.view.default(floor_divide_4, [-1, 1]);  floor_divide_4 = None
    expand_5 = torch.ops.aten.expand.default(view_81, [-1, 32]);  view_81 = None
    gather_5 = torch.ops.aten.gather.default(view_76, 0, expand_5)
    view_82 = torch.ops.aten.view.default(gather_5, [48, 32]);  gather_5 = None
    all_to_all_single_6 = torch.ops._c10d_functional.all_to_all_single.default(histc_5, [4, 4], [4, 4], '8')
    wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_6);  all_to_all_single_6 = None
    wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_8);  wait_tensor_8 = None
    view_83 = torch.ops.aten.view.default(histc_5, [2, -1]);  histc_5 = None
    sum_7 = torch.ops.aten.sum.dim_IntList(view_83, [1]);  view_83 = None
    _to_copy_45 = torch.ops.aten._to_copy.default(sum_7, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), non_blocking = True);  sum_7 = _to_copy_45 = None
    view_84 = torch.ops.aten.view.default(wait_tensor_9, [2, -1])
    sum_8 = torch.ops.aten.sum.dim_IntList(view_84, [1]);  view_84 = None
    _to_copy_46 = torch.ops.aten._to_copy.default(sum_8, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'));  sum_8 = _to_copy_46 = None
    all_to_all_single_7 = torch.ops._c10d_functional.all_to_all_single.default(view_82, [23, 26], [23, 25], '8');  view_82 = None
    wait_tensor_10 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_7);  all_to_all_single_7 = None
    view_85 = torch.ops.aten.view.default(wait_tensor_10, [49, 32]);  wait_tensor_10 = None
    _tensor_constant14 = self._tensor_constant14
    view_86 = torch.ops.aten.view.default(_tensor_constant14, [4, 256, 32]);  _tensor_constant14 = None
    _tensor_constant15 = self._tensor_constant15
    view_87 = torch.ops.aten.view.default(_tensor_constant15, [4, 32, 256]);  _tensor_constant15 = None
    _tensor_constant16 = self._tensor_constant16
    view_88 = torch.ops.aten.view.default(_tensor_constant16, [4, 256, 32]);  _tensor_constant16 = None
    cumsum_6 = torch.ops.aten.cumsum.default(wait_tensor_9, 0)
    sub_6 = torch.ops.aten.sub.Tensor(cumsum_6, wait_tensor_9);  cumsum_6 = sub_6 = None
    view_89 = torch.ops.aten.view.default(wait_tensor_9, [2, -1]);  wait_tensor_9 = None
    sum_9 = torch.ops.aten.sum.dim_IntList(view_89, [0]);  view_89 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(sum_9, 8);  sum_9 = None
    add_10 = torch.ops.aten.add.Tensor(clamp_min_2, 8);  clamp_min_2 = None
    sub_7 = torch.ops.aten.sub.Tensor(add_10, 1);  add_10 = None
    floor_divide_5 = torch.ops.aten.floor_divide.default(sub_7, 8);  sub_7 = None
    mul_35 = torch.ops.aten.mul.Tensor(floor_divide_5, 8);  floor_divide_5 = None
    _to_copy_47 = torch.ops.aten._to_copy.default(mul_35, dtype = torch.int32);  mul_35 = None
    cumsum_7 = torch.ops.aten.cumsum.default(_to_copy_47, 0)
    sub_8 = torch.ops.aten.sub.Tensor(cumsum_7, _to_copy_47);  sub_8 = None
    full_2 = torch.ops.aten.full.default([81], -1, dtype = torch.int32, device = device(type='cuda', index=2), pin_memory = False)
    _to_copy_48 = torch.ops.aten._to_copy.default(cumsum_7, dtype = torch.int32);  cumsum_7 = _to_copy_48 = None
    new_zeros_2 = torch.ops.aten.new_zeros.default(view_85, [32], pin_memory = False)
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(new_zeros_2, 0);  new_zeros_2 = None
    cat_8 = torch.ops.aten.cat.default([view_85, unsqueeze_5]);  view_85 = unsqueeze_5 = None
    index_5 = torch.ops.aten.index.Tensor(cat_8, [full_2]);  cat_8 = None
    cumsum_8 = torch.ops.aten.cumsum.default(_to_copy_47, 0, dtype = torch.int32);  _to_copy_47 = None
    transpose_18 = torch.ops.aten.transpose.int(view_86, -2, -1);  view_86 = None
    _grouped_mm_6 = torch.ops.aten._grouped_mm.default(index_5, transpose_18, cumsum_8)
    silu_4 = torch.ops.aten.silu.default(_grouped_mm_6)
    transpose_19 = torch.ops.aten.transpose.int(view_88, -2, -1);  view_88 = None
    _grouped_mm_7 = torch.ops.aten._grouped_mm.default(index_5, transpose_19, cumsum_8);  index_5 = None
    mul_36 = torch.ops.aten.mul.Tensor(silu_4, _grouped_mm_7)
    transpose_20 = torch.ops.aten.transpose.int(view_87, -2, -1);  view_87 = None
    _grouped_mm_8 = torch.ops.aten._grouped_mm.default(mul_36, transpose_20, cumsum_8);  mul_36 = None
    new_empty_2 = torch.ops.aten.new_empty.default(_grouped_mm_8, [50, 32], pin_memory = False)
    index_put__2 = torch.ops.aten.index_put_.default(new_empty_2, [full_2], _grouped_mm_8);  new_empty_2 = _grouped_mm_8 = None
    slice_3 = torch.ops.aten.slice.Tensor(index_put__2, 0, 0, -1);  index_put__2 = None
    all_to_all_single_8 = torch.ops._c10d_functional.all_to_all_single.default(slice_3, [23, 25], [23, 26], '8');  slice_3 = None
    wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_8);  all_to_all_single_8 = None
    view_90 = torch.ops.aten.view.default(wait_tensor_11, [48, 32]);  wait_tensor_11 = None
    _to_copy_49 = torch.ops.aten._to_copy.default(view_90, dtype = torch.float32);  view_90 = None
    view_91 = torch.ops.aten.view.default(index_4, [-1, 1]);  index_4 = None
    mul_37 = torch.ops.aten.mul.Tensor(_to_copy_49, view_91)
    _to_copy_50 = torch.ops.aten._to_copy.default(mul_37, dtype = torch.bfloat16);  mul_37 = None
    _param_constant30 = self._param_constant30
    t_21 = torch.ops.aten.t.default(_param_constant30);  _param_constant30 = None
    mm_21 = torch.ops.aten.mm.default(view_76, t_21)
    silu_5 = torch.ops.aten.silu.default(mm_21)
    _param_constant31 = self._param_constant31
    t_22 = torch.ops.aten.t.default(_param_constant31);  _param_constant31 = None
    mm_22 = torch.ops.aten.mm.default(view_76, t_22);  view_76 = None
    mul_38 = torch.ops.aten.mul.Tensor(silu_5, mm_22)
    _param_constant32 = self._param_constant32
    t_23 = torch.ops.aten.t.default(_param_constant32);  _param_constant32 = None
    mm_23 = torch.ops.aten.mm.default(mul_38, t_23);  mul_38 = None
    scatter_add_2 = torch.ops.aten.scatter_add.default(mm_23, 0, expand_5, _to_copy_50);  mm_23 = _to_copy_50 = None
    view_92 = torch.ops.aten.view.default(scatter_add_2, [1, 16, 32]);  scatter_add_2 = None
    add_11 = torch.ops.aten.add.Tensor(add_8, view_92);  view_92 = None
    view_93 = torch.ops.aten.view.default(add_11, [1, 16, 32]);  add_11 = None
    _to_copy_51 = torch.ops.aten._to_copy.default(view_93, dtype = torch.float32)
    pow_10 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_51, 2)
    mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
    add__12 = torch.ops.aten.add_.Scalar(mean_9, 9.999999747378752e-06);  mean_9 = None
    rsqrt_9 = torch.ops.aten.rsqrt.default(add__12);  add__12 = None
    mul_39 = torch.ops.aten.mul.Tensor(_to_copy_51, rsqrt_9);  _to_copy_51 = None
    _param_constant33 = self._param_constant33
    mul_40 = torch.ops.aten.mul.Tensor(mul_39, _param_constant33);  mul_39 = _param_constant33 = None
    _to_copy_52 = torch.ops.aten._to_copy.default(mul_40, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_40 = None
    detach_15 = torch.ops.aten.detach.default(rsqrt_9);  rsqrt_9 = None
    _param_constant34 = self._param_constant34
    t_24 = torch.ops.aten.t.default(_param_constant34);  _param_constant34 = None
    view_94 = torch.ops.aten.view.default(_to_copy_52, [16, 32])
    mm_24 = torch.ops.aten.mm.default(view_94, t_24);  view_94 = None
    _unsafe_view_12 = torch.ops.aten._unsafe_view.default(mm_24, [1, 16, 3072]);  mm_24 = None
    view_95 = torch.ops.aten.view.default(_unsafe_view_12, [1, 16, -1, 192]);  _unsafe_view_12 = None
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_95, [128, 64], -1);  view_95 = None
    getitem_42 = split_with_sizes_9[0]
    getitem_43 = split_with_sizes_9[1];  split_with_sizes_9 = None
    _to_copy_53 = torch.ops.aten._to_copy.default(getitem_43, dtype = torch.float32);  getitem_43 = None
    view_96 = torch.ops.aten.view.default(_to_copy_53, [1, 16, 16, -1, 2]);  _to_copy_53 = None
    view_as_complex_6 = torch.ops.aten.view_as_complex.default(view_96);  view_96 = None
    _tensor_constant1_6 = self._tensor_constant1
    view_97 = torch.ops.aten.view.default(_tensor_constant1_6, [1, 16, 1, 32]);  _tensor_constant1_6 = None
    mul_41 = torch.ops.aten.mul.Tensor(view_as_complex_6, view_97);  view_as_complex_6 = None
    view_as_real_6 = torch.ops.aten.view_as_real.default(mul_41);  mul_41 = None
    view_98 = torch.ops.aten.view.default(view_as_real_6, [1, 16, 16, 64]);  view_as_real_6 = None
    _to_copy_54 = torch.ops.aten._to_copy.default(view_98, dtype = torch.bfloat16);  view_98 = None
    cat_9 = torch.ops.aten.cat.default([getitem_42, _to_copy_54], -1);  getitem_42 = _to_copy_54 = None
    _param_constant35 = self._param_constant35
    t_25 = torch.ops.aten.t.default(_param_constant35);  _param_constant35 = None
    view_99 = torch.ops.aten.view.default(_to_copy_52, [16, 32]);  _to_copy_52 = None
    mm_25 = torch.ops.aten.mm.default(view_99, t_25);  view_99 = None
    _unsafe_view_13 = torch.ops.aten._unsafe_view.default(mm_25, [1, 16, 576]);  mm_25 = None
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(_unsafe_view_13, [512, 64], -1);  _unsafe_view_13 = None
    getitem_44 = split_with_sizes_10[0]
    getitem_45 = split_with_sizes_10[1];  split_with_sizes_10 = None
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(getitem_45, 2);  getitem_45 = None
    _to_copy_55 = torch.ops.aten._to_copy.default(unsqueeze_6, dtype = torch.float32);  unsqueeze_6 = None
    view_100 = torch.ops.aten.view.default(_to_copy_55, [1, 16, 1, -1, 2]);  _to_copy_55 = None
    view_as_complex_7 = torch.ops.aten.view_as_complex.default(view_100);  view_100 = None
    _tensor_constant1_7 = self._tensor_constant1
    view_101 = torch.ops.aten.view.default(_tensor_constant1_7, [1, 16, 1, 32]);  _tensor_constant1_7 = None
    mul_42 = torch.ops.aten.mul.Tensor(view_as_complex_7, view_101);  view_as_complex_7 = None
    view_as_real_7 = torch.ops.aten.view_as_real.default(mul_42);  mul_42 = None
    view_102 = torch.ops.aten.view.default(view_as_real_7, [1, 16, 1, 64]);  view_as_real_7 = None
    _to_copy_56 = torch.ops.aten._to_copy.default(view_102, dtype = torch.bfloat16);  view_102 = None
    _to_copy_57 = torch.ops.aten._to_copy.default(getitem_44, dtype = torch.float32)
    pow_11 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_57, 2)
    mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
    add__13 = torch.ops.aten.add_.Scalar(mean_10, 9.999999747378752e-06);  mean_10 = None
    rsqrt_10 = torch.ops.aten.rsqrt.default(add__13);  add__13 = None
    mul_43 = torch.ops.aten.mul.Tensor(_to_copy_57, rsqrt_10);  _to_copy_57 = None
    _param_constant36 = self._param_constant36
    mul_44 = torch.ops.aten.mul.Tensor(mul_43, _param_constant36);  mul_43 = _param_constant36 = None
    _to_copy_58 = torch.ops.aten._to_copy.default(mul_44, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_44 = None
    detach_16 = torch.ops.aten.detach.default(rsqrt_10);  rsqrt_10 = None
    _param_constant37 = self._param_constant37
    t_26 = torch.ops.aten.t.default(_param_constant37);  _param_constant37 = None
    view_103 = torch.ops.aten.view.default(_to_copy_58, [16, 512]);  _to_copy_58 = None
    mm_26 = torch.ops.aten.mm.default(view_103, t_26);  view_103 = None
    _unsafe_view_14 = torch.ops.aten._unsafe_view.default(mm_26, [1, 16, 4096]);  mm_26 = None
    view_104 = torch.ops.aten.view.default(_unsafe_view_14, [1, 16, -1, 256]);  _unsafe_view_14 = None
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_104, [128, 128], -1);  view_104 = None
    getitem_46 = split_with_sizes_11[0]
    getitem_47 = split_with_sizes_11[1];  split_with_sizes_11 = None
    expand_6 = torch.ops.aten.expand.default(_to_copy_56, [-1, -1, 16, -1]);  _to_copy_56 = None
    cat_10 = torch.ops.aten.cat.default([getitem_46, expand_6], -1);  getitem_46 = expand_6 = None
    transpose_21 = torch.ops.aten.transpose.int(cat_9, 1, 2);  cat_9 = None
    transpose_22 = torch.ops.aten.transpose.int(cat_10, 1, 2);  cat_10 = None
    transpose_23 = torch.ops.aten.transpose.int(getitem_47, 1, 2);  getitem_47 = None
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(transpose_21, transpose_22, transpose_23, None, True, 0.0, True, scale = 0.07216878364870322)
    getitem_48 = _scaled_dot_product_efficient_attention_3[0]
    getitem_49 = _scaled_dot_product_efficient_attention_3[1]
    getitem_50 = _scaled_dot_product_efficient_attention_3[2]
    getitem_51 = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    detach_17 = torch.ops.aten.detach.default(getitem_48)
    transpose_24 = torch.ops.aten.transpose.int(getitem_48, 1, 2);  getitem_48 = None
    view_105 = torch.ops.aten.view.default(transpose_24, [1, 16, -1]);  transpose_24 = None
    _param_constant38 = self._param_constant38
    t_27 = torch.ops.aten.t.default(_param_constant38);  _param_constant38 = None
    view_106 = torch.ops.aten.view.default(view_105, [16, 2048]);  view_105 = None
    mm_27 = torch.ops.aten.mm.default(view_106, t_27);  view_106 = None
    _unsafe_view_15 = torch.ops.aten._unsafe_view.default(mm_27, [1, 16, 32]);  mm_27 = None
    add_12 = torch.ops.aten.add.Tensor(view_93, _unsafe_view_15);  _unsafe_view_15 = None
    _to_copy_59 = torch.ops.aten._to_copy.default(add_12, dtype = torch.float32)
    pow_12 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_59, 2)
    mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
    add__14 = torch.ops.aten.add_.Scalar(mean_11, 9.999999747378752e-06);  mean_11 = None
    rsqrt_11 = torch.ops.aten.rsqrt.default(add__14);  add__14 = None
    mul_45 = torch.ops.aten.mul.Tensor(_to_copy_59, rsqrt_11);  _to_copy_59 = None
    _param_constant39 = self._param_constant39
    mul_46 = torch.ops.aten.mul.Tensor(mul_45, _param_constant39);  mul_45 = _param_constant39 = None
    _to_copy_60 = torch.ops.aten._to_copy.default(mul_46, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_46 = None
    detach_18 = torch.ops.aten.detach.default(rsqrt_11);  rsqrt_11 = None
    view_107 = torch.ops.aten.view.default(_to_copy_60, [-1, 32]);  _to_copy_60 = None
    _param_constant40 = self._param_constant40
    t_28 = torch.ops.aten.t.default(_param_constant40);  _param_constant40 = None
    mm_28 = torch.ops.aten.mm.default(view_107, t_28)
    _to_copy_61 = torch.ops.aten._to_copy.default(mm_28, dtype = torch.float32);  mm_28 = None
    _softmax_3 = torch.ops.aten._softmax.default(_to_copy_61, 1, False);  _to_copy_61 = None
    detach_19 = torch.ops.aten.detach.default(_softmax_3)
    _tensor_constant17 = self._tensor_constant17
    add_13 = torch.ops.aten.add.Tensor(_softmax_3, _tensor_constant17);  _tensor_constant17 = None
    topk_3 = torch.ops.aten.topk.default(add_13, 3, 1);  add_13 = None
    getitem_52 = topk_3[0];  getitem_52 = None
    getitem_53 = topk_3[1];  topk_3 = None
    gather_6 = torch.ops.aten.gather.default(_softmax_3, 1, getitem_53);  _softmax_3 = None
    mul_47 = torch.ops.aten.mul.Tensor(gather_6, 1.0);  gather_6 = None
    view_108 = torch.ops.aten.view.default(getitem_53, [-1])
    histc_6 = torch.ops.aten.histc.default(view_108, 8, 0, 8);  view_108 = None
    _tensor_constant18 = self._tensor_constant18
    add__15 = torch.ops.aten.add_.Tensor(_tensor_constant18, histc_6);  _tensor_constant18 = histc_6 = add__15 = None
    view_109 = torch.ops.aten.view.default(getitem_53, [-1])
    histc_7 = torch.ops.aten.histc.default(view_109, 8, 0, 8);  view_109 = None
    view_110 = torch.ops.aten.view.default(getitem_53, [-1])
    sort_3 = torch.ops.aten.sort.stable(view_110, stable = True);  view_110 = None
    getitem_54 = sort_3[0];  getitem_54 = None
    getitem_55 = sort_3[1];  sort_3 = None
    view_111 = torch.ops.aten.view.default(mul_47, [-1]);  mul_47 = None
    index_6 = torch.ops.aten.index.Tensor(view_111, [getitem_55]);  view_111 = None
    floor_divide_6 = torch.ops.aten.floor_divide.default(getitem_55, 3)
    view_112 = torch.ops.aten.view.default(floor_divide_6, [-1, 1]);  floor_divide_6 = None
    expand_7 = torch.ops.aten.expand.default(view_112, [-1, 32]);  view_112 = None
    gather_7 = torch.ops.aten.gather.default(view_107, 0, expand_7)
    view_113 = torch.ops.aten.view.default(gather_7, [48, 32]);  gather_7 = None
    all_to_all_single_9 = torch.ops._c10d_functional.all_to_all_single.default(histc_7, [4, 4], [4, 4], '8')
    wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_9);  all_to_all_single_9 = None
    wait_tensor_13 = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_12);  wait_tensor_12 = None
    view_114 = torch.ops.aten.view.default(histc_7, [2, -1]);  histc_7 = None
    sum_10 = torch.ops.aten.sum.dim_IntList(view_114, [1]);  view_114 = None
    _to_copy_62 = torch.ops.aten._to_copy.default(sum_10, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), non_blocking = True);  sum_10 = _to_copy_62 = None
    view_115 = torch.ops.aten.view.default(wait_tensor_13, [2, -1])
    sum_11 = torch.ops.aten.sum.dim_IntList(view_115, [1]);  view_115 = None
    _to_copy_63 = torch.ops.aten._to_copy.default(sum_11, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'));  sum_11 = _to_copy_63 = None
    all_to_all_single_10 = torch.ops._c10d_functional.all_to_all_single.default(view_113, [25, 24], [25, 23], '8');  view_113 = None
    wait_tensor_14 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_10);  all_to_all_single_10 = None
    view_116 = torch.ops.aten.view.default(wait_tensor_14, [49, 32]);  wait_tensor_14 = None
    _tensor_constant19 = self._tensor_constant19
    view_117 = torch.ops.aten.view.default(_tensor_constant19, [4, 256, 32]);  _tensor_constant19 = None
    _tensor_constant20 = self._tensor_constant20
    view_118 = torch.ops.aten.view.default(_tensor_constant20, [4, 32, 256]);  _tensor_constant20 = None
    _tensor_constant21 = self._tensor_constant21
    view_119 = torch.ops.aten.view.default(_tensor_constant21, [4, 256, 32]);  _tensor_constant21 = None
    cumsum_9 = torch.ops.aten.cumsum.default(wait_tensor_13, 0)
    sub_9 = torch.ops.aten.sub.Tensor(cumsum_9, wait_tensor_13);  cumsum_9 = sub_9 = None
    view_120 = torch.ops.aten.view.default(wait_tensor_13, [2, -1]);  wait_tensor_13 = None
    sum_12 = torch.ops.aten.sum.dim_IntList(view_120, [0]);  view_120 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(sum_12, 8);  sum_12 = None
    add_14 = torch.ops.aten.add.Tensor(clamp_min_3, 8);  clamp_min_3 = None
    sub_10 = torch.ops.aten.sub.Tensor(add_14, 1);  add_14 = None
    floor_divide_7 = torch.ops.aten.floor_divide.default(sub_10, 8);  sub_10 = None
    mul_48 = torch.ops.aten.mul.Tensor(floor_divide_7, 8);  floor_divide_7 = None
    _to_copy_64 = torch.ops.aten._to_copy.default(mul_48, dtype = torch.int32);  mul_48 = None
    cumsum_10 = torch.ops.aten.cumsum.default(_to_copy_64, 0)
    sub_11 = torch.ops.aten.sub.Tensor(cumsum_10, _to_copy_64);  sub_11 = None
    full_3 = torch.ops.aten.full.default([81], -1, dtype = torch.int32, device = device(type='cuda', index=2), pin_memory = False)
    _to_copy_65 = torch.ops.aten._to_copy.default(cumsum_10, dtype = torch.int32);  cumsum_10 = _to_copy_65 = None
    new_zeros_3 = torch.ops.aten.new_zeros.default(view_116, [32], pin_memory = False)
    unsqueeze_7 = torch.ops.aten.unsqueeze.default(new_zeros_3, 0);  new_zeros_3 = None
    cat_11 = torch.ops.aten.cat.default([view_116, unsqueeze_7]);  view_116 = unsqueeze_7 = None
    index_7 = torch.ops.aten.index.Tensor(cat_11, [full_3]);  cat_11 = None
    cumsum_11 = torch.ops.aten.cumsum.default(_to_copy_64, 0, dtype = torch.int32);  _to_copy_64 = None
    transpose_25 = torch.ops.aten.transpose.int(view_117, -2, -1);  view_117 = None
    _grouped_mm_9 = torch.ops.aten._grouped_mm.default(index_7, transpose_25, cumsum_11)
    silu_6 = torch.ops.aten.silu.default(_grouped_mm_9)
    transpose_26 = torch.ops.aten.transpose.int(view_119, -2, -1);  view_119 = None
    _grouped_mm_10 = torch.ops.aten._grouped_mm.default(index_7, transpose_26, cumsum_11);  index_7 = None
    mul_49 = torch.ops.aten.mul.Tensor(silu_6, _grouped_mm_10)
    transpose_27 = torch.ops.aten.transpose.int(view_118, -2, -1);  view_118 = None
    _grouped_mm_11 = torch.ops.aten._grouped_mm.default(mul_49, transpose_27, cumsum_11);  mul_49 = None
    new_empty_3 = torch.ops.aten.new_empty.default(_grouped_mm_11, [50, 32], pin_memory = False)
    index_put__3 = torch.ops.aten.index_put_.default(new_empty_3, [full_3], _grouped_mm_11);  new_empty_3 = _grouped_mm_11 = None
    slice_4 = torch.ops.aten.slice.Tensor(index_put__3, 0, 0, -1);  index_put__3 = None
    all_to_all_single_11 = torch.ops._c10d_functional.all_to_all_single.default(slice_4, [25, 23], [25, 24], '8');  slice_4 = None
    wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_11);  all_to_all_single_11 = None
    view_121 = torch.ops.aten.view.default(wait_tensor_15, [48, 32]);  wait_tensor_15 = None
    _to_copy_66 = torch.ops.aten._to_copy.default(view_121, dtype = torch.float32);  view_121 = None
    view_122 = torch.ops.aten.view.default(index_6, [-1, 1]);  index_6 = None
    mul_50 = torch.ops.aten.mul.Tensor(_to_copy_66, view_122)
    _to_copy_67 = torch.ops.aten._to_copy.default(mul_50, dtype = torch.bfloat16);  mul_50 = None
    _param_constant41 = self._param_constant41
    t_29 = torch.ops.aten.t.default(_param_constant41);  _param_constant41 = None
    mm_29 = torch.ops.aten.mm.default(view_107, t_29)
    silu_7 = torch.ops.aten.silu.default(mm_29)
    _param_constant42 = self._param_constant42
    t_30 = torch.ops.aten.t.default(_param_constant42);  _param_constant42 = None
    mm_30 = torch.ops.aten.mm.default(view_107, t_30);  view_107 = None
    mul_51 = torch.ops.aten.mul.Tensor(silu_7, mm_30)
    _param_constant43 = self._param_constant43
    t_31 = torch.ops.aten.t.default(_param_constant43);  _param_constant43 = None
    mm_31 = torch.ops.aten.mm.default(mul_51, t_31);  mul_51 = None
    scatter_add_3 = torch.ops.aten.scatter_add.default(mm_31, 0, expand_7, _to_copy_67);  mm_31 = _to_copy_67 = None
    view_123 = torch.ops.aten.view.default(scatter_add_3, [1, 16, 32]);  scatter_add_3 = None
    add_15 = torch.ops.aten.add.Tensor(add_12, view_123);  view_123 = None
    ones_like = torch.ops.aten.ones_like.default(add_15, pin_memory = False)
    view_124 = torch.ops.aten.view.default(ones_like, [16, 32])
    gather_8 = torch.ops.aten.gather.default(view_124, 0, expand_7)
    t_32 = torch.ops.aten.t.default(t_31);  t_31 = None
    mm_32 = torch.ops.aten.mm.default(view_124, t_32);  view_124 = t_32 = None
    mul_52 = torch.ops.aten.mul.Tensor(mm_32, silu_7);  silu_7 = None
    mul_53 = torch.ops.aten.mul.Tensor(mm_32, mm_30);  mm_32 = mm_30 = None
    t_33 = torch.ops.aten.t.default(t_30);  t_30 = None
    mm_33 = torch.ops.aten.mm.default(mul_52, t_33);  mul_52 = t_33 = None
    sigmoid = torch.ops.aten.sigmoid.default(mm_29)
    empty_like = torch.ops.aten.empty_like.default(sigmoid, memory_format = torch.preserve_format)
    fill_ = torch.ops.aten.fill_.Scalar(empty_like, 1);  empty_like = None
    sub_ = torch.ops.aten.sub_.Tensor(fill_, sigmoid);  fill_ = None
    mul_54 = torch.ops.aten.mul.Tensor(mm_29, sub_);  mm_29 = sub_ = None
    add_16 = torch.ops.aten.add.Scalar(mul_54, 1);  mul_54 = None
    mul_55 = torch.ops.aten.mul.Tensor(sigmoid, add_16);  sigmoid = add_16 = None
    mul_56 = torch.ops.aten.mul.Tensor(mul_53, mul_55);  mul_53 = mul_55 = None
    t_34 = torch.ops.aten.t.default(t_29);  t_29 = None
    mm_34 = torch.ops.aten.mm.default(mul_56, t_34);  mul_56 = t_34 = None
    add_17 = torch.ops.aten.add.Tensor(mm_33, mm_34);  mm_33 = mm_34 = None
    _to_copy_68 = torch.ops.aten._to_copy.default(gather_8, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  gather_8 = None
    mul_57 = torch.ops.aten.mul.Tensor(_to_copy_68, _to_copy_66);  _to_copy_66 = None
    mul_58 = torch.ops.aten.mul.Tensor(_to_copy_68, view_122);  _to_copy_68 = view_122 = None
    sum_13 = torch.ops.aten.sum.dim_IntList(mul_57, [1], True);  mul_57 = None
    view_125 = torch.ops.aten.view.default(sum_13, [48]);  sum_13 = None
    _to_copy_69 = torch.ops.aten._to_copy.default(mul_58, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_58 = None
    all_to_all_single_12 = torch.ops._c10d_functional.all_to_all_single.default(_to_copy_69, [25, 24], [25, 23], '8');  _to_copy_69 = None
    wait_tensor_16 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_12);  all_to_all_single_12 = None
    slice_backward = torch.ops.aten.slice_backward.default(wait_tensor_16, [50, 32], 0, 0, -1, 1);  wait_tensor_16 = None
    index_8 = torch.ops.aten.index.Tensor(slice_backward, [full_3]);  slice_backward = None
    transpose_28 = torch.ops.aten.transpose.int(transpose_27, -2, -1);  transpose_27 = None
    _grouped_mm_12 = torch.ops.aten._grouped_mm.default(index_8, transpose_28, cumsum_11);  index_8 = transpose_28 = None
    mul_59 = torch.ops.aten.mul.Tensor(_grouped_mm_12, silu_6);  silu_6 = None
    mul_60 = torch.ops.aten.mul.Tensor(_grouped_mm_12, _grouped_mm_10);  _grouped_mm_12 = _grouped_mm_10 = None
    transpose_29 = torch.ops.aten.transpose.int(transpose_26, -2, -1);  transpose_26 = None
    _grouped_mm_13 = torch.ops.aten._grouped_mm.default(mul_59, transpose_29, cumsum_11);  mul_59 = transpose_29 = None
    sigmoid_1 = torch.ops.aten.sigmoid.default(_grouped_mm_9)
    empty_like_1 = torch.ops.aten.empty_like.default(sigmoid_1, memory_format = torch.preserve_format)
    fill__1 = torch.ops.aten.fill_.Scalar(empty_like_1, 1);  empty_like_1 = None
    sub__1 = torch.ops.aten.sub_.Tensor(fill__1, sigmoid_1);  fill__1 = None
    mul_61 = torch.ops.aten.mul.Tensor(_grouped_mm_9, sub__1);  _grouped_mm_9 = sub__1 = None
    add_18 = torch.ops.aten.add.Scalar(mul_61, 1);  mul_61 = None
    mul_62 = torch.ops.aten.mul.Tensor(sigmoid_1, add_18);  sigmoid_1 = add_18 = None
    mul_63 = torch.ops.aten.mul.Tensor(mul_60, mul_62);  mul_60 = mul_62 = None
    transpose_30 = torch.ops.aten.transpose.int(transpose_25, -2, -1);  transpose_25 = None
    _grouped_mm_14 = torch.ops.aten._grouped_mm.default(mul_63, transpose_30, cumsum_11);  mul_63 = transpose_30 = cumsum_11 = None
    add_19 = torch.ops.aten.add.Tensor(_grouped_mm_13, _grouped_mm_14);  _grouped_mm_13 = _grouped_mm_14 = None
    new_zeros_4 = torch.ops.aten.new_zeros.default(add_19, [50, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2))
    index_put = torch.ops.aten.index_put.default(new_zeros_4, [full_3], add_19, True);  new_zeros_4 = full_3 = add_19 = None
    slice_5 = torch.ops.aten.slice.Tensor(index_put, 0, 0, 49)
    slice_6 = torch.ops.aten.slice.Tensor(index_put, 0, 49, 50);  index_put = slice_6 = None
    all_to_all_single_13 = torch.ops._c10d_functional.all_to_all_single.default(slice_5, [25, 23], [25, 24], '8');  slice_5 = None
    wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_13);  all_to_all_single_13 = None
    new_zeros_5 = torch.ops.aten.new_zeros.default(wait_tensor_17, [16, 32])
    scatter_add_4 = torch.ops.aten.scatter_add.default(new_zeros_5, 0, expand_7, wait_tensor_17);  new_zeros_5 = expand_7 = wait_tensor_17 = None
    add_20 = torch.ops.aten.add.Tensor(add_17, scatter_add_4);  add_17 = scatter_add_4 = None
    new_zeros_6 = torch.ops.aten.new_zeros.default(view_125, [48], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_1 = torch.ops.aten.index_put.default(new_zeros_6, [getitem_55], view_125, True);  new_zeros_6 = getitem_55 = view_125 = None
    view_126 = torch.ops.aten.view.default(index_put_1, [16, 3]);  index_put_1 = None
    mul_64 = torch.ops.aten.mul.Tensor(view_126, 1.0);  view_126 = None
    new_zeros_7 = torch.ops.aten.new_zeros.default(mul_64, [16, 8])
    scatter_add_5 = torch.ops.aten.scatter_add.default(new_zeros_7, 1, getitem_53, mul_64);  new_zeros_7 = getitem_53 = mul_64 = None
    detach_20 = torch.ops.aten.detach.default(detach_19);  detach_19 = None
    _softmax_backward_data = torch.ops.aten._softmax_backward_data.default(scatter_add_5, detach_20, 1, torch.float32);  scatter_add_5 = detach_20 = None
    _to_copy_70 = torch.ops.aten._to_copy.default(_softmax_backward_data, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  _softmax_backward_data = None
    t_35 = torch.ops.aten.t.default(t_28);  t_28 = None
    mm_35 = torch.ops.aten.mm.default(_to_copy_70, t_35);  _to_copy_70 = t_35 = None
    add_21 = torch.ops.aten.add.Tensor(add_20, mm_35);  add_20 = mm_35 = None
    view_127 = torch.ops.aten.view.default(add_21, [1, 16, 32]);  add_21 = None
    detach_21 = torch.ops.aten.detach.default(detach_18);  detach_18 = None
    _param_constant39_1 = self._param_constant39
    _fused_rms_norm_backward = torch.ops.aten._fused_rms_norm_backward.default(view_127, add_12, [32], detach_21, _param_constant39_1, [True, False]);  view_127 = add_12 = detach_21 = _param_constant39_1 = None
    getitem_56 = _fused_rms_norm_backward[0]
    getitem_57 = _fused_rms_norm_backward[1];  _fused_rms_norm_backward = getitem_57 = None
    add_22 = torch.ops.aten.add.Tensor(ones_like, getitem_56);  ones_like = getitem_56 = None
    view_128 = torch.ops.aten.view.default(add_22, [16, 32])
    t_36 = torch.ops.aten.t.default(t_27);  t_27 = None
    mm_36 = torch.ops.aten.mm.default(view_128, t_36);  view_128 = t_36 = None
    view_129 = torch.ops.aten.view.default(mm_36, [1, 16, 2048]);  mm_36 = None
    view_130 = torch.ops.aten.view.default(view_129, [1, 16, 16, 128]);  view_129 = None
    transpose_31 = torch.ops.aten.transpose.int(view_130, 1, 2);  view_130 = None
    detach_22 = torch.ops.aten.detach.default(detach_17);  detach_17 = None
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(transpose_31, transpose_21, transpose_22, transpose_23, None, detach_22, getitem_49, getitem_50, getitem_51, 0.0, [True, True, True, False], True, scale = 0.07216878364870322);  transpose_31 = transpose_21 = transpose_22 = transpose_23 = detach_22 = getitem_49 = getitem_50 = getitem_51 = None
    getitem_58 = _scaled_dot_product_efficient_attention_backward[0]
    getitem_59 = _scaled_dot_product_efficient_attention_backward[1]
    getitem_60 = _scaled_dot_product_efficient_attention_backward[2]
    getitem_61 = _scaled_dot_product_efficient_attention_backward[3];  _scaled_dot_product_efficient_attention_backward = getitem_61 = None
    transpose_32 = torch.ops.aten.transpose.int(getitem_60, 1, 2);  getitem_60 = None
    transpose_33 = torch.ops.aten.transpose.int(getitem_59, 1, 2);  getitem_59 = None
    transpose_34 = torch.ops.aten.transpose.int(getitem_58, 1, 2);  getitem_58 = None
    slice_7 = torch.ops.aten.slice.Tensor(transpose_33, 3, 0, 128)
    slice_8 = torch.ops.aten.slice.Tensor(transpose_33, 3, 128, 192);  transpose_33 = None
    sum_14 = torch.ops.aten.sum.dim_IntList(slice_8, [2], True);  slice_8 = None
    cat_12 = torch.ops.aten.cat.default([slice_7, transpose_32], 3);  slice_7 = transpose_32 = None
    view_131 = torch.ops.aten.view.default(cat_12, [1, 16, 4096]);  cat_12 = None
    view_132 = torch.ops.aten.view.default(view_131, [16, 4096]);  view_131 = None
    t_37 = torch.ops.aten.t.default(t_26);  t_26 = None
    mm_37 = torch.ops.aten.mm.default(view_132, t_37);  view_132 = t_37 = None
    view_133 = torch.ops.aten.view.default(mm_37, [1, 16, 512]);  mm_37 = None
    detach_23 = torch.ops.aten.detach.default(detach_16);  detach_16 = None
    _param_constant36_1 = self._param_constant36
    _fused_rms_norm_backward_1 = torch.ops.aten._fused_rms_norm_backward.default(view_133, getitem_44, [512], detach_23, _param_constant36_1, [True, False]);  view_133 = getitem_44 = detach_23 = _param_constant36_1 = None
    getitem_62 = _fused_rms_norm_backward_1[0]
    getitem_63 = _fused_rms_norm_backward_1[1];  _fused_rms_norm_backward_1 = getitem_63 = None
    _to_copy_71 = torch.ops.aten._to_copy.default(sum_14, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  sum_14 = None
    view_134 = torch.ops.aten.view.default(_to_copy_71, [1, 16, 1, 32, 2]);  _to_copy_71 = None
    view_as_complex_8 = torch.ops.aten.view_as_complex.default(view_134);  view_134 = None
    _conj = torch.ops.aten._conj.default(view_101);  view_101 = None
    clone = torch.ops.aten.clone.default(_conj);  _conj = None
    mul_65 = torch.ops.aten.mul.Tensor(view_as_complex_8, clone);  view_as_complex_8 = clone = None
    view_as_real_8 = torch.ops.aten.view_as_real.default(mul_65);  mul_65 = None
    view_135 = torch.ops.aten.view.default(view_as_real_8, [1, 16, 1, 64]);  view_as_real_8 = None
    _to_copy_72 = torch.ops.aten._to_copy.default(view_135, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_135 = None
    squeeze = torch.ops.aten.squeeze.dim(_to_copy_72, 2);  _to_copy_72 = None
    cat_13 = torch.ops.aten.cat.default([getitem_62, squeeze], 2);  getitem_62 = squeeze = None
    view_136 = torch.ops.aten.view.default(cat_13, [16, 576]);  cat_13 = None
    t_38 = torch.ops.aten.t.default(t_25);  t_25 = None
    mm_38 = torch.ops.aten.mm.default(view_136, t_38);  view_136 = t_38 = None
    view_137 = torch.ops.aten.view.default(mm_38, [1, 16, 32]);  mm_38 = None
    slice_9 = torch.ops.aten.slice.Tensor(transpose_34, 3, 0, 128)
    slice_10 = torch.ops.aten.slice.Tensor(transpose_34, 3, 128, 192);  transpose_34 = None
    _to_copy_73 = torch.ops.aten._to_copy.default(slice_10, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  slice_10 = None
    view_138 = torch.ops.aten.view.default(_to_copy_73, [1, 16, 16, 32, 2]);  _to_copy_73 = None
    view_as_complex_9 = torch.ops.aten.view_as_complex.default(view_138);  view_138 = None
    _conj_1 = torch.ops.aten._conj.default(view_97);  view_97 = None
    clone_1 = torch.ops.aten.clone.default(_conj_1);  _conj_1 = None
    mul_66 = torch.ops.aten.mul.Tensor(view_as_complex_9, clone_1);  view_as_complex_9 = clone_1 = None
    view_as_real_9 = torch.ops.aten.view_as_real.default(mul_66);  mul_66 = None
    view_139 = torch.ops.aten.view.default(view_as_real_9, [1, 16, 16, 64]);  view_as_real_9 = None
    _to_copy_74 = torch.ops.aten._to_copy.default(view_139, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_139 = None
    cat_14 = torch.ops.aten.cat.default([slice_9, _to_copy_74], 3);  slice_9 = _to_copy_74 = None
    view_140 = torch.ops.aten.view.default(cat_14, [1, 16, 3072]);  cat_14 = None
    view_141 = torch.ops.aten.view.default(view_140, [16, 3072]);  view_140 = None
    t_39 = torch.ops.aten.t.default(t_24);  t_24 = None
    mm_39 = torch.ops.aten.mm.default(view_141, t_39);  view_141 = t_39 = None
    view_142 = torch.ops.aten.view.default(mm_39, [1, 16, 32]);  mm_39 = None
    add_23 = torch.ops.aten.add.Tensor(view_137, view_142);  view_137 = view_142 = None
    detach_24 = torch.ops.aten.detach.default(detach_15);  detach_15 = None
    _param_constant33_1 = self._param_constant33
    _fused_rms_norm_backward_2 = torch.ops.aten._fused_rms_norm_backward.default(add_23, view_93, [32], detach_24, _param_constant33_1, [True, False]);  add_23 = view_93 = detach_24 = _param_constant33_1 = None
    getitem_64 = _fused_rms_norm_backward_2[0]
    getitem_65 = _fused_rms_norm_backward_2[1];  _fused_rms_norm_backward_2 = getitem_65 = None
    add_24 = torch.ops.aten.add.Tensor(add_22, getitem_64);  add_22 = getitem_64 = None
    view_143 = torch.ops.aten.view.default(add_24, [16, 32])
    gather_9 = torch.ops.aten.gather.default(view_143, 0, expand_5)
    t_40 = torch.ops.aten.t.default(t_23);  t_23 = None
    mm_40 = torch.ops.aten.mm.default(view_143, t_40);  view_143 = t_40 = None
    mul_67 = torch.ops.aten.mul.Tensor(mm_40, silu_5);  silu_5 = None
    mul_68 = torch.ops.aten.mul.Tensor(mm_40, mm_22);  mm_40 = mm_22 = None
    t_41 = torch.ops.aten.t.default(t_22);  t_22 = None
    mm_41 = torch.ops.aten.mm.default(mul_67, t_41);  mul_67 = t_41 = None
    sigmoid_2 = torch.ops.aten.sigmoid.default(mm_21)
    empty_like_2 = torch.ops.aten.empty_like.default(sigmoid_2, memory_format = torch.preserve_format)
    fill__2 = torch.ops.aten.fill_.Scalar(empty_like_2, 1);  empty_like_2 = None
    sub__2 = torch.ops.aten.sub_.Tensor(fill__2, sigmoid_2);  fill__2 = None
    mul_69 = torch.ops.aten.mul.Tensor(mm_21, sub__2);  mm_21 = sub__2 = None
    add_25 = torch.ops.aten.add.Scalar(mul_69, 1);  mul_69 = None
    mul_70 = torch.ops.aten.mul.Tensor(sigmoid_2, add_25);  sigmoid_2 = add_25 = None
    mul_71 = torch.ops.aten.mul.Tensor(mul_68, mul_70);  mul_68 = mul_70 = None
    t_42 = torch.ops.aten.t.default(t_21);  t_21 = None
    mm_42 = torch.ops.aten.mm.default(mul_71, t_42);  mul_71 = t_42 = None
    add_26 = torch.ops.aten.add.Tensor(mm_41, mm_42);  mm_41 = mm_42 = None
    _to_copy_75 = torch.ops.aten._to_copy.default(gather_9, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  gather_9 = None
    mul_72 = torch.ops.aten.mul.Tensor(_to_copy_75, _to_copy_49);  _to_copy_49 = None
    mul_73 = torch.ops.aten.mul.Tensor(_to_copy_75, view_91);  _to_copy_75 = view_91 = None
    sum_15 = torch.ops.aten.sum.dim_IntList(mul_72, [1], True);  mul_72 = None
    view_144 = torch.ops.aten.view.default(sum_15, [48]);  sum_15 = None
    _to_copy_76 = torch.ops.aten._to_copy.default(mul_73, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_73 = None
    all_to_all_single_14 = torch.ops._c10d_functional.all_to_all_single.default(_to_copy_76, [23, 26], [23, 25], '8');  _to_copy_76 = None
    wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_14);  all_to_all_single_14 = None
    slice_backward_1 = torch.ops.aten.slice_backward.default(wait_tensor_18, [50, 32], 0, 0, -1, 1);  wait_tensor_18 = None
    index_9 = torch.ops.aten.index.Tensor(slice_backward_1, [full_2]);  slice_backward_1 = None
    transpose_35 = torch.ops.aten.transpose.int(transpose_20, -2, -1);  transpose_20 = None
    _grouped_mm_15 = torch.ops.aten._grouped_mm.default(index_9, transpose_35, cumsum_8);  index_9 = transpose_35 = None
    mul_74 = torch.ops.aten.mul.Tensor(_grouped_mm_15, silu_4);  silu_4 = None
    mul_75 = torch.ops.aten.mul.Tensor(_grouped_mm_15, _grouped_mm_7);  _grouped_mm_15 = _grouped_mm_7 = None
    transpose_36 = torch.ops.aten.transpose.int(transpose_19, -2, -1);  transpose_19 = None
    _grouped_mm_16 = torch.ops.aten._grouped_mm.default(mul_74, transpose_36, cumsum_8);  mul_74 = transpose_36 = None
    sigmoid_3 = torch.ops.aten.sigmoid.default(_grouped_mm_6)
    empty_like_3 = torch.ops.aten.empty_like.default(sigmoid_3, memory_format = torch.preserve_format)
    fill__3 = torch.ops.aten.fill_.Scalar(empty_like_3, 1);  empty_like_3 = None
    sub__3 = torch.ops.aten.sub_.Tensor(fill__3, sigmoid_3);  fill__3 = None
    mul_76 = torch.ops.aten.mul.Tensor(_grouped_mm_6, sub__3);  _grouped_mm_6 = sub__3 = None
    add_27 = torch.ops.aten.add.Scalar(mul_76, 1);  mul_76 = None
    mul_77 = torch.ops.aten.mul.Tensor(sigmoid_3, add_27);  sigmoid_3 = add_27 = None
    mul_78 = torch.ops.aten.mul.Tensor(mul_75, mul_77);  mul_75 = mul_77 = None
    transpose_37 = torch.ops.aten.transpose.int(transpose_18, -2, -1);  transpose_18 = None
    _grouped_mm_17 = torch.ops.aten._grouped_mm.default(mul_78, transpose_37, cumsum_8);  mul_78 = transpose_37 = cumsum_8 = None
    add_28 = torch.ops.aten.add.Tensor(_grouped_mm_16, _grouped_mm_17);  _grouped_mm_16 = _grouped_mm_17 = None
    new_zeros_8 = torch.ops.aten.new_zeros.default(add_28, [50, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_2 = torch.ops.aten.index_put.default(new_zeros_8, [full_2], add_28, True);  new_zeros_8 = full_2 = add_28 = None
    slice_11 = torch.ops.aten.slice.Tensor(index_put_2, 0, 0, 49)
    slice_12 = torch.ops.aten.slice.Tensor(index_put_2, 0, 49, 50);  index_put_2 = slice_12 = None
    all_to_all_single_15 = torch.ops._c10d_functional.all_to_all_single.default(slice_11, [23, 25], [23, 26], '8');  slice_11 = None
    wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_15);  all_to_all_single_15 = None
    new_zeros_9 = torch.ops.aten.new_zeros.default(wait_tensor_19, [16, 32])
    scatter_add_6 = torch.ops.aten.scatter_add.default(new_zeros_9, 0, expand_5, wait_tensor_19);  new_zeros_9 = expand_5 = wait_tensor_19 = None
    add_29 = torch.ops.aten.add.Tensor(add_26, scatter_add_6);  add_26 = scatter_add_6 = None
    new_zeros_10 = torch.ops.aten.new_zeros.default(view_144, [48], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_3 = torch.ops.aten.index_put.default(new_zeros_10, [getitem_41], view_144, True);  new_zeros_10 = getitem_41 = view_144 = None
    view_145 = torch.ops.aten.view.default(index_put_3, [16, 3]);  index_put_3 = None
    mul_79 = torch.ops.aten.mul.Tensor(view_145, 1.0);  view_145 = None
    new_zeros_11 = torch.ops.aten.new_zeros.default(mul_79, [16, 8])
    scatter_add_7 = torch.ops.aten.scatter_add.default(new_zeros_11, 1, getitem_39, mul_79);  new_zeros_11 = getitem_39 = mul_79 = None
    detach_25 = torch.ops.aten.detach.default(detach_14);  detach_14 = None
    _softmax_backward_data_1 = torch.ops.aten._softmax_backward_data.default(scatter_add_7, detach_25, 1, torch.float32);  scatter_add_7 = detach_25 = None
    _to_copy_77 = torch.ops.aten._to_copy.default(_softmax_backward_data_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  _softmax_backward_data_1 = None
    t_43 = torch.ops.aten.t.default(t_20);  t_20 = None
    mm_43 = torch.ops.aten.mm.default(_to_copy_77, t_43);  _to_copy_77 = t_43 = None
    add_30 = torch.ops.aten.add.Tensor(add_29, mm_43);  add_29 = mm_43 = None
    view_146 = torch.ops.aten.view.default(add_30, [1, 16, 32]);  add_30 = None
    detach_26 = torch.ops.aten.detach.default(detach_13);  detach_13 = None
    _param_constant28_1 = self._param_constant28
    _fused_rms_norm_backward_3 = torch.ops.aten._fused_rms_norm_backward.default(view_146, add_8, [32], detach_26, _param_constant28_1, [True, False]);  view_146 = add_8 = detach_26 = _param_constant28_1 = None
    getitem_66 = _fused_rms_norm_backward_3[0]
    getitem_67 = _fused_rms_norm_backward_3[1];  _fused_rms_norm_backward_3 = getitem_67 = None
    add_31 = torch.ops.aten.add.Tensor(add_24, getitem_66);  add_24 = getitem_66 = None
    view_147 = torch.ops.aten.view.default(add_31, [16, 32])
    t_44 = torch.ops.aten.t.default(t_19);  t_19 = None
    mm_44 = torch.ops.aten.mm.default(view_147, t_44);  view_147 = t_44 = None
    view_148 = torch.ops.aten.view.default(mm_44, [1, 16, 2048]);  mm_44 = None
    view_149 = torch.ops.aten.view.default(view_148, [1, 16, 16, 128]);  view_148 = None
    transpose_38 = torch.ops.aten.transpose.int(view_149, 1, 2);  view_149 = None
    detach_27 = torch.ops.aten.detach.default(detach_12);  detach_12 = None
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(transpose_38, transpose_14, transpose_15, transpose_16, None, detach_27, getitem_35, getitem_36, getitem_37, 0.0, [True, True, True, False], True, scale = 0.07216878364870322);  transpose_38 = transpose_14 = transpose_15 = transpose_16 = detach_27 = getitem_35 = getitem_36 = getitem_37 = None
    getitem_68 = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_69 = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_70 = _scaled_dot_product_efficient_attention_backward_1[2]
    getitem_71 = _scaled_dot_product_efficient_attention_backward_1[3];  _scaled_dot_product_efficient_attention_backward_1 = getitem_71 = None
    transpose_39 = torch.ops.aten.transpose.int(getitem_70, 1, 2);  getitem_70 = None
    transpose_40 = torch.ops.aten.transpose.int(getitem_69, 1, 2);  getitem_69 = None
    transpose_41 = torch.ops.aten.transpose.int(getitem_68, 1, 2);  getitem_68 = None
    slice_13 = torch.ops.aten.slice.Tensor(transpose_40, 3, 0, 128)
    slice_14 = torch.ops.aten.slice.Tensor(transpose_40, 3, 128, 192);  transpose_40 = None
    sum_16 = torch.ops.aten.sum.dim_IntList(slice_14, [2], True);  slice_14 = None
    cat_15 = torch.ops.aten.cat.default([slice_13, transpose_39], 3);  slice_13 = transpose_39 = None
    view_150 = torch.ops.aten.view.default(cat_15, [1, 16, 4096]);  cat_15 = None
    view_151 = torch.ops.aten.view.default(view_150, [16, 4096]);  view_150 = None
    t_45 = torch.ops.aten.t.default(t_18);  t_18 = None
    mm_45 = torch.ops.aten.mm.default(view_151, t_45);  view_151 = t_45 = None
    view_152 = torch.ops.aten.view.default(mm_45, [1, 16, 512]);  mm_45 = None
    detach_28 = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    _param_constant25_1 = self._param_constant25
    _fused_rms_norm_backward_4 = torch.ops.aten._fused_rms_norm_backward.default(view_152, getitem_30, [512], detach_28, _param_constant25_1, [True, False]);  view_152 = getitem_30 = detach_28 = _param_constant25_1 = None
    getitem_72 = _fused_rms_norm_backward_4[0]
    getitem_73 = _fused_rms_norm_backward_4[1];  _fused_rms_norm_backward_4 = getitem_73 = None
    _to_copy_78 = torch.ops.aten._to_copy.default(sum_16, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  sum_16 = None
    view_153 = torch.ops.aten.view.default(_to_copy_78, [1, 16, 1, 32, 2]);  _to_copy_78 = None
    view_as_complex_10 = torch.ops.aten.view_as_complex.default(view_153);  view_153 = None
    _conj_2 = torch.ops.aten._conj.default(view_70);  view_70 = None
    clone_2 = torch.ops.aten.clone.default(_conj_2);  _conj_2 = None
    mul_80 = torch.ops.aten.mul.Tensor(view_as_complex_10, clone_2);  view_as_complex_10 = clone_2 = None
    view_as_real_10 = torch.ops.aten.view_as_real.default(mul_80);  mul_80 = None
    view_154 = torch.ops.aten.view.default(view_as_real_10, [1, 16, 1, 64]);  view_as_real_10 = None
    _to_copy_79 = torch.ops.aten._to_copy.default(view_154, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_154 = None
    squeeze_1 = torch.ops.aten.squeeze.dim(_to_copy_79, 2);  _to_copy_79 = None
    cat_16 = torch.ops.aten.cat.default([getitem_72, squeeze_1], 2);  getitem_72 = squeeze_1 = None
    view_155 = torch.ops.aten.view.default(cat_16, [16, 576]);  cat_16 = None
    t_46 = torch.ops.aten.t.default(t_17);  t_17 = None
    mm_46 = torch.ops.aten.mm.default(view_155, t_46);  view_155 = t_46 = None
    view_156 = torch.ops.aten.view.default(mm_46, [1, 16, 32]);  mm_46 = None
    slice_15 = torch.ops.aten.slice.Tensor(transpose_41, 3, 0, 128)
    slice_16 = torch.ops.aten.slice.Tensor(transpose_41, 3, 128, 192);  transpose_41 = None
    _to_copy_80 = torch.ops.aten._to_copy.default(slice_16, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  slice_16 = None
    view_157 = torch.ops.aten.view.default(_to_copy_80, [1, 16, 16, 32, 2]);  _to_copy_80 = None
    view_as_complex_11 = torch.ops.aten.view_as_complex.default(view_157);  view_157 = None
    _conj_3 = torch.ops.aten._conj.default(view_66);  view_66 = None
    clone_3 = torch.ops.aten.clone.default(_conj_3);  _conj_3 = None
    mul_81 = torch.ops.aten.mul.Tensor(view_as_complex_11, clone_3);  view_as_complex_11 = clone_3 = None
    view_as_real_11 = torch.ops.aten.view_as_real.default(mul_81);  mul_81 = None
    view_158 = torch.ops.aten.view.default(view_as_real_11, [1, 16, 16, 64]);  view_as_real_11 = None
    _to_copy_81 = torch.ops.aten._to_copy.default(view_158, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_158 = None
    cat_17 = torch.ops.aten.cat.default([slice_15, _to_copy_81], 3);  slice_15 = _to_copy_81 = None
    view_159 = torch.ops.aten.view.default(cat_17, [1, 16, 3072]);  cat_17 = None
    view_160 = torch.ops.aten.view.default(view_159, [16, 3072]);  view_159 = None
    t_47 = torch.ops.aten.t.default(t_16);  t_16 = None
    mm_47 = torch.ops.aten.mm.default(view_160, t_47);  view_160 = t_47 = None
    view_161 = torch.ops.aten.view.default(mm_47, [1, 16, 32]);  mm_47 = None
    add_32 = torch.ops.aten.add.Tensor(view_156, view_161);  view_156 = view_161 = None
    detach_29 = torch.ops.aten.detach.default(detach_10);  detach_10 = None
    _param_constant22_1 = self._param_constant22
    _fused_rms_norm_backward_5 = torch.ops.aten._fused_rms_norm_backward.default(add_32, view_62, [32], detach_29, _param_constant22_1, [True, False]);  add_32 = view_62 = detach_29 = _param_constant22_1 = None
    getitem_74 = _fused_rms_norm_backward_5[0]
    getitem_75 = _fused_rms_norm_backward_5[1];  _fused_rms_norm_backward_5 = getitem_75 = None
    add_33 = torch.ops.aten.add.Tensor(add_31, getitem_74);  add_31 = getitem_74 = None
    view_162 = torch.ops.aten.view.default(add_33, [16, 32])
    gather_10 = torch.ops.aten.gather.default(view_162, 0, expand_3)
    t_48 = torch.ops.aten.t.default(t_15);  t_15 = None
    mm_48 = torch.ops.aten.mm.default(view_162, t_48);  view_162 = t_48 = None
    mul_82 = torch.ops.aten.mul.Tensor(mm_48, silu_3);  silu_3 = None
    mul_83 = torch.ops.aten.mul.Tensor(mm_48, mm_14);  mm_48 = mm_14 = None
    t_49 = torch.ops.aten.t.default(t_14);  t_14 = None
    mm_49 = torch.ops.aten.mm.default(mul_82, t_49);  mul_82 = t_49 = None
    sigmoid_4 = torch.ops.aten.sigmoid.default(mm_13)
    empty_like_4 = torch.ops.aten.empty_like.default(sigmoid_4, memory_format = torch.preserve_format)
    fill__4 = torch.ops.aten.fill_.Scalar(empty_like_4, 1);  empty_like_4 = None
    sub__4 = torch.ops.aten.sub_.Tensor(fill__4, sigmoid_4);  fill__4 = None
    mul_84 = torch.ops.aten.mul.Tensor(mm_13, sub__4);  mm_13 = sub__4 = None
    add_34 = torch.ops.aten.add.Scalar(mul_84, 1);  mul_84 = None
    mul_85 = torch.ops.aten.mul.Tensor(sigmoid_4, add_34);  sigmoid_4 = add_34 = None
    mul_86 = torch.ops.aten.mul.Tensor(mul_83, mul_85);  mul_83 = mul_85 = None
    t_50 = torch.ops.aten.t.default(t_13);  t_13 = None
    mm_50 = torch.ops.aten.mm.default(mul_86, t_50);  mul_86 = t_50 = None
    add_35 = torch.ops.aten.add.Tensor(mm_49, mm_50);  mm_49 = mm_50 = None
    _to_copy_82 = torch.ops.aten._to_copy.default(gather_10, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  gather_10 = None
    mul_87 = torch.ops.aten.mul.Tensor(_to_copy_82, _to_copy_32);  _to_copy_32 = None
    mul_88 = torch.ops.aten.mul.Tensor(_to_copy_82, view_60);  _to_copy_82 = view_60 = None
    sum_17 = torch.ops.aten.sum.dim_IntList(mul_87, [1], True);  mul_87 = None
    view_163 = torch.ops.aten.view.default(sum_17, [48]);  sum_17 = None
    _to_copy_83 = torch.ops.aten._to_copy.default(mul_88, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_88 = None
    all_to_all_single_16 = torch.ops._c10d_functional.all_to_all_single.default(_to_copy_83, [25, 19], [25, 23], '8');  _to_copy_83 = None
    wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_16);  all_to_all_single_16 = None
    slice_backward_2 = torch.ops.aten.slice_backward.default(wait_tensor_20, [45, 32], 0, 0, -1, 1);  wait_tensor_20 = None
    index_10 = torch.ops.aten.index.Tensor(slice_backward_2, [full_1]);  slice_backward_2 = None
    transpose_42 = torch.ops.aten.transpose.int(transpose_13, -2, -1);  transpose_13 = None
    _grouped_mm_18 = torch.ops.aten._grouped_mm.default(index_10, transpose_42, cumsum_5);  index_10 = transpose_42 = None
    mul_89 = torch.ops.aten.mul.Tensor(_grouped_mm_18, silu_2);  silu_2 = None
    mul_90 = torch.ops.aten.mul.Tensor(_grouped_mm_18, _grouped_mm_4);  _grouped_mm_18 = _grouped_mm_4 = None
    transpose_43 = torch.ops.aten.transpose.int(transpose_12, -2, -1);  transpose_12 = None
    _grouped_mm_19 = torch.ops.aten._grouped_mm.default(mul_89, transpose_43, cumsum_5);  mul_89 = transpose_43 = None
    sigmoid_5 = torch.ops.aten.sigmoid.default(_grouped_mm_3)
    empty_like_5 = torch.ops.aten.empty_like.default(sigmoid_5, memory_format = torch.preserve_format)
    fill__5 = torch.ops.aten.fill_.Scalar(empty_like_5, 1);  empty_like_5 = None
    sub__5 = torch.ops.aten.sub_.Tensor(fill__5, sigmoid_5);  fill__5 = None
    mul_91 = torch.ops.aten.mul.Tensor(_grouped_mm_3, sub__5);  _grouped_mm_3 = sub__5 = None
    add_36 = torch.ops.aten.add.Scalar(mul_91, 1);  mul_91 = None
    mul_92 = torch.ops.aten.mul.Tensor(sigmoid_5, add_36);  sigmoid_5 = add_36 = None
    mul_93 = torch.ops.aten.mul.Tensor(mul_90, mul_92);  mul_90 = mul_92 = None
    transpose_44 = torch.ops.aten.transpose.int(transpose_11, -2, -1);  transpose_11 = None
    _grouped_mm_20 = torch.ops.aten._grouped_mm.default(mul_93, transpose_44, cumsum_5);  mul_93 = transpose_44 = cumsum_5 = None
    add_37 = torch.ops.aten.add.Tensor(_grouped_mm_19, _grouped_mm_20);  _grouped_mm_19 = _grouped_mm_20 = None
    new_zeros_12 = torch.ops.aten.new_zeros.default(add_37, [45, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_4 = torch.ops.aten.index_put.default(new_zeros_12, [full_1], add_37, True);  new_zeros_12 = full_1 = add_37 = None
    slice_17 = torch.ops.aten.slice.Tensor(index_put_4, 0, 0, 44)
    slice_18 = torch.ops.aten.slice.Tensor(index_put_4, 0, 44, 45);  index_put_4 = slice_18 = None
    all_to_all_single_17 = torch.ops._c10d_functional.all_to_all_single.default(slice_17, [25, 23], [25, 19], '8');  slice_17 = None
    wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_17);  all_to_all_single_17 = None
    new_zeros_13 = torch.ops.aten.new_zeros.default(wait_tensor_21, [16, 32])
    scatter_add_8 = torch.ops.aten.scatter_add.default(new_zeros_13, 0, expand_3, wait_tensor_21);  new_zeros_13 = expand_3 = wait_tensor_21 = None
    add_38 = torch.ops.aten.add.Tensor(add_35, scatter_add_8);  add_35 = scatter_add_8 = None
    new_zeros_14 = torch.ops.aten.new_zeros.default(view_163, [48], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_5 = torch.ops.aten.index_put.default(new_zeros_14, [getitem_27], view_163, True);  new_zeros_14 = getitem_27 = view_163 = None
    view_164 = torch.ops.aten.view.default(index_put_5, [16, 3]);  index_put_5 = None
    mul_94 = torch.ops.aten.mul.Tensor(view_164, 1.0);  view_164 = None
    new_zeros_15 = torch.ops.aten.new_zeros.default(mul_94, [16, 8])
    scatter_add_9 = torch.ops.aten.scatter_add.default(new_zeros_15, 1, getitem_25, mul_94);  new_zeros_15 = getitem_25 = mul_94 = None
    detach_30 = torch.ops.aten.detach.default(detach_9);  detach_9 = None
    _softmax_backward_data_2 = torch.ops.aten._softmax_backward_data.default(scatter_add_9, detach_30, 1, torch.float32);  scatter_add_9 = detach_30 = None
    _to_copy_84 = torch.ops.aten._to_copy.default(_softmax_backward_data_2, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  _softmax_backward_data_2 = None
    t_51 = torch.ops.aten.t.default(t_12);  t_12 = None
    mm_51 = torch.ops.aten.mm.default(_to_copy_84, t_51);  _to_copy_84 = t_51 = None
    add_39 = torch.ops.aten.add.Tensor(add_38, mm_51);  add_38 = mm_51 = None
    view_165 = torch.ops.aten.view.default(add_39, [1, 16, 32]);  add_39 = None
    detach_31 = torch.ops.aten.detach.default(detach_8);  detach_8 = None
    _param_constant17_1 = self._param_constant17
    _fused_rms_norm_backward_6 = torch.ops.aten._fused_rms_norm_backward.default(view_165, add_4, [32], detach_31, _param_constant17_1, [True, False]);  view_165 = add_4 = detach_31 = _param_constant17_1 = None
    getitem_76 = _fused_rms_norm_backward_6[0]
    getitem_77 = _fused_rms_norm_backward_6[1];  _fused_rms_norm_backward_6 = getitem_77 = None
    add_40 = torch.ops.aten.add.Tensor(add_33, getitem_76);  add_33 = getitem_76 = None
    view_166 = torch.ops.aten.view.default(add_40, [16, 32])
    t_52 = torch.ops.aten.t.default(t_11);  t_11 = None
    mm_52 = torch.ops.aten.mm.default(view_166, t_52);  view_166 = t_52 = None
    view_167 = torch.ops.aten.view.default(mm_52, [1, 16, 2048]);  mm_52 = None
    view_168 = torch.ops.aten.view.default(view_167, [1, 16, 16, 128]);  view_167 = None
    transpose_45 = torch.ops.aten.transpose.int(view_168, 1, 2);  view_168 = None
    detach_32 = torch.ops.aten.detach.default(detach_7);  detach_7 = None
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(transpose_45, transpose_7, transpose_8, transpose_9, None, detach_32, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False], True, scale = 0.07216878364870322);  transpose_45 = transpose_7 = transpose_8 = transpose_9 = detach_32 = getitem_21 = getitem_22 = getitem_23 = None
    getitem_78 = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_79 = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_80 = _scaled_dot_product_efficient_attention_backward_2[2]
    getitem_81 = _scaled_dot_product_efficient_attention_backward_2[3];  _scaled_dot_product_efficient_attention_backward_2 = getitem_81 = None
    transpose_46 = torch.ops.aten.transpose.int(getitem_80, 1, 2);  getitem_80 = None
    transpose_47 = torch.ops.aten.transpose.int(getitem_79, 1, 2);  getitem_79 = None
    transpose_48 = torch.ops.aten.transpose.int(getitem_78, 1, 2);  getitem_78 = None
    slice_19 = torch.ops.aten.slice.Tensor(transpose_47, 3, 0, 128)
    slice_20 = torch.ops.aten.slice.Tensor(transpose_47, 3, 128, 192);  transpose_47 = None
    sum_18 = torch.ops.aten.sum.dim_IntList(slice_20, [2], True);  slice_20 = None
    cat_18 = torch.ops.aten.cat.default([slice_19, transpose_46], 3);  slice_19 = transpose_46 = None
    view_169 = torch.ops.aten.view.default(cat_18, [1, 16, 4096]);  cat_18 = None
    view_170 = torch.ops.aten.view.default(view_169, [16, 4096]);  view_169 = None
    t_53 = torch.ops.aten.t.default(t_10);  t_10 = None
    mm_53 = torch.ops.aten.mm.default(view_170, t_53);  view_170 = t_53 = None
    view_171 = torch.ops.aten.view.default(mm_53, [1, 16, 512]);  mm_53 = None
    detach_33 = torch.ops.aten.detach.default(detach_6);  detach_6 = None
    _param_constant14_1 = self._param_constant14
    _fused_rms_norm_backward_7 = torch.ops.aten._fused_rms_norm_backward.default(view_171, getitem_16, [512], detach_33, _param_constant14_1, [True, False]);  view_171 = getitem_16 = detach_33 = _param_constant14_1 = None
    getitem_82 = _fused_rms_norm_backward_7[0]
    getitem_83 = _fused_rms_norm_backward_7[1];  _fused_rms_norm_backward_7 = getitem_83 = None
    _to_copy_85 = torch.ops.aten._to_copy.default(sum_18, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  sum_18 = None
    view_172 = torch.ops.aten.view.default(_to_copy_85, [1, 16, 1, 32, 2]);  _to_copy_85 = None
    view_as_complex_12 = torch.ops.aten.view_as_complex.default(view_172);  view_172 = None
    _conj_4 = torch.ops.aten._conj.default(view_39);  view_39 = None
    clone_4 = torch.ops.aten.clone.default(_conj_4);  _conj_4 = None
    mul_95 = torch.ops.aten.mul.Tensor(view_as_complex_12, clone_4);  view_as_complex_12 = clone_4 = None
    view_as_real_12 = torch.ops.aten.view_as_real.default(mul_95);  mul_95 = None
    view_173 = torch.ops.aten.view.default(view_as_real_12, [1, 16, 1, 64]);  view_as_real_12 = None
    _to_copy_86 = torch.ops.aten._to_copy.default(view_173, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_173 = None
    squeeze_2 = torch.ops.aten.squeeze.dim(_to_copy_86, 2);  _to_copy_86 = None
    cat_19 = torch.ops.aten.cat.default([getitem_82, squeeze_2], 2);  getitem_82 = squeeze_2 = None
    view_174 = torch.ops.aten.view.default(cat_19, [16, 576]);  cat_19 = None
    t_54 = torch.ops.aten.t.default(t_9);  t_9 = None
    mm_54 = torch.ops.aten.mm.default(view_174, t_54);  view_174 = t_54 = None
    view_175 = torch.ops.aten.view.default(mm_54, [1, 16, 32]);  mm_54 = None
    slice_21 = torch.ops.aten.slice.Tensor(transpose_48, 3, 0, 128)
    slice_22 = torch.ops.aten.slice.Tensor(transpose_48, 3, 128, 192);  transpose_48 = None
    _to_copy_87 = torch.ops.aten._to_copy.default(slice_22, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  slice_22 = None
    view_176 = torch.ops.aten.view.default(_to_copy_87, [1, 16, 16, 32, 2]);  _to_copy_87 = None
    view_as_complex_13 = torch.ops.aten.view_as_complex.default(view_176);  view_176 = None
    _conj_5 = torch.ops.aten._conj.default(view_35);  view_35 = None
    clone_5 = torch.ops.aten.clone.default(_conj_5);  _conj_5 = None
    mul_96 = torch.ops.aten.mul.Tensor(view_as_complex_13, clone_5);  view_as_complex_13 = clone_5 = None
    view_as_real_13 = torch.ops.aten.view_as_real.default(mul_96);  mul_96 = None
    view_177 = torch.ops.aten.view.default(view_as_real_13, [1, 16, 16, 64]);  view_as_real_13 = None
    _to_copy_88 = torch.ops.aten._to_copy.default(view_177, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_177 = None
    cat_20 = torch.ops.aten.cat.default([slice_21, _to_copy_88], 3);  slice_21 = _to_copy_88 = None
    view_178 = torch.ops.aten.view.default(cat_20, [1, 16, 3072]);  cat_20 = None
    view_179 = torch.ops.aten.view.default(view_178, [16, 3072]);  view_178 = None
    t_55 = torch.ops.aten.t.default(t_8);  t_8 = None
    mm_55 = torch.ops.aten.mm.default(view_179, t_55);  view_179 = t_55 = None
    view_180 = torch.ops.aten.view.default(mm_55, [1, 16, 32]);  mm_55 = None
    add_41 = torch.ops.aten.add.Tensor(view_175, view_180);  view_175 = view_180 = None
    detach_34 = torch.ops.aten.detach.default(detach_5);  detach_5 = None
    _param_constant11_1 = self._param_constant11
    _fused_rms_norm_backward_8 = torch.ops.aten._fused_rms_norm_backward.default(add_41, view_31, [32], detach_34, _param_constant11_1, [True, False]);  add_41 = view_31 = detach_34 = _param_constant11_1 = None
    getitem_84 = _fused_rms_norm_backward_8[0]
    getitem_85 = _fused_rms_norm_backward_8[1];  _fused_rms_norm_backward_8 = getitem_85 = None
    add_42 = torch.ops.aten.add.Tensor(add_40, getitem_84);  add_40 = getitem_84 = None
    view_181 = torch.ops.aten.view.default(add_42, [16, 32])
    gather_11 = torch.ops.aten.gather.default(view_181, 0, expand_1)
    t_56 = torch.ops.aten.t.default(t_7);  t_7 = None
    mm_56 = torch.ops.aten.mm.default(view_181, t_56);  view_181 = t_56 = None
    mul_97 = torch.ops.aten.mul.Tensor(mm_56, silu_1);  silu_1 = None
    mul_98 = torch.ops.aten.mul.Tensor(mm_56, mm_6);  mm_56 = mm_6 = None
    t_57 = torch.ops.aten.t.default(t_6);  t_6 = None
    mm_57 = torch.ops.aten.mm.default(mul_97, t_57);  mul_97 = t_57 = None
    sigmoid_6 = torch.ops.aten.sigmoid.default(mm_5)
    empty_like_6 = torch.ops.aten.empty_like.default(sigmoid_6, memory_format = torch.preserve_format)
    fill__6 = torch.ops.aten.fill_.Scalar(empty_like_6, 1);  empty_like_6 = None
    sub__6 = torch.ops.aten.sub_.Tensor(fill__6, sigmoid_6);  fill__6 = None
    mul_99 = torch.ops.aten.mul.Tensor(mm_5, sub__6);  mm_5 = sub__6 = None
    add_43 = torch.ops.aten.add.Scalar(mul_99, 1);  mul_99 = None
    mul_100 = torch.ops.aten.mul.Tensor(sigmoid_6, add_43);  sigmoid_6 = add_43 = None
    mul_101 = torch.ops.aten.mul.Tensor(mul_98, mul_100);  mul_98 = mul_100 = None
    t_58 = torch.ops.aten.t.default(t_5);  t_5 = None
    mm_58 = torch.ops.aten.mm.default(mul_101, t_58);  mul_101 = t_58 = None
    add_44 = torch.ops.aten.add.Tensor(mm_57, mm_58);  mm_57 = mm_58 = None
    _to_copy_89 = torch.ops.aten._to_copy.default(gather_11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  gather_11 = None
    mul_102 = torch.ops.aten.mul.Tensor(_to_copy_89, _to_copy_15);  _to_copy_15 = None
    mul_103 = torch.ops.aten.mul.Tensor(_to_copy_89, view_29);  _to_copy_89 = view_29 = None
    sum_19 = torch.ops.aten.sum.dim_IntList(mul_102, [1], True);  mul_102 = None
    view_182 = torch.ops.aten.view.default(sum_19, [48]);  sum_19 = None
    _to_copy_90 = torch.ops.aten._to_copy.default(mul_103, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  mul_103 = None
    all_to_all_single_18 = torch.ops._c10d_functional.all_to_all_single.default(_to_copy_90, [22, 23], [22, 26], '8');  _to_copy_90 = None
    wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_18);  all_to_all_single_18 = None
    slice_backward_3 = torch.ops.aten.slice_backward.default(wait_tensor_22, [46, 32], 0, 0, -1, 1);  wait_tensor_22 = None
    index_11 = torch.ops.aten.index.Tensor(slice_backward_3, [full]);  slice_backward_3 = None
    transpose_49 = torch.ops.aten.transpose.int(transpose_6, -2, -1);  transpose_6 = None
    _grouped_mm_21 = torch.ops.aten._grouped_mm.default(index_11, transpose_49, cumsum_2);  index_11 = transpose_49 = None
    mul_104 = torch.ops.aten.mul.Tensor(_grouped_mm_21, silu);  silu = None
    mul_105 = torch.ops.aten.mul.Tensor(_grouped_mm_21, _grouped_mm_1);  _grouped_mm_21 = _grouped_mm_1 = None
    transpose_50 = torch.ops.aten.transpose.int(transpose_5, -2, -1);  transpose_5 = None
    _grouped_mm_22 = torch.ops.aten._grouped_mm.default(mul_104, transpose_50, cumsum_2);  mul_104 = transpose_50 = None
    sigmoid_7 = torch.ops.aten.sigmoid.default(_grouped_mm)
    empty_like_7 = torch.ops.aten.empty_like.default(sigmoid_7, memory_format = torch.preserve_format)
    fill__7 = torch.ops.aten.fill_.Scalar(empty_like_7, 1);  empty_like_7 = None
    sub__7 = torch.ops.aten.sub_.Tensor(fill__7, sigmoid_7);  fill__7 = None
    mul_106 = torch.ops.aten.mul.Tensor(_grouped_mm, sub__7);  _grouped_mm = sub__7 = None
    add_45 = torch.ops.aten.add.Scalar(mul_106, 1);  mul_106 = None
    mul_107 = torch.ops.aten.mul.Tensor(sigmoid_7, add_45);  sigmoid_7 = add_45 = None
    mul_108 = torch.ops.aten.mul.Tensor(mul_105, mul_107);  mul_105 = mul_107 = None
    transpose_51 = torch.ops.aten.transpose.int(transpose_4, -2, -1);  transpose_4 = None
    _grouped_mm_23 = torch.ops.aten._grouped_mm.default(mul_108, transpose_51, cumsum_2);  mul_108 = transpose_51 = cumsum_2 = None
    add_46 = torch.ops.aten.add.Tensor(_grouped_mm_22, _grouped_mm_23);  _grouped_mm_22 = _grouped_mm_23 = None
    new_zeros_16 = torch.ops.aten.new_zeros.default(add_46, [46, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_6 = torch.ops.aten.index_put.default(new_zeros_16, [full], add_46, True);  new_zeros_16 = full = add_46 = None
    slice_23 = torch.ops.aten.slice.Tensor(index_put_6, 0, 0, 45)
    slice_24 = torch.ops.aten.slice.Tensor(index_put_6, 0, 45, 46);  index_put_6 = slice_24 = None
    all_to_all_single_19 = torch.ops._c10d_functional.all_to_all_single.default(slice_23, [22, 26], [22, 23], '8');  slice_23 = None
    wait_tensor_23 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_19);  all_to_all_single_19 = None
    new_zeros_17 = torch.ops.aten.new_zeros.default(wait_tensor_23, [16, 32])
    scatter_add_10 = torch.ops.aten.scatter_add.default(new_zeros_17, 0, expand_1, wait_tensor_23);  new_zeros_17 = expand_1 = wait_tensor_23 = None
    add_47 = torch.ops.aten.add.Tensor(add_44, scatter_add_10);  add_44 = scatter_add_10 = None
    new_zeros_18 = torch.ops.aten.new_zeros.default(view_182, [48], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2))
    index_put_7 = torch.ops.aten.index_put.default(new_zeros_18, [getitem_13], view_182, True);  new_zeros_18 = getitem_13 = view_182 = None
    view_183 = torch.ops.aten.view.default(index_put_7, [16, 3]);  index_put_7 = None
    mul_109 = torch.ops.aten.mul.Tensor(view_183, 1.0);  view_183 = None
    new_zeros_19 = torch.ops.aten.new_zeros.default(mul_109, [16, 8])
    scatter_add_11 = torch.ops.aten.scatter_add.default(new_zeros_19, 1, getitem_11, mul_109);  new_zeros_19 = getitem_11 = mul_109 = None
    detach_35 = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    _softmax_backward_data_3 = torch.ops.aten._softmax_backward_data.default(scatter_add_11, detach_35, 1, torch.float32);  scatter_add_11 = detach_35 = None
    _to_copy_91 = torch.ops.aten._to_copy.default(_softmax_backward_data_3, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  _softmax_backward_data_3 = None
    t_59 = torch.ops.aten.t.default(t_4);  t_4 = None
    mm_59 = torch.ops.aten.mm.default(_to_copy_91, t_59);  _to_copy_91 = t_59 = None
    add_48 = torch.ops.aten.add.Tensor(add_47, mm_59);  add_47 = mm_59 = None
    view_184 = torch.ops.aten.view.default(add_48, [1, 16, 32]);  add_48 = None
    detach_36 = torch.ops.aten.detach.default(detach_3);  detach_3 = None
    _param_constant6_1 = self._param_constant6
    _fused_rms_norm_backward_9 = torch.ops.aten._fused_rms_norm_backward.default(view_184, add, [32], detach_36, _param_constant6_1, [True, False]);  view_184 = add = detach_36 = _param_constant6_1 = None
    getitem_86 = _fused_rms_norm_backward_9[0]
    getitem_87 = _fused_rms_norm_backward_9[1];  _fused_rms_norm_backward_9 = getitem_87 = None
    add_49 = torch.ops.aten.add.Tensor(add_42, getitem_86);  add_42 = getitem_86 = None
    view_185 = torch.ops.aten.view.default(add_49, [16, 32])
    t_60 = torch.ops.aten.t.default(t_3);  t_3 = None
    mm_60 = torch.ops.aten.mm.default(view_185, t_60);  view_185 = t_60 = None
    view_186 = torch.ops.aten.view.default(mm_60, [1, 16, 2048]);  mm_60 = None
    view_187 = torch.ops.aten.view.default(view_186, [1, 16, 16, 128]);  view_186 = None
    transpose_52 = torch.ops.aten.transpose.int(view_187, 1, 2);  view_187 = None
    detach_37 = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(transpose_52, transpose, transpose_1, transpose_2, None, detach_37, getitem_7, getitem_8, getitem_9, 0.0, [True, True, True, False], True, scale = 0.07216878364870322);  transpose_52 = transpose = transpose_1 = transpose_2 = detach_37 = getitem_7 = getitem_8 = getitem_9 = None
    getitem_88 = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_89 = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_90 = _scaled_dot_product_efficient_attention_backward_3[2]
    getitem_91 = _scaled_dot_product_efficient_attention_backward_3[3];  _scaled_dot_product_efficient_attention_backward_3 = getitem_91 = None
    transpose_53 = torch.ops.aten.transpose.int(getitem_90, 1, 2);  getitem_90 = None
    transpose_54 = torch.ops.aten.transpose.int(getitem_89, 1, 2);  getitem_89 = None
    transpose_55 = torch.ops.aten.transpose.int(getitem_88, 1, 2);  getitem_88 = None
    slice_25 = torch.ops.aten.slice.Tensor(transpose_54, 3, 0, 128)
    slice_26 = torch.ops.aten.slice.Tensor(transpose_54, 3, 128, 192);  transpose_54 = None
    sum_20 = torch.ops.aten.sum.dim_IntList(slice_26, [2], True);  slice_26 = None
    cat_21 = torch.ops.aten.cat.default([slice_25, transpose_53], 3);  slice_25 = transpose_53 = None
    view_188 = torch.ops.aten.view.default(cat_21, [1, 16, 4096]);  cat_21 = None
    view_189 = torch.ops.aten.view.default(view_188, [16, 4096]);  view_188 = None
    t_61 = torch.ops.aten.t.default(t_2);  t_2 = None
    mm_61 = torch.ops.aten.mm.default(view_189, t_61);  view_189 = t_61 = None
    view_190 = torch.ops.aten.view.default(mm_61, [1, 16, 512]);  mm_61 = None
    detach_38 = torch.ops.aten.detach.default(detach_1);  detach_1 = None
    _param_constant3_1 = self._param_constant3
    _fused_rms_norm_backward_10 = torch.ops.aten._fused_rms_norm_backward.default(view_190, getitem_2, [512], detach_38, _param_constant3_1, [True, False]);  view_190 = getitem_2 = detach_38 = _param_constant3_1 = None
    getitem_92 = _fused_rms_norm_backward_10[0]
    getitem_93 = _fused_rms_norm_backward_10[1];  _fused_rms_norm_backward_10 = getitem_93 = None
    _to_copy_92 = torch.ops.aten._to_copy.default(sum_20, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  sum_20 = None
    view_191 = torch.ops.aten.view.default(_to_copy_92, [1, 16, 1, 32, 2]);  _to_copy_92 = None
    view_as_complex_14 = torch.ops.aten.view_as_complex.default(view_191);  view_191 = None
    _conj_6 = torch.ops.aten._conj.default(view_8);  view_8 = None
    clone_6 = torch.ops.aten.clone.default(_conj_6);  _conj_6 = None
    mul_110 = torch.ops.aten.mul.Tensor(view_as_complex_14, clone_6);  view_as_complex_14 = clone_6 = None
    view_as_real_14 = torch.ops.aten.view_as_real.default(mul_110);  mul_110 = None
    view_192 = torch.ops.aten.view.default(view_as_real_14, [1, 16, 1, 64]);  view_as_real_14 = None
    _to_copy_93 = torch.ops.aten._to_copy.default(view_192, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_192 = None
    squeeze_3 = torch.ops.aten.squeeze.dim(_to_copy_93, 2);  _to_copy_93 = None
    cat_22 = torch.ops.aten.cat.default([getitem_92, squeeze_3], 2);  getitem_92 = squeeze_3 = None
    view_193 = torch.ops.aten.view.default(cat_22, [16, 576]);  cat_22 = None
    t_62 = torch.ops.aten.t.default(t_1);  t_1 = None
    mm_62 = torch.ops.aten.mm.default(view_193, t_62);  view_193 = t_62 = None
    view_194 = torch.ops.aten.view.default(mm_62, [1, 16, 32]);  mm_62 = None
    slice_27 = torch.ops.aten.slice.Tensor(transpose_55, 3, 0, 128)
    slice_28 = torch.ops.aten.slice.Tensor(transpose_55, 3, 128, 192);  transpose_55 = None
    _to_copy_94 = torch.ops.aten._to_copy.default(slice_28, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=2));  slice_28 = None
    view_195 = torch.ops.aten.view.default(_to_copy_94, [1, 16, 16, 32, 2]);  _to_copy_94 = None
    view_as_complex_15 = torch.ops.aten.view_as_complex.default(view_195);  view_195 = None
    _conj_7 = torch.ops.aten._conj.default(view_4);  view_4 = None
    clone_7 = torch.ops.aten.clone.default(_conj_7);  _conj_7 = None
    mul_111 = torch.ops.aten.mul.Tensor(view_as_complex_15, clone_7);  view_as_complex_15 = clone_7 = None
    view_as_real_15 = torch.ops.aten.view_as_real.default(mul_111);  mul_111 = None
    view_196 = torch.ops.aten.view.default(view_as_real_15, [1, 16, 16, 64]);  view_as_real_15 = None
    _to_copy_95 = torch.ops.aten._to_copy.default(view_196, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=2));  view_196 = None
    cat_23 = torch.ops.aten.cat.default([slice_27, _to_copy_95], 3);  slice_27 = _to_copy_95 = None
    view_197 = torch.ops.aten.view.default(cat_23, [1, 16, 3072]);  cat_23 = None
    view_198 = torch.ops.aten.view.default(view_197, [16, 3072]);  view_197 = None
    t_63 = torch.ops.aten.t.default(t);  t = None
    mm_63 = torch.ops.aten.mm.default(view_198, t_63);  view_198 = t_63 = None
    view_199 = torch.ops.aten.view.default(mm_63, [1, 16, 32]);  mm_63 = None
    add_50 = torch.ops.aten.add.Tensor(view_194, view_199);  view_194 = view_199 = None
    detach_39 = torch.ops.aten.detach.default(detach);  detach = None
    _param_constant0_1 = self._param_constant0
    _fused_rms_norm_backward_11 = torch.ops.aten._fused_rms_norm_backward.default(add_50, view, [32], detach_39, _param_constant0_1, [True, False]);  add_50 = view = detach_39 = _param_constant0_1 = None
    getitem_94 = _fused_rms_norm_backward_11[0]
    getitem_95 = _fused_rms_norm_backward_11[1];  _fused_rms_norm_backward_11 = getitem_95 = None
    add_51 = torch.ops.aten.add.Tensor(add_49, getitem_94);  add_49 = getitem_94 = add_51 = None
    return pytree.tree_unflatten([add_15], self._out_spec)
    