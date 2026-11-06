import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3):
        v4 = torch.ops.aten.new_zeros.default(in0, [1], pin_memory = False)
        v5 = torch.ops.aten.cat.default([v4, in0]);  v4 = None
        v6 = torch.ops.aten.slice.Tensor(v5, -1, 1, 9)
        v7 = torch.ops.aten.slice.Tensor(v5, -1, 0, 8);  v5 = None
        v8 = torch.ops.aten.sub.Tensor(v6, v7);  v6 = v7 = None
        v9 = torch.ops.aten.select.int(in0, 0, -1);  in0 = None
        v10 = torch.ops.aten.rsub.Scalar(v9, in1);  v9 = None
        v11 = torch.ops.aten.unsqueeze.default(v10, 0);  v10 = None
        v12 = torch.ops.aten.new_zeros.default(in2, [1, 2880], pin_memory = False)
        v13 = torch.ops.aten.cat.default([in2, v12]);  in2 = v12 = None
        v14 = torch.ops.aten.cat.default([v8, v11]);  v8 = v11 = None
        v15 = torch.ops.aten._to_copy.default(v14, dtype = torch.int64);  v14 = None
        v16 = torch.ops.aten.repeat_interleave.Tensor(v15, output_size = in1);  v15 = in1 = None
        v17 = torch.ops.aten.index_select.default(v13, 0, v16);  v13 = None
        v18 = torch.ops.aten.add.Tensor(in3, v17);  in3 = v17 = None
        return (v16, v18)


def run(*args):
    return Reference()(*args)
