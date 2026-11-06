import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3):
        v4 = torch.ops.aten.reshape.default(in0, [in1, 2, 2880]);  in0 = None
        v5 = torch.ops.aten.reshape.default(in2, [8, 5760]);  in2 = None
        v6 = torch.ops.aten.new_zeros.default(in3, [1], pin_memory = False)
        v7 = torch.ops.aten.cat.default([v6, in3]);  v6 = None
        v8 = torch.ops.aten.slice.Tensor(v7, -1, 1, 9)
        v9 = torch.ops.aten.slice.Tensor(v7, -1, 0, 8);  v7 = None
        v10 = torch.ops.aten.sub.Tensor(v8, v9);  v8 = v9 = None
        v11 = torch.ops.aten.select.int(in3, 0, -1);  in3 = None
        v12 = torch.ops.aten.rsub.Scalar(v11, in1);  v11 = None
        v13 = torch.ops.aten.unsqueeze.default(v12, 0);  v12 = None
        v14 = torch.ops.aten.new_zeros.default(v5, [1, 5760], pin_memory = False)
        v15 = torch.ops.aten.cat.default([v5, v14]);  v5 = v14 = None
        v16 = torch.ops.aten.cat.default([v10, v13]);  v10 = v13 = None
        v17 = torch.ops.aten._to_copy.default(v16, dtype = torch.int64);  v16 = None
        v18 = torch.ops.aten.repeat_interleave.Tensor(v17, output_size = in1);  v17 = None
        v19 = torch.ops.aten.index_select.default(v15, 0, v18);  v15 = None
        v20 = torch.ops.aten.reshape.default(v19, [in1, 2, 2880]);  v19 = in1 = None
        v21 = torch.ops.aten.add.Tensor(v4, v20);  v4 = v20 = None
        return (v18, v21)


def run(*args):
    return Reference()(*args)
