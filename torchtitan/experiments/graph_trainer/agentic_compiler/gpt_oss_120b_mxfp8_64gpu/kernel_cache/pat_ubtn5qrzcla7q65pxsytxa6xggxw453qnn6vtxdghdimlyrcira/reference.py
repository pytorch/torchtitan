import torch
import torchtitan.ops.scatter_add  # torchtitan::deterministic_scatter_add


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        v5 = torch.ops.aten.reshape.default(in0, [-1]);  in0 = None
        v6 = torch.ops.aten.index.Tensor(v5, [in1]);  v5 = in1 = None
        v7 = torch.ops.aten.reshape.default(v6, [-1, 1]);  v6 = None
        v8 = torch.ops.aten.mul.Tensor(in2, v7);  in2 = None
        v9 = torch.ops.aten._to_copy.default(v8, dtype = torch.bfloat16);  v8 = None
        v10 = torch.ops.aten.zeros_like.default(in3, pin_memory = False);  in3 = None
        v11 = torch.ops.torchtitan.deterministic_scatter_add.default(v10, in4, v9);  v10 = in4 = v9 = None
        return (v7, v11)


def run(*args):
    return Reference()(*args)
