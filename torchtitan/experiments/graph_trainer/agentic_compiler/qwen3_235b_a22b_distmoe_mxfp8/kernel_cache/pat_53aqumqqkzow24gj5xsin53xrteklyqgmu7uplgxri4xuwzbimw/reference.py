from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5):
        v6 = torch.ops.aten.div.Tensor(in0, in1);  in0 = in1 = None
        v7 = torch.ops.aten.nll_loss_backward.default(v6, in2, in3, None, 2, -100, in4);  v6 = in2 = in3 = in4 = None
        v8 = torch.ops.aten.alias.default(in5);  in5 = None
        v9 = torch.ops.aten._log_softmax_backward_data.default(v7, v8, 1, torch.float32);  v7 = v8 = None
        v10 = torch.ops.aten._to_copy.default(v9, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v9 = None
        return v10


def run(*args):
    return Reference()(*args)
