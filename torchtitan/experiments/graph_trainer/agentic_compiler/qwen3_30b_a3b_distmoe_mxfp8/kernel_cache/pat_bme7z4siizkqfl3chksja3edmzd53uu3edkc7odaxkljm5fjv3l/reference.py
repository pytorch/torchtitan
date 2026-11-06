import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32);  in0 = None
        v4 = torch.ops.aten._log_softmax.default(v3, 1, False);  v3 = None
        v5 = torch.ops.aten.nll_loss_forward.default(v4, in1, None, 2, -100);  in1 = None
        v6 = v5[0]
        v7 = v5[1];  v5 = None
        v8 = torch.ops.aten.div.Tensor(v6, in2);  v6 = in2 = None
        return (v4, v7, v8)


def run(*args):
    return Reference()(*args)
