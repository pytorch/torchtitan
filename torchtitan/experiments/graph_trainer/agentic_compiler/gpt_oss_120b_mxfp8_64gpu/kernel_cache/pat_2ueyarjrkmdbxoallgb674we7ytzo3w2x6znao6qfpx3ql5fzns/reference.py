import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.new_empty.default(in0, [in1, 2880], pin_memory = False);  in1 = None
        v4 = torch.ops.aten.index_put_.default(v3, [in2], in0);  v3 = in2 = in0 = None
        return v4


def run(*args):
    return Reference()(*args)
