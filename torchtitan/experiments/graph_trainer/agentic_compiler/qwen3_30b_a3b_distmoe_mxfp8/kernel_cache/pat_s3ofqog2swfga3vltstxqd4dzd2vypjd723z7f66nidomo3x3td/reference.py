from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
        v13 = torch.ops.aten.masked_fill.Scalar(in0, in1, 0);  in0 = in1 = None
        v14 = torch.ops.aten.ones_like.default(in2, pin_memory = False);  in2 = None
        v15 = torch.ops.aten.mul.Tensor(v14, in3);  v14 = in3 = None
        v16 = torch.ops.aten.expand.default(v15, [128]);  v15 = None
        v17 = torch.ops.aten.mul.Tensor(v16, in4);  v16 = in4 = None
        v18 = torch.ops.aten.unsqueeze.default(v17, 0);  v17 = None
        v19 = torch.ops.aten.expand.default(v18, [32768, 128]);  v18 = None
        v20 = torch.ops.aten.mul.Tensor(v19, in5);  v19 = in5 = None
        v21 = torch.ops.aten.neg.default(v20)
        v22 = torch.ops.aten.div.Tensor(in6, in7)
        v23 = torch.ops.aten.div.Tensor(v22, in7);  v22 = None
        v24 = torch.ops.aten.mul.Tensor(v21, v23);  v21 = v23 = None
        v25 = torch.ops.aten.div.Tensor(v20, in7);  v20 = in7 = None
        v26 = torch.ops.aten.sum.dim_IntList(v24, [1], True);  v24 = None
        v27 = torch.ops.aten.gt.Scalar(in8, 1e-12);  in8 = None
        v28 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
        v29 = torch.ops.aten.where.self(v27, v26, v28);  v27 = v26 = v28 = None
        v30 = torch.ops.aten.sgn.default(in6);  in6 = None
        v31 = torch.ops.aten.mul.Tensor(v30, v29);  v30 = v29 = None
        v32 = torch.ops.aten.add.Tensor(v25, v31);  v25 = v31 = None
        v33 = torch.ops.aten.mul.Tensor(v13, 1.0);  v13 = None
        v34 = torch.ops.aten.neg.default(v33)
        v35 = torch.ops.aten.div.Tensor(in9, in10);  in9 = None
        v36 = torch.ops.aten.div.Tensor(v35, in10);  v35 = None
        v37 = torch.ops.aten.mul.Tensor(v34, v36);  v34 = v36 = None
        v38 = torch.ops.aten.div.Tensor(v33, in10);  v33 = in10 = None
        v39 = torch.ops.aten.sum.dim_IntList(v37, [1], True);  v37 = None
        v40 = torch.ops.aten.expand.default(v39, [32768, 8]);  v39 = None
        v41 = torch.ops.aten.add.Tensor(v38, v40);  v38 = v40 = None
        v42 = torch.ops.aten.new_zeros.default(v41, [32768, 128])
        v43 = torch.ops.aten.scatter_add.default(v42, -1, in11, v41);  v42 = in11 = v41 = None
        v44 = torch.ops.aten.add.Tensor(v32, v43);  v32 = v43 = None
        v45 = torch.ops.aten.alias.default(in12);  in12 = None
        v46 = torch.ops.aten._softmax_backward_data.default(v44, v45, -1, torch.float32);  v44 = v45 = None
        v47 = torch.ops.aten._to_copy.default(v46, dtype = torch.bfloat16)
        v48 = torch.ops.aten.view.dtype(v47, torch.int16)
        v49 = torch.ops.aten.alias.default(v48);  v48 = None
        v50 = torch.ops.aten._to_copy.default(v49, dtype = torch.int32);  v49 = None
        v51 = torch.ops.aten.__lshift__.Scalar(v50, 16);  v50 = None
        v52 = torch.ops.aten.view.dtype(v51, torch.float32);  v51 = None
        v53 = torch.ops.aten.alias.default(v52);  v52 = None
        v54 = torch.ops.aten.sub.Tensor(v46, v53);  v46 = v53 = None
        v55 = torch.ops.aten._to_copy.default(v54, dtype = torch.bfloat16)
        v56 = torch.ops.aten.view.dtype(v55, torch.int16)
        v57 = torch.ops.aten.alias.default(v56);  v56 = None
        v58 = torch.ops.aten._to_copy.default(v57, dtype = torch.int32);  v57 = None
        v59 = torch.ops.aten.__lshift__.Scalar(v58, 16);  v58 = None
        v60 = torch.ops.aten.view.dtype(v59, torch.float32);  v59 = None
        v61 = torch.ops.aten.alias.default(v60);  v60 = None
        v62 = torch.ops.aten.sub.Tensor(v54, v61);  v54 = v61 = None
        v63 = torch.ops.aten._to_copy.default(v62, dtype = torch.bfloat16);  v62 = None
        v64 = torch.ops.aten.cat.default([v47, v55, v63], 1);  v47 = v55 = v63 = None
        v65 = torch.ops.aten.split.Tensor(v64, 128, 1)
        v66 = v65[0]
        v67 = v65[1]
        v68 = v65[2];  v65 = None
        v69 = torch.ops.aten.permute.default(v66, [1, 0]);  v66 = None
        v70 = torch.ops.aten.permute.default(v67, [1, 0]);  v67 = None
        v71 = torch.ops.aten.permute.default(v68, [1, 0]);  v68 = None
        return (v64, v69, v70, v71)


def run(*args):
    return Reference()(*args)
