"""NPU execution for Dist-MoE routed experts.

The upstream package is a CUDA SM100 kernel. On Ascend this path keeps that
module boundary and runs expert GEMMs with npu_grouped_matmul plus npu_swiglu.
Token exchange calls AllToAllTokenDispatcher._dispatch and _combine directly.
The public methods open recompute remat regions, which cannot nest inside the
saved dist_moe region.
"""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_npu

from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    if type(tensor) is torch.Tensor:
        return tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _grouped_mm(
    rows: torch.Tensor,
    weight_out_in: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute rows @ weight.transpose(-1, -2), grouped by offsets."""
    rows_bf = rows if rows.dtype == torch.bfloat16 else rows.bfloat16()
    weight_in_out = weight_out_in.transpose(-1, -2)
    weight_bf = (
        weight_in_out
        if weight_in_out.dtype == torch.bfloat16
        else weight_in_out.bfloat16()
    )
    local_rows = _local(rows_bf).contiguous()
    local_weight = _local(weight_bf).contiguous()
    local_offsets = _local(offsets).to(device=local_rows.device, dtype=torch.int64)
    if (
        type(local_rows) is torch.Tensor
        and type(local_weight) is torch.Tensor
        and local_rows.device.type == "npu"
    ):
        return torch_npu.npu_grouped_matmul(
            [local_rows],
            [local_weight],
            group_list=local_offsets,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
    return torch._grouped_mm(
        rows_bf,
        weight_bf,
        offs=offsets.to(dtype=torch.int32),
    )


def _activate(module, gate_up: torch.Tensor) -> torch.Tensor:
    if module.activation == "swiglu":
        if type(_local(gate_up)) is torch.Tensor and gate_up.device.type == "npu":
            return torch_npu.npu_swiglu(_local(gate_up).contiguous(), dim=-1)
        hidden = gate_up.shape[-1] // 2
        gate, up = gate_up.split(hidden, dim=-1)
        return F.silu(gate) * up
    hidden = gate_up.shape[-1] // 2
    gate, up = gate_up.split(hidden, dim=-1)
    limit = module.swiglu_limit
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    silu = gate * torch.sigmoid(module.swiglu_alpha * gate)
    return torch.addcmul(silu, silu, up)


def _dispatcher(module) -> AllToAllTokenDispatcher:
    dispatcher = getattr(module, "_npu_dispatcher", None)
    if dispatcher is None:
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=module.num_experts,
                top_k=module.top_k,
            )
        )
        object.__setattr__(module, "_npu_dispatcher", dispatcher)
    return dispatcher


def _exchange(dispatcher, x_TD, topk_scores_TK, topk_expert_ids_TK, counts):
    """Run the stock all-to-all without opening nested remat regions.

    This forward already sits in the saved ``dist_moe`` region. The public
    dispatcher methods open ``dispatch`` and ``combine`` regions that default
    to recompute, and a recompute region cannot nest inside a saved one.
    """
    if dispatcher.ep_mesh is None:
        return dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, counts
        )
    if spmd.is_type_checking():
        spmd.mutate_type(
            counts,
            src=spmd.P,
            dst={"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        )
    (
        routed_input,
        local_counts,
        token_indices,
        scores_sorted,
        permuted_indices,
        input_splits,
        output_splits,
    ) = dispatcher._dispatch(
        x_TD, topk_scores_TK, topk_expert_ids_TK, counts
    )
    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()
    metadata = AllToAllDispatchMetadata(
        token_indices_experts_sorted_N=token_indices,
        topk_scores_experts_sorted_N=scores_sorted,
        input_shape=torch.Size((sum(output_splits_list), x_TD.shape[-1])),
        permuted_indices=permuted_indices,
        input_splits=input_splits_list,
        output_splits=output_splits_list,
    )
    return routed_input, local_counts, metadata


def _combine(dispatcher, routed_output, metadata, x_TD):
    if dispatcher.ep_mesh is None:
        return dispatcher.combine(routed_output, metadata, x_TD)
    return dispatcher._combine(
        routed_output,
        metadata.topk_scores_experts_sorted_N,
        metadata.token_indices_experts_sorted_N,
        metadata.permuted_indices,
        metadata.input_shape,
        metadata.input_splits,
        metadata.output_splits,
        x_TD,
    )


def npu_dist_moe_forward(
    module,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,
) -> torch.Tensor:
    dispatcher = _dispatcher(module)
    routed_input, counts, metadata = _exchange(
        dispatcher,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    )
    offsets = torch.cumsum(counts, dim=0)
    w13 = module.w13.weight.flatten(1, -2)
    gate_up = _grouped_mm(routed_input, w13, offsets)
    hidden = _activate(module, gate_up)
    routed_output = _grouped_mm(hidden, module.w2.weight, offsets)
    if routed_output.dtype != routed_input.dtype:
        routed_output = routed_output.to(routed_input.dtype)
    if module.output_postprocess is not None:
        routed_output = module.output_postprocess(routed_output)
    return _combine(dispatcher, routed_output, metadata, x_TD)
