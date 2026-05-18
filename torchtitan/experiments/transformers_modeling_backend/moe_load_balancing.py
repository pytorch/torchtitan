# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate


class _HFMoeLoadBalancingHandle(nn.Module):
    """Adapter exposing the native MoE load-balancing contract on HF MoE blocks.

    Native titan's ``register_moe_load_balancing_hook`` expects each MoE
    transformer block to expose ``block.moe.{load_balance_coeff,
    tokens_per_expert, expert_bias}``.  HF models have no such module,
    so this handle is attached as ``block.moe`` and owns the buffers
    directly.  For DeepSeek-style routers that already carry an
    ``e_score_correction_bias`` buffer, ``expert_bias`` proxies to the
    gate's buffer to avoid duplication.
    """

    def __init__(self, *, num_experts: int, load_balance_coeff: float | None):
        super().__init__()
        self.load_balance_coeff = load_balance_coeff
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Set by attach_hf_moe_load_balancing for DeepSeek-style routers
        # whose e_score_correction_bias must stay on the gate module.
        self._expert_bias_ref: tuple[nn.Module, str] | None = None

    @property
    def expert_bias(self) -> torch.Tensor | None:
        if self.load_balance_coeff is None:
            return None
        if self._expert_bias_ref is not None:
            owner, name = self._expert_bias_ref
            buf = owner._buffers[name]
            # Under TP the gate buffer becomes a DTensor, but the optimizer
            # hook computes expert_bias_delta as a plain tensor.  Return the
            # local view so in-place add_() works and propagates back to the
            # DTensor (to_local() shares storage).
            if isinstance(buf, DTensor):
                return buf.to_local()
            return buf
        # Qwen-style: buffer is owned by this module under a private name
        # to avoid colliding with this property descriptor.
        return self._expert_bias


def _resolve_num_experts(moe_block: nn.Module, gate: nn.Module) -> int:
    """Infer the total expert count from the HF MoE block or gate module."""
    for owner in (moe_block.experts, gate, moe_block):
        for attr in ("num_experts", "n_routed_experts"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    raise ValueError(
        f"Could not determine number of experts for HF MoE block {type(moe_block)}"
    )


def _is_qwen_style_router(gate: nn.Module) -> bool:
    return all(hasattr(gate, attr) for attr in ("hidden_dim", "top_k", "weight"))


def _match_expert_bias_placement(expert_bias, scores):
    """Redistribute or unwrap ``expert_bias`` to match ``scores``'s DTensor/local state."""
    if isinstance(scores, DTensor):
        placements = [Replicate()] * scores.device_mesh.ndim
        if isinstance(expert_bias, DTensor):
            return expert_bias.redistribute(
                device_mesh=scores.device_mesh,
                placements=placements,
            )
        return distribute_tensor(expert_bias, scores.device_mesh, placements)
    if isinstance(expert_bias, DTensor):
        return expert_bias.to_local()
    return expert_bias


def _install_qwen_bias_routing(
    gate: nn.Module, handle: _HFMoeLoadBalancingHandle
) -> None:
    """Register a post-hook on ``gate`` that re-routes using ``handle.expert_bias``.

    Runs after the original Qwen router forward, so all upstream logic
    (softmax, normalization, future changes) executes unmodified.  Only
    expert *selection* is adjusted; gating weights are re-gathered from
    the original un-biased scores to match the native titan contract.
    """
    if getattr(gate, "_titan_moe_load_balance_patched", False):
        return

    def hook(gate_mod, args, output):
        expert_bias = handle.expert_bias
        if expert_bias is None:
            return output

        router_logits, _original_scores, _original_indices = output
        bias = _match_expert_bias_placement(expert_bias, router_logits)
        scores_for_choice = router_logits + bias
        _, new_indices = torch.topk(scores_for_choice, gate_mod.top_k, dim=-1)

        new_scores = router_logits.gather(1, new_indices)
        if getattr(gate_mod, "norm_topk_prob", False):
            new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
        new_scores = new_scores.to(router_logits.dtype)
        return router_logits, new_scores, new_indices

    gate.register_forward_hook(hook)
    gate._titan_moe_load_balance_patched = True


def _register_tokens_per_expert_hook(
    handle: _HFMoeLoadBalancingHandle,
    experts: nn.Module,
    num_experts: int,
) -> None:
    """Register a pre-hook on ``experts`` that accumulates per-expert token counts."""
    if getattr(experts, "_titan_moe_load_balance_count_hook_registered", False):
        return

    def count_tokens_per_expert(_module, args):
        if len(args) < 2:
            return None
        top_k_index = args[1]
        if isinstance(top_k_index, DTensor):
            top_k_index = top_k_index.to_local()
        if top_k_index.numel() == 0:
            return None

        with torch.no_grad():
            selected_experts = top_k_index.reshape(-1).to(dtype=torch.long)
            if selected_experts.max() >= num_experts:
                raise RuntimeError(
                    f"Expert index {selected_experts.max().item()} >= num_experts "
                    f"{num_experts}. The counting hook must run before EP dispatch "
                    f"transforms indices to local. Ensure prepend=True keeps this "
                    f"hook ahead of distribute_module hooks."
                )
            counts = torch.bincount(selected_experts, minlength=num_experts)[
                :num_experts
            ]
            tokens_per_expert = handle.tokens_per_expert
            counts = counts.to(
                device=tokens_per_expert.device,
                dtype=tokens_per_expert.dtype,
            )
            tokens_per_expert.add_(counts)
        return None

    # prepend=True is critical: this hook must run BEFORE the EP dispatch
    # hook registered by distribute_module (which transforms global expert
    # indices to local indices). Since attach_hf_moe_load_balancing() runs
    # during model construction and distribute_module runs during
    # parallelization (later), prepend=True keeps this hook at the front.
    experts.register_forward_pre_hook(
        count_tokens_per_expert,
        prepend=True,
    )
    experts._titan_moe_load_balance_count_hook_registered = True


def attach_hf_moe_load_balancing(
    transformer_block: nn.Module,
    *,
    load_balance_coeff: float | None,
) -> None:
    """Wire an HF MoE transformer block into titan's load-balancing protocol.

    Registers ``tokens_per_expert`` / ``expert_bias`` buffers and hooks so
    that ``register_moe_load_balancing_hook`` (the optimizer-step pre-hook)
    can read counts, update biases, and reset counters — the same contract
    native titan MoE modules expose via ``block.moe``.

    Supported routers:
      - DeepSeek-style (reuses existing ``e_score_correction_bias`` buffer)
      - Qwen-style (adds an ``expert_bias`` buffer and a routing post-hook)

    Must be called **before** parallelization so that the token-counting
    pre-hook (registered with ``prepend=True``) runs ahead of EP dispatch.
    """
    if load_balance_coeff is not None and load_balance_coeff <= 0.0:
        raise ValueError(
            "load_balance_coeff must be positive when MoE load balancing is enabled"
        )

    moe_block = transformer_block.mlp
    gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    if gate is None:
        raise ValueError(f"HF MoE block {type(moe_block)} does not have a gate/router")

    num_experts = _resolve_num_experts(moe_block, gate)
    handle = _HFMoeLoadBalancingHandle(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
    )

    if load_balance_coeff is not None:
        if "e_score_correction_bias" in gate._buffers:
            handle._expert_bias_ref = (gate, "e_score_correction_bias")
        else:
            if not _is_qwen_style_router(gate):
                raise ValueError(
                    "HF MoE load balancing only supports routers with "
                    "DeepSeek-style e_score_correction_bias or Qwen-style "
                    "hidden_dim/top_k/weight attributes. Set "
                    "load_balance_coeff=None to disable load balancing for "
                    f"unsupported router {type(gate)}."
                )
            handle.register_buffer(
                "_expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
            _install_qwen_bias_routing(gate, handle)

    _register_tokens_per_expert_hook(handle, moe_block.experts, num_experts)
    transformer_block.moe = handle
