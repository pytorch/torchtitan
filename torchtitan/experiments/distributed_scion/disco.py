# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from collections import defaultdict
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard
from torch.profiler import record_function

from .abstract_disco import AbstractDiSCO
from .norm_helper import calculate_norm
from .utils import remove_orig_mod_and_weight_for_p_name

__all__ = [
    "Disco",
]

logger = logging.getLogger(__name__)

# these variables are hardcoded for now, taken from torchtitan
CONST_NAME_OF_EMBEDDING = "tok_embeddings"

# Environment variables and default values
# DISCO_DEBUG_MODE = "0"


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2
    Unknown = 3


def get_param_type(p, fsdp_enabled, expert_enabled):
    """
    We can aggressively assume that the param is FSDP-Sharded
    """
    # if p.grad is None:
    #     return ParamType.Unknown
    if p.numel() == 1:
        # treat scalars separately in _build_param_lists(); return DDP here by default
        return ParamType.DDP
    if p.ndim == 3 and (expert_enabled or fsdp_enabled):
        # in torchtitan, one EP is enabled, FSDP is automatically enabled
        # here just for compatibility
        return ParamType.Expert
    if fsdp_enabled:
        return ParamType.FSDP
    return ParamType.DDP


def tp_axis(placements: tuple, tp_enabled: bool = False) -> int | None:
    """
    Return the index in `placements` that belongs to *tensor-parallel* (TP).

    Heuristics (PyTorch-TP default layouts):
      1. Row-parallel weights ⇒ `_StridedShard`  ⟶ that axis is TP.
      2. Col-parallel weights ⇒ `Shard(dim != 0)` ⟶ that axis is TP
         (FSDP shards dim-0, so a non-zero dim means TP).
    """
    # rule 1 – row-parallel
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            return i

    # rule 2 – col-parallel
    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim != 0:
            return i

    # this is a special case, We do TP only
    if tp_enabled and len(placements) == 1:
        if isinstance(placements[0], Shard):
            return 0
    return None  # could not infer


def gather_tp_shard(tensor, tp_group, tp_world_size, original_placements):
    # TP is used, we need to gather the TP-shard params first
    tp_mesh_dim = tp_axis(original_placements, True)
    assert tp_mesh_dim is not None, "TP mesh dimension not found"
    shard_dim = original_placements[tp_mesh_dim].dim

    output_tensors = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    dist.all_gather(output_tensors, tensor, group=tp_group)
    return torch.cat(output_tensors, dim=shard_dim)


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    splits = torch.arange(full).chunk(world_size)
    if rank >= len(splits):
        dim0 = 0
    else:
        dim0 = len(splits[rank])

    return (dim0, *shape[1:])


class Disco(AbstractDiSCO):
    def __init__(
        self,
        params,
        is_light,
        weight_decay,
        lr,
        momentum,
        nesterov,
        eps,
        norm_factor,
        backend,
        backend_steps,
        parallel_dims,
        communication_dtype=torch.bfloat16,
        extra_reduce_for_HSDP=False,
        experts_weights_layout="G-D_out-D_in",
        name_of_embedding=CONST_NAME_OF_EMBEDDING,
    ):

        debug_mode = os.environ.get("DISCO_DEBUG_MODE", "0") == "1"

        # Initialize base optimizer and common state
        self.extra_reduce_for_HSDP = False
        self.log_parameters_types = True
        self.is_light = is_light

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor if not debug_mode else "none",
            backend=backend if not debug_mode else "identity",
            backend_steps=backend_steps,
        )
        assert is_light is False, "is_light must be False, its not supported yet"

        is_unconstrained = weight_decay == 0

        self.world_mesh = parallel_dims.world_mesh

        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled

        # this is used to ensure only the DP or FSDP rank 0 will have norms
        if self.dp_replicate_enabled or self.fsdp_enabled:
            self.is_dp_rank_0 = dist.get_rank(self.world_mesh["dp_cp"].get_group()) == 0
        else:
            # only PP (and/or) TP enabled
            self.is_dp_rank_0 = dist.get_rank() == 0

        assert experts_weights_layout in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ], f"Unknown experts weights layout: {experts_weights_layout}"
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"
        self.extra_reduce_for_HSDP = extra_reduce_for_HSDP

        self.name_of_embedding = name_of_embedding

        logger.info(
            f"Distributed Spectral Conditioned Optimizer "
            f"(is_light={self.is_light}, is_unconstrained={is_unconstrained}) "
            f"is enabled with world_mesh={self.world_mesh} | fsdp_enabled={self.fsdp_enabled} | "
            f"EP={self.expert_enabled} | TP={self.tp_enabled} | DP={self.dp_replicate_enabled}"
        )

        super().__init__(params, defaults, is_light=is_light)
        # Register light-mode grad state hooks if needed
        self.setup_light_state_hooks()

        self.communication_dtype = communication_dtype
        self.groups_info = {}
        self.parameters_to_groups = {}
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"] if not debug_mode else "none",
                "zeropower_backend": group["backend"] if not debug_mode else "identity",
                "backend_steps": group["backend_steps"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]
            for param in group["params"]:
                self.parameters_to_groups[id(param)] = group_idx

                # check sanity of MoE, do not allow the Shard(1) for grouped experts
                if (
                    param.ndim == 3
                    and self.expert_enabled
                    and Shard(1) in param.placements
                ):
                    raise NotImplementedError(
                        "we should now allow the Shard(1) for grouped experts"
                    )

            logger.info(
                f"group_idx: {group_idx} has {len(group['params'])} params ||"
                f"lr: {lr} || nesterov: {nesterov} || momentum: {momentum} || wd: {wd} ||"
                f"{param_kwargs}"
            )

            if self.is_light and nesterov:
                raise RuntimeError(
                    "Nesterov momentum is not supported for spectral conditioned optimizer's light mode. "
                    "Please set nesterov=False."
                )

        # public caches
        self.scale_params, self.scale_param_names = [], []
        self.embed_params, self.embed_param_names = [], []
        self.ddp_params, self.ddp_param_names = [], []
        self.fsdp_params, self.fsdp_param_names = [], []
        self.expert_params, self.expert_param_names = [], []
        # build once now
        self._build_param_lists()

    def _build_param_lists(self):
        # clear
        self.scale_params.clear()
        self.scale_param_names.clear()
        self.embed_params.clear()
        self.embed_param_names.clear()
        self.ddp_params.clear()
        self.ddp_param_names.clear()
        self.fsdp_params.clear()
        self.fsdp_param_names.clear()
        self.expert_params.clear()
        self.expert_param_names.clear()

        # decide embedding per-group exactly like in step()
        # (backend == "identity" and norm_factor startswith embed/unembed)
        def _is_embed_group(g):
            nf, be = g["norm_factor"], g["backend"]
            return (be == "identity") and (
                nf.startswith("embed") or nf.startswith("unembed")
            )

        for group in self.param_groups:
            route_to_embed = _is_embed_group(group)
            for p_name, p in zip(group["param_names"], group["params"]):
                if not p.requires_grad:
                    # ignore the non-trainable parameters
                    continue

                # 1) scalar branch identical to step()
                if p.numel() == 1:
                    assert (
                        group["backend"] == "identity"
                    ), "scale params must use identity backend"
                    assert (
                        group["norm_factor"] == "sign"
                    ), "scale params must use sign norm factor"
                    self.scale_params.append(p)
                    self.scale_param_names.append(p_name)
                    continue
                # 2) embedding fast path identical to step() predicate
                if route_to_embed:
                    self.embed_params.append(p)
                    self.embed_param_names.append(p_name)
                    continue
                # 3) structural type without reading p.grad (init-time)
                ptype = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if ptype == ParamType.DDP:
                    self.ddp_params.append(p)
                    self.ddp_param_names.append(p_name)
                elif ptype == ParamType.FSDP:
                    self.fsdp_params.append(p)
                    self.fsdp_param_names.append(p_name)
                elif ptype == ParamType.Expert:
                    self.expert_params.append(p)
                    self.expert_param_names.append(p_name)
                else:
                    # static classifier should not return Unknown
                    pass
        if self.ddp_params:
            pairs = list(zip(self.ddp_params, self.ddp_param_names))
            # sort big → small to reduce padding and make buckets well-conditioned
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)

            # snake interleave across buckets to balance per-rank Phase-A compute
            dp_group = (
                self.world_mesh["dp_replicate"].get_group()
                if self.dp_replicate_enabled
                else None
            )
            w = dp_group.size() if dp_group is not None else 1
            if w > 1:
                blocks = [pairs[i : i + w] for i in range(0, len(pairs), w)]
                for b, blk in enumerate(blocks):
                    if b % 2 == 1:
                        blk.reverse()
                pairs = [p for blk in blocks for p in blk]

            self.ddp_params, self.ddp_param_names = (
                (list(t) if pairs else [] for t in zip(*pairs)) if pairs else ([], [])
            )

        if self.fsdp_params:
            pairs = list(zip(self.fsdp_params, self.fsdp_param_names))
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)
            self.fsdp_params, self.fsdp_param_names = list(zip(*pairs))
            self.fsdp_params, self.fsdp_param_names = list(self.fsdp_params), list(
                self.fsdp_param_names
            )

        if self.expert_params:
            pairs = list(zip(self.expert_params, self.expert_param_names))
            pairs.sort(key=lambda x: (x[0].numel(), x[0].shape[1]), reverse=True)
            self.expert_params, self.expert_param_names = list(zip(*pairs))
            self.expert_params, self.expert_param_names = list(
                self.expert_params
            ), list(self.expert_param_names)

        if self.log_parameters_types:
            logger.info(
                f"fsdp_params: {len(self.fsdp_params)} | expert_params: {len(self.expert_params)} | "
                f"ddp_params: {len(self.ddp_params)} | embed_params: {len(self.embed_params)} | "
                f"scale_params: {len(self.scale_params)}"
            )
            self.log_parameters_types = False

    @record_function("disco.step")
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # only refresh schedulables each step (unchanged behaviour)
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]

        self.prepare_gradients_and_momentum()

        # If you ever flip routing (backend/norm_factor) and want new buckets, call self._build_param_lists() explicitly.
        # Otherwise, just dispatch with the cached lists

        self.step_scalar(self.scale_params, self.scale_param_names)
        self.step_embedding(self.embed_params, self.embed_param_names)
        self.step_experts(self.expert_params, self.expert_param_names)
        self.step_ddp(self.ddp_params, self.ddp_param_names)

        self.step_fsdp(self.fsdp_params, self.fsdp_param_names)

        self.need_to_calculate_norm = False
        return loss

    @record_function("disco.step_scalar")
    @torch.compile()
    def step_scalar(
        self,
        scalar_params,
        scalar_param_names,
        skip_update=False,
        apply_on_weight=True,
    ):
        """
        We hardcode the update for scalar parameters to be the `sign` of the gradient.
        """
        if not scalar_params:
            return

        updates = []
        for p in scalar_params:
            _, nesterov, momentum, _, _ = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)

            if g is None:
                updates.append(None)
                continue

            g_local = g.to_local() if isinstance(g, DTensor) else g
            u = torch.sign(g_local)
            updates.append(u)

        if not skip_update:
            # Scalar parameters are not TP-sharded, so tp_group is None.
            self.update_bucket_params(
                scalar_params, updates, 0, len(scalar_params), tp_group=None
            )

        if not self.need_to_calculate_norm:
            return

        final_norms = {}
        if apply_on_weight and self.need_to_calculate_norm:
            for i, p in enumerate(scalar_params):
                p_local = p.to_local() if isinstance(p, DTensor) else p
                cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                    scalar_param_names[i]
                )
                # The original code only logs the parameter's absolute value, as the
                # update norm is constant (learning_rate * 1.0).
                final_norms[f"scalar_param_supremum/{cleaned_p_name}"] = p_local.abs()

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_embedding")
    def step_embedding(
        self, embed_params, embed_param_names, skip_update=False, apply_on_weight=True
    ):
        if len(embed_params) == 0:
            return

        tp_group = self.world_mesh["tp"].get_group() if self.tp_enabled else None

        # --- Phase 1: Parameter Update (Efficient, on shards) ---
        # This part of your refactor is efficient and correct for the update.
        # It computes updates on shards and applies them locally.
        effective_grads = []
        for p in embed_params:
            _, nesterov, momentum, _, _ = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            # Get gradient without gathering for efficiency
            g = self.get_momentum_or_grad(p, momentum, nesterov, gather_to_local=False)

            if not self.fsdp_enabled and self.tp_enabled:
                # here is an edge case where [e.g. GPUS=TP, or GPUS=PP x TP]
                # model weight is sharded, but gradient is Replicate
                original_placements = p.placements
                tp_mesh_dim = tp_axis(original_placements, True)
                tp_sharded_dim = original_placements[tp_mesh_dim].dim
                chunk_size = p.to_local().shape[tp_sharded_dim]
                start_offset = tp_group.rank() * chunk_size
                slicer = [slice(None)] * g.dim()
                slicer[tp_sharded_dim] = slice(start_offset, start_offset + chunk_size)
                g = g[tuple(slicer)]

            effective_grads.append(g)

        # LMO bucketed by param_kwargs
        lmo_buckets = defaultdict(lambda: {"grads": [], "indices": []})
        for i, g in enumerate(effective_grads):
            if g is not None:
                p = embed_params[i]
                *_, param_kwargs = self.groups_info[self.parameters_to_groups[id(p)]]
                kwargs_key = tuple(sorted(param_kwargs.items()))
                lmo_buckets[kwargs_key]["grads"].append(g)
                lmo_buckets[kwargs_key]["indices"].append(i)

        updates = [None] * len(embed_params)
        for kwargs_key, data in lmo_buckets.items():
            param_kwargs = dict(kwargs_key)
            for j, g in enumerate(data["grads"]):
                updates[data["indices"][j]] = self.lmo(g, **param_kwargs)

        if not skip_update:
            self.update_bucket_params(
                embed_params, updates, 0, len(embed_params), tp_group=None
            )

        # --- Phase 2: Norm Calculation (On Full Tensors for Correctness) ---
        if not self.need_to_calculate_norm:
            return
        final_norms = {}
        apply_on_weight = apply_on_weight and self.need_to_calculate_norm

        for i, (p, p_name) in enumerate(zip(embed_params, embed_param_names)):
            lr, nesterov, momentum, _, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]

            # Gathering Gradient to a full tensor for norm calculation
            g = self.get_momentum_or_grad(p, momentum, nesterov, gather_to_local=True)
            u = self.lmo(g, **param_kwargs)

            # Calculate norms on the full tensors
            need_T = self.name_of_embedding in p_name
            upd_norms = calculate_norm(-lr * u, self.norms_to_log, transpose=need_T)

            # Gather the parameter itself to a full tensor if needed
            if apply_on_weight and isinstance(p, DTensor):
                p = p.full_tensor()

            if apply_on_weight:
                wnorm = calculate_norm(p, self.norms_to_log, transpose=need_T)

            cleaned_p_name = remove_orig_mod_and_weight_for_p_name(p_name)
            for norm_name in self.norms_to_log:
                final_norms[f"track_update_{norm_name}/{cleaned_p_name}"] = upd_norms[
                    norm_name
                ]
                if apply_on_weight:
                    final_norms[f"track_param_{norm_name}/{cleaned_p_name}"] = wnorm[
                        norm_name
                    ]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_experts")
    def step_experts(
        self,
        expert_params,
        expert_param_names,
        skip_update=False,
        apply_on_weight=True,
    ):
        if len(expert_params) == 0:
            return

        need_to_calculate_norm = self.need_to_calculate_norm

        norms_of_update, norms_of_weight, final_norms = [], [], {}
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        device = expert_params[0].device
        fsdp_group = self.world_mesh["dp_shard_cp"].get_group()
        world_size = dist.get_world_size(fsdp_group)
        local_rank = dist.get_rank(fsdp_group)
        ep_per_rank = math.ceil(expert_params[0].shape[0] / world_size)

        kinds_of_norms = len(self.norms_to_log)

        padding_norms = torch.tensor(0.0, device=device)
        # each rank will process `len(expert_params) * ep_per_rank` experts
        # each expert will have `self.norms_to_log` norms
        # so each rank will have `len(expert_params) * ep_per_rank * len(self.norms_to_log)`
        # norms
        # globally, its [[g0-ep0, g0-ep1, g0-ep2, ...], [g1-ep0, g1-ep1, g1-ep2, ...], ...] on each
        # rank

        transpose = self.experts_need_transpose
        for param_idx in range(len(expert_params)):
            p = expert_params[param_idx]
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)
            u = self.lmo(g, **param_kwargs, transpose_experts=transpose)

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

            if need_to_calculate_norm:
                # cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                #     expert_param_names[param_idx]
                # )
                assert u.ndim == 3
                for ep_idx in range(u.shape[0]):
                    update_norms = calculate_norm(
                        u[ep_idx], self.norms_to_log, transpose=transpose
                    )
                    # Template for MoE norm keys
                    norms_of_update.extend(update_norms.values())
                    if apply_on_weight:
                        weight_norms = calculate_norm(
                            p.to_local()[ep_idx], self.norms_to_log, transpose=transpose
                        )
                        norms_of_weight.extend(weight_norms.values())

        if need_to_calculate_norm:
            expected_total = len(expert_params) * ep_per_rank * kinds_of_norms
            pad_needed = expected_total - len(norms_of_update)
            if pad_needed > 0:
                norms_of_update.extend([padding_norms] * pad_needed)
                if apply_on_weight:  # keep weight-norms aligned
                    norms_of_weight.extend([padding_norms] * pad_needed)

            norms_tensor = torch.stack(norms_of_update).float().to(device)
            gathered_update_norms = torch.empty(
                world_size * norms_tensor.shape[0],
                dtype=norms_tensor.dtype,
                device=norms_tensor.device,
            )
            dist.all_gather_into_tensor(
                gathered_update_norms, norms_tensor, group=fsdp_group
            )

            if apply_on_weight:
                norms_tensor = torch.stack(norms_of_weight).float().to(device)
                gathered_weight_norms = torch.empty(
                    world_size * norms_tensor.shape[0],
                    dtype=norms_tensor.dtype,
                    device=norms_tensor.device,
                )
                dist.barrier()
                dist.all_gather_into_tensor(
                    gathered_weight_norms, norms_tensor, group=fsdp_group
                )

            if local_rank == 0:
                norm_names = list(self.norms_to_log)

                P = len(expert_params)  # parameters per rank
                E = ep_per_rank  # experts per rank
                K = kinds_of_norms  # norms per expert
                block = P * E * K  # values contributed by each rank

                for idx in range(world_size * block):
                    r, rem = divmod(idx, block)  # producing rank
                    p, rem = divmod(rem, E * K)  # parameter index
                    e, k = divmod(rem, K)  # expert, norm indices

                    actual_ep_idx = e + r * E
                    if actual_ep_idx >= expert_params[0].shape[0]:
                        continue  # skip pure padding slots

                    cleaned_name = remove_orig_mod_and_weight_for_p_name(
                        expert_param_names[p]
                    )
                    norm_name = norm_names[k]

                    key_update = (
                        f"track_update_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                    )
                    final_norms[key_update] = gathered_update_norms[idx]

                    if apply_on_weight:
                        key_param = (
                            f"track_param_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                        )
                        final_norms[key_param] = gathered_weight_norms[idx]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_ddp")
    def step_ddp(
        self,
        ddp_params,
        ddp_param_names,
        skip_update: bool = False,
        apply_on_weight: bool = True,
    ):
        if len(ddp_params) == 0:
            return

        need_to_calculate_norm = self.need_to_calculate_norm
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        # --- distributed groups ---
        dp_group = (
            self.world_mesh["dp_replicate"].get_group()
            if self.dp_replicate_enabled
            else None
        )  # DDP group accessor
        world_size = dp_group.size() if dp_group is not None else 1
        rank = dp_group.rank() if dp_group is not None else 0

        tp_group = self.world_mesh["tp"].get_group() if self.tp_enabled else None
        tp_world_size = (
            dist.get_world_size(group=tp_group) if tp_group is not None else 1
        )

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype  # comm/exchange dtype

        # one DDP bucket spans `world_size` params
        bucket_size = world_size
        total_buckets = (
            math.ceil(len(ddp_params) / bucket_size)
            if world_size > 1
            else len(ddp_params)
        )

        # --- Initializations for norm calculation ---
        norms_of_update, norms_of_weight, final_norms = [], [], {}

        # -------- Phase A: precompute local LMO updates (no comm) --------
        local_updates: dict[int, torch.Tensor] = {}
        local_indices = range(rank, len(ddp_params), world_size)
        for i in local_indices:
            p = ddp_params[i]
            _, nesterov, momentum, _, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)  # relies on pre-pass
            if isinstance(g, DTensor) and tp_group is not None:
                g = gather_tp_shard(
                    g.to_local(), tp_group, tp_world_size, g.placements
                )  # TP tolerant
            else:
                g = g.to_local() if isinstance(g, DTensor) else g
            u = self.lmo(g.to(dtype=cast_dtype), **param_kwargs)
            local_updates[i] = u

        # -------- Phase B: DDP communication (per bucket) and build a global update cache --------
        global_updates: list[torch.Tensor | None] = [None] * len(ddp_params)

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(ddp_params))
            my_idx = start_idx + rank

            if my_idx < end_idx:
                send_u = local_updates[my_idx]
            else:
                ref = ddp_params[end_idx - 1]
                send_u = torch.zeros(ref.shape, dtype=cast_dtype, device=device)

            if dp_group is not None and world_size > 1 and not skip_update:
                gathered = []
                pad_buffer = None
                for i in range(world_size):
                    param_idx = start_idx + i
                    if param_idx < len(ddp_params):
                        ref = ddp_params[param_idx]
                        if global_updates[param_idx] is None:
                            global_updates[param_idx] = torch.empty(
                                ref.shape, dtype=cast_dtype, device=device
                            )
                        recv = global_updates[param_idx]
                    else:
                        ref = ddp_params[end_idx - 1]
                        if pad_buffer is None or pad_buffer.shape != ref.shape:
                            pad_buffer = torch.empty(
                                ref.shape, dtype=cast_dtype, device=device
                            )
                        recv = pad_buffer
                    gathered.append(recv)
                dist.all_gather(gathered, send_u, group=dp_group)
            else:
                gathered = [send_u] if not skip_update else []
            if not skip_update:
                for i in range(end_idx - start_idx):
                    global_updates[start_idx + i] = gathered[i]

            # ---- local norms (on our owned slot only) ----
            if need_to_calculate_norm:
                if my_idx < end_idx:
                    p = ddp_params[my_idx]
                    lr, *_ = self.groups_info[self.parameters_to_groups[id(p)]]
                    upd_norms = calculate_norm(
                        -lr * local_updates[my_idx], self.norms_to_log
                    )
                else:
                    upd_norms = {
                        k: torch.tensor(0.0, device=device) for k in self.norms_to_log
                    }
                for v in upd_norms.values():
                    norms_of_update.append(v)

        # -------- Phase C: apply once (vectorised foreach inside update_bucket_params) --------
        if not skip_update:
            self.update_bucket_params(
                ddp_params, global_updates, 0, len(ddp_params), tp_group=tp_group
            )

        # -------- Phase C.5: Calculate Weight Norms (POST-UPDATE) --------
        if apply_on_weight:
            for bucket_idx in range(total_buckets):
                my_idx = bucket_idx * world_size + rank
                if my_idx < len(ddp_params):
                    w = ddp_params[my_idx]
                    if isinstance(w, DTensor) and tp_group is not None:
                        w = gather_tp_shard(
                            w.to_local(), tp_group, tp_world_size, w.placements
                        )
                    w_norms = calculate_norm(w, self.norms_to_log)
                else:
                    w_norms = {
                        k: torch.tensor(0.0, device=device) for k in self.norms_to_log
                    }
                for v in w_norms.values():
                    norms_of_weight.append(v)

        # -------- Phase D: final norm gather/log --------
        if not need_to_calculate_norm:
            return

        upd = torch.stack(norms_of_update).float().to(device)
        if dp_group is not None and world_size > 1:
            gathered_upd = torch.empty(
                world_size * upd.shape[0], dtype=upd.dtype, device=device
            )
            dist.all_gather_into_tensor(gathered_upd, upd, group=dp_group)
        else:
            gathered_upd = upd

        if apply_on_weight:
            w = torch.stack(norms_of_weight).float().to(device)
            if dp_group is not None and world_size > 1:
                gathered_w = torch.empty(
                    world_size * w.shape[0], dtype=w.dtype, device=device
                )
                dist.all_gather_into_tensor(gathered_w, w, group=dp_group)
            else:
                gathered_w = w
        else:
            gathered_w = None

        if self.is_dp_rank_0:
            num_norm_types = len(self.norms_to_log)
            for param_idx, p_name in enumerate(ddp_param_names):
                cleaned = remove_orig_mod_and_weight_for_p_name(p_name)
                owner_rank = param_idx % world_size
                owner_bucket = param_idx // world_size
                base = (owner_rank * total_buckets + owner_bucket) * num_norm_types
                for k, norm_name in enumerate(self.norms_to_log):
                    idx = base + k
                    final_norms[f"track_update_{norm_name}/{cleaned}"] = gathered_upd[
                        idx
                    ]
                    if apply_on_weight:
                        final_norms[f"track_param_{norm_name}/{cleaned}"] = gathered_w[
                            idx
                        ]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    def _gather_and_log_fsdp_norms(
        self,
        norms_of_update,
        norms_of_weight,
        fsdp_group,
        rank,
        device,
        fsdp_param_names,
        world_size,
        total_buckets,
        apply_on_weight,
    ):
        """
        Gathers FSDP norm tensors from all ranks and logs them on rank 0.
        This is a collective operation followed by a rank-0 logging step.
        """
        # --- 1. Collective Communication: All ranks must participate ---
        upd = torch.stack(norms_of_update).float().to(device)
        gathered_update_norms = torch.empty(
            world_size * upd.numel(), dtype=upd.dtype, device=device
        )
        dist.all_gather_into_tensor(gathered_update_norms, upd, group=fsdp_group)

        gathered_weight_norms = None
        if apply_on_weight and norms_of_weight:
            w = torch.stack(norms_of_weight).float().to(device)
            gathered_weight_norms = torch.empty(
                world_size * w.numel(), dtype=w.dtype, device=device
            )
            dist.all_gather_into_tensor(gathered_weight_norms, w, group=fsdp_group)

        # --- 2. Local Processing: Only rank 0 processes and logs the results ---
        final_norms = {}
        if self.is_dp_rank_0:
            num_norm_types = len(self.norms_to_log)
            entries_per_rank = total_buckets * num_norm_types
            cleaned_names = [
                remove_orig_mod_and_weight_for_p_name(pn) for pn in fsdp_param_names
            ]

            for param_idx, cleaned_p_name in enumerate(cleaned_names):
                owner_rank = param_idx % world_size
                bucket_idx_on_owner = param_idx // world_size
                base = (
                    owner_rank * entries_per_rank + bucket_idx_on_owner * num_norm_types
                )

                for norm_idx, norm_name in enumerate(self.norms_to_log):
                    idx = base + norm_idx
                    final_norms[
                        f"track_update_{norm_name}/{cleaned_p_name}"
                    ] = gathered_update_norms[idx]
                    if apply_on_weight and gathered_weight_norms is not None:
                        final_norms[
                            f"track_param_{norm_name}/{cleaned_p_name}"
                        ] = gathered_weight_norms[idx]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_fsdp")
    def step_fsdp(
        self, fsdp_params, fsdp_param_names, skip_update=False, apply_on_weight=True
    ):
        if not fsdp_params:
            return

        # ---- Setup (as in your code) ----
        need_to_calculate_norm = self.need_to_calculate_norm
        apply_on_weight = apply_on_weight and need_to_calculate_norm

        fsdp_group = self.world_mesh["dp_shard_cp"].get_group()
        world_size = dist.get_world_size(fsdp_group)
        rank = dist.get_rank(fsdp_group)
        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype

        tp_group = self.world_mesh["tp"].get_group() if self.tp_enabled else None
        tp_world_size = dist.get_world_size(group=tp_group) if tp_group else 1
        dp_replicate_group = (
            self.world_mesh["dp_replicate"].get_group()
            if self.dp_replicate_enabled
            else None
        )

        global_updates = [None] * len(fsdp_params)
        norms_of_update, norms_of_weight = [], []
        padding_norms = {k: torch.tensor(0.0, device=device) for k in self.norms_to_log}

        # ---- Owner-per-bucket scheme (bucket size = world_size) ----
        total_buckets = math.ceil(len(fsdp_params) / world_size)

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * world_size
            end_idx = min(start_idx + world_size, len(fsdp_params))

            grads_send_list, send_shapes = [], []
            target_shape, param_kwargs_me = None, None

            # Build per-destination payloads:
            for i in range(world_size):
                p_idx = start_idx + i
                if p_idx < end_idx:
                    p = fsdp_params[p_idx]
                    _, nesterov, momentum, _, param_kwargs = self.groups_info[
                        self.parameters_to_groups[id(p)]
                    ]
                    g = self.get_momentum_or_grad(
                        p, momentum, nesterov
                    )  # may be DTensor

                    if isinstance(g, DTensor):
                        original_placements = g.placements
                        tp_mesh_dim = tp_axis(original_placements)
                        if tp_group and tp_mesh_dim is not None:
                            g_local = gather_tp_shard(
                                g.to_local(),
                                tp_group,
                                tp_world_size,
                                original_placements,
                            )
                        else:
                            g_local = g.to_local()
                    else:
                        g_local = g

                    grads_send_list.append(g_local.to(dtype=cast_dtype))
                    send_shapes.append(g_local.shape)

                    if i == rank:
                        target_shape = p.shape
                        param_kwargs_me = param_kwargs
                else:
                    # pad so that list length == world_size
                    ref_p = fsdp_params[end_idx - 1]
                    dummy = torch.zeros(
                        ref_p.to_local().shape, dtype=cast_dtype, device=device
                    )
                    grads_send_list.append(dummy)
                    send_shapes.append(dummy.shape)

            # Pick a reference target if this rank owns none in this bucket
            if target_shape is None:
                ref_idx = end_idx - 1
                target_shape = fsdp_params[ref_idx].shape
                param_kwargs_me = self.groups_info[
                    self.parameters_to_groups[id(fsdp_params[ref_idx])]
                ][-1]

            # Receive slices (one from each source) that together make the full grad for the owned param
            recv_shapes = [
                calculate_shard_shape(target_shape, r, world_size)
                for r in range(world_size)
            ]
            recv_list_grads = [
                torch.empty(s, dtype=cast_dtype, device=device) for s in recv_shapes
            ]

            # A2A: shards -> owners
            dist.all_to_all(
                recv_list_grads, grads_send_list, group=fsdp_group
            )  # list API

            full_g = torch.cat(recv_list_grads, dim=0)
            u = self.lmo(full_g, **param_kwargs_me)

            if dp_replicate_group and self.extra_reduce_for_HSDP:
                dist.all_reduce(u, group=dp_replicate_group, op=dist.ReduceOp.AVG)

            if not skip_update:
                # Split owner’s update by destination rows and scatter back
                split_rows = [s[0] for s in recv_shapes]
                updates_send_list = list(torch.split(u, split_rows, dim=0))
                recv_list_updates = [
                    torch.empty(s, dtype=cast_dtype, device=device) for s in send_shapes
                ]

                # A2A: owners -> shards
                dist.all_to_all(recv_list_updates, updates_send_list, group=fsdp_group)

                # Materialise bucket’s updates into global list
                for i in range(end_idx - start_idx):
                    global_updates[start_idx + i] = recv_list_updates[i]

            # optional logging of update norms
            if need_to_calculate_norm:
                my_param_in_bucket = (start_idx + rank) < end_idx
                if my_param_in_bucket:
                    p = fsdp_params[start_idx + rank]
                    lr, *_ = self.groups_info[self.parameters_to_groups[id(p)]]
                    upd_norms = calculate_norm(-lr * u, self.norms_to_log)
                else:
                    upd_norms = padding_norms
                norms_of_update.extend(upd_norms.values())

        # Single vectorised apply (as in your file)
        if not skip_update:
            self.update_bucket_params(
                fsdp_params,
                global_updates,
                0,
                len(fsdp_params),
                tp_group=self.world_mesh["tp"].get_group() if self.tp_enabled else None,
            )

        # --- Calculate Weight Norms (POST-UPDATE) ---
        if apply_on_weight:
            for bucket_idx in range(total_buckets):
                start_idx = bucket_idx * world_size
                end_idx = min(start_idx + world_size, len(fsdp_params))
                my_param_in_bucket = (start_idx + rank) < end_idx

                # Determine target shape for reconstruction, even if this rank is padding
                ref_p_idx = start_idx + rank if my_param_in_bucket else end_idx - 1
                target_shape = fsdp_params[ref_p_idx].shape

                params_send_list = []
                for i in range(world_size):
                    param_idx = start_idx + i
                    p = fsdp_params[param_idx if param_idx < end_idx else end_idx - 1]

                    original_placements = p.placements
                    tp_mesh_dim = tp_axis(original_placements)
                    if tp_group and tp_mesh_dim is not None:
                        p_local = gather_tp_shard(
                            p.to_local(), tp_group, tp_world_size, original_placements
                        )
                    else:
                        p_local = p.to_local()
                    params_send_list.append(p_local.to(dtype=cast_dtype))

                recv_shapes = [
                    calculate_shard_shape(target_shape, r, world_size)
                    for r in range(world_size)
                ]
                recv_list_params = [
                    torch.empty(s, dtype=cast_dtype, device=device) for s in recv_shapes
                ]
                dist.all_to_all(recv_list_params, params_send_list, group=fsdp_group)

                full_weight = torch.cat(recv_list_params, dim=0)

                w_norms = (
                    calculate_norm(full_weight, self.norms_to_log)
                    if my_param_in_bucket
                    else padding_norms
                )
                norms_of_weight.extend(w_norms.values())

        # --- Phase D: gather and log norms once ---

        if need_to_calculate_norm and norms_of_update:
            self._gather_and_log_fsdp_norms(
                norms_of_update,
                norms_of_weight,
                fsdp_group,
                rank,
                device,
                fsdp_param_names,
                world_size,
                total_buckets,
                apply_on_weight,
            )

    @record_function("disco._prepare_gradients_and_momentum")
    def prepare_gradients_and_momentum(self) -> None:
        """
        Fused pre-pass that updates momentum buffers for *all* parameters
        with available grads:
            buf <- (1 - m) * buf + m * g
        It performs foreach-kernel updates per (device, dtype, momentum),
        """
        buckets = defaultdict(lambda: {"bufs": [], "grads": [], "m": 0.0})

        for group in self.param_groups:
            m = float(group["momentum"])
            use_momentum = (not self.is_light) and (0.0 < m < 1.0)

            if not use_momentum:
                continue

            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is None or not p.requires_grad:
                    continue

                # Initialize the momentum buffer if it's the first time.
                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # Add the buffer and gradient to the appropriate bucket.
                key = (g.dtype, m)
                bucket = buckets[key]
                bucket["bufs"].append(buf)
                bucket["grads"].append(g)
                bucket["m"] = m

        # Launch the fused kernels for each bucket.
        for (_, m), bucket_data in buckets.items():
            if not bucket_data["bufs"]:
                continue
            # Perform the momentum update: buf = buf * (1 - m) + g * m
            torch._foreach_mul_(bucket_data["bufs"], 1.0 - m)
            torch._foreach_add_(bucket_data["bufs"], bucket_data["grads"], alpha=m)

    @record_function("disco.get_momentum_or_grad")
    def get_momentum_or_grad(self, p, momentum, nesterov, gather_to_local=False):
        """
        Retrieves the effective gradient for a parameter.
        Assumes the momentum buffer has already been updated in a pre-pass.
        """
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            state = self.state.get(p, None)
            if state is None or "momentum_buffer" not in state:
                raise ValueError(
                    "Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad."
                )
            # The buffer is already updated, so we just retrieve it.
            buf = state["momentum_buffer"]
            g = buf if not nesterov else buf.mul(1 - momentum).add(g, alpha=momentum)

        if gather_to_local and isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()

        return g

    @record_function("disco.update_bucket_params")
    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_group=None):
        slice_params = params[start_idx:end_idx]
        slice_updates = updates[: (end_idx - start_idx)]
        # already prepare a bucket of same length

        prepared = []
        if tp_group is not None:
            tp_rank = tp_group.rank()
            for p, u in zip(slice_params, slice_updates):
                if u is None:
                    prepared.append((p, None))
                    continue
                if isinstance(p, DTensor):
                    p_local = p.to_local()
                    placements = p.placements
                    tp_mesh_dim = tp_axis(placements, tp_enabled=(u.shape == p.shape))
                    if tp_mesh_dim is not None:
                        shard_dim = placements[tp_mesh_dim].dim
                        # Skip TP slicing if the update already matches the local shard.
                        if u.shape != p_local.shape:
                            chunk_size = p_local.shape[shard_dim]
                            start = tp_rank * chunk_size
                            slicer = [slice(None)] * u.dim()
                            slicer[shard_dim] = slice(start, start + chunk_size)
                            u = u[tuple(slicer)]
                prepared.append((p, u))
        else:
            prepared = list(zip(slice_params, slice_updates))

        # ------------- Phase 2: foreach buckets -------------
        buckets = defaultdict(
            lambda: {
                "orig": [],
                "locals": [],
                "updates": [],
                "lr": None,
                "wd": None,
                "m": None,
            }
        )

        for p, u in prepared:
            if u is None:
                continue
            lr, _, momentum, wd, _ = self.groups_info[self.parameters_to_groups[id(p)]]
            p_local = p.to_local() if isinstance(p, DTensor) else p

            # robust shape check after TP slicing; if it still fails, it’s a caller misalignment
            if p_local.shape != u.shape:
                raise ValueError(
                    f"Shape mismatch between parameter shard {p_local.shape} and update slice {u.shape}. "
                    f"Ensure you pass the same DDP/FSDP window and slice TP updates. "
                )

            # dtype/device consistency for foreach
            if u.dtype is not p_local.dtype:
                u = u.to(p_local.dtype)

            key = (p_local.device, p_local.dtype, float(lr), float(wd), float(momentum))
            b = buckets[key]
            b["orig"].append(p)
            b["locals"].append(p_local)
            b["updates"].append(u)
            b["lr"], b["wd"], b["m"] = lr, wd, momentum

        for (_, _, lr, wd, m), data in buckets.items():
            if not data["locals"]:
                continue
            if wd != 0.0:
                torch._foreach_mul_(data["locals"], 1.0 - wd * lr)
            torch._foreach_add_(data["locals"], data["updates"], alpha=-lr)

            # light-mode grad decay retained for parity (won’t run; is_light is False in this optimizer)
            if self.is_light and m != 1.0:
                # group grads by device/dtype for foreach
                gmap = defaultdict(list)
                for p in data["orig"]:
                    if p.grad is not None:
                        g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                        gmap[(g.device, g.dtype)].append(g)
                for _, gs in gmap.items():
                    torch._foreach_mul_(gs, 1.0 - m)
