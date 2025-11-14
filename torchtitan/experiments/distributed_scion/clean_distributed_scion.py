# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.tools.logging import logger

from .muon_utils import zeropower_backends
from .norm_helper import NORM_FUNCTIONS

__all__ = [
    "DistributedScion",
]


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2
    Unknown = 3


def get_param_type(p, fsdp_enabled, expert_enabled):
    """
    We can aggressively assume that the param is FSDP-Sharded
    """
    if p.grad is None:
        return ParamType.Unknown
    if not fsdp_enabled and not expert_enabled and isinstance(p, torch.Tensor):
        return ParamType.DDP
    if p.ndim == 3:
        return ParamType.Expert
    elif fsdp_enabled:
        return ParamType.FSDP
    else:
        return ParamType.Unknown


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
    assert tp_mesh_dim is not None, "something wrong here"
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


class DistributedScion(torch.optim.Optimizer):
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
    ):
        self.need_to_calculate_norm = False
        self.norms_to_log: list[str] = list(NORM_FUNCTIONS.keys())
        self.norms_at_current_step = {}
        self.extra_reduce_for_HSDP = False
        self.log_parameters_types = True

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor,
            backend=backend,
            backend_steps=backend_steps,
        )
        self.is_light = is_light

        assert self.is_light is False, " light mode not tested yet"

        is_unconstrained = weight_decay == 0

        self.world_mesh = parallel_dims.world_mesh

        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled

        # this is used to ensure only the DP or FSDP rank 0 will have norms
        self.is_dp_rank_0 = dist.get_rank(self.world_mesh["dp_cp"].get_group()) == 0

        assert experts_weights_layout in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ], f"Unknown experts weights layout: {experts_weights_layout}"
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"
        self.extra_reduce_for_HSDP = extra_reduce_for_HSDP

        logger.info(
            f"Distributed Scion optimizer "
            f"(is_light={self.is_light}, is_unconstrained={is_unconstrained}) "
            f"is enabled with world_mesh={self.world_mesh} | fsdp_enabled={self.fsdp_enabled} | "
            f"EP={self.expert_enabled} | TP={self.tp_enabled} | DP={self.dp_replicate_enabled}"
        )

        super().__init__(params, defaults)
        if self.is_light:
            # Initialize state
            self._store_grads_in_state()
            # Do not pass `self` through syntactic sugar. We need the
            # argument to not be populated.
            self.register_state_dict_pre_hook(
                type(self)._store_grads_in_state,
            )
            self.register_load_state_dict_post_hook(
                type(self)._load_grads_from_state,
            )

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
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
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

            if self.is_light and nesterov:
                raise RuntimeError(
                    "Nesterov momentum is not supported for Scion's light mode. "
                    "Please set nesterov=False."
                )

    def calculate_norm_at_next_step(self, norms_to_log: list[str]):
        self.need_to_calculate_norm = True
        self.norms_to_log = norms_to_log
        self.norms_at_current_step = {}

    def get_norms_at_current_step(self):
        if self.is_dp_rank_0:
            return self.norms_at_current_step
        else:
            return {}

    def zero_grad(self, *args, **kwargs):
        if self.is_light:
            pass
        else:
            super().zero_grad(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        scale_params, scale_param_names = [], []
        embed_params, embed_param_names = [], []
        ddp_params, ddp_param_names = [], []
        fsdp_params, fsdp_param_names = [], []
        expert_params, expert_param_names = [], []

        for group_idx, group in enumerate(self.param_groups):
            # we should update self.groups_info here incase we have LR and momentum scheduler
            # We can also optionally do norm_factor and backend scheduler if we want to
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

            for p_name, p in zip(group["param_names"], group["params"]):
                norm_factor = group["norm_factor"]
                backend = group["backend"]
                is_embed_norm = norm_factor.startswith(
                    "embed"
                ) or norm_factor.startswith("unembed")

                if p.numel() == 1:
                    assert (
                        backend == "identity"
                    ), "scale params must use identity backend"
                    assert (
                        norm_factor == "sign"
                    ), "scale params must use sign norm factor"
                    scale_params.append(p)
                    scale_param_names.append(p_name)
                    continue

                if backend == "identity" and is_embed_norm:
                    # for these Row/Col-wise norm, there is no need to gather the gradient
                    embed_params.append(p)
                    embed_param_names.append(p_name)
                    continue

                param_type = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if param_type == ParamType.DDP:
                    ddp_params.append(p)
                    ddp_param_names.append(p_name)
                elif param_type == ParamType.FSDP:
                    fsdp_params.append(p)
                    fsdp_param_names.append(p_name)
                elif param_type == ParamType.Expert:
                    expert_params.append(p)
                    expert_param_names.append(p_name)
                elif param_type == ParamType.Unknown:
                    logger.warning(
                        f"Unknown param type: {p_name}, p.shape {p.shape}, grad is None[?] "
                        f"{p.grad is None}, the optimizer will skip this param"
                    )
                    # raise ValueError(f"Unknown param type: {p_name}")
                    continue
                else:
                    raise ValueError("param_type")

        # Sort fsdp_params and their names together
        fsdp_pairs = list(zip(fsdp_params, fsdp_param_names))
        fsdp_pairs.sort(key=lambda x: x[0].numel(), reverse=True)
        fsdp_params, fsdp_param_names = zip(*fsdp_pairs) if fsdp_pairs else ([], [])
        # Sort expert_params and their names together
        expert_pairs = list(zip(expert_params, expert_param_names))
        expert_pairs.sort(key=lambda x: (x[0].numel(), x[0].shape[1]), reverse=True)
        expert_params, expert_param_names = (
            zip(*expert_pairs) if expert_pairs else ([], [])
        )
        if self.log_parameters_types:
            # only log once
            logger.info(
                f"fsdp_params: {len(fsdp_params)} | expert_params: {len(expert_params)} | "
                f"ddp_params: {len(ddp_params)} | embed_params: {len(embed_params)} | "
                f"scale_params: {len(scale_params)}"
            )
            self.log_parameters_types = False

        """
        We could merge `embed_params` and `expert_params` into one list.
        The diff is, we are sure expert_params have bunch of 2D full-matrixs
        But we might need to gather the `embed_params` to 2D full-matrixs
        if we wanna to get the norm of the gradient.
        """
        self.step_scalar(scale_params, scale_param_names)
        self.step_embedding(embed_params, embed_param_names)
        self.step_experts(expert_params, expert_param_names)
        self.step_ddp(ddp_params, ddp_param_names)
        self.step_fsdp(fsdp_params, fsdp_param_names)

        # reset the flag for the next step
        self.need_to_calculate_norm = False
        return loss

    @torch.no_grad()
    def lmo(
        self,
        g,
        eps,
        norm_factor,
        zeropower_backend,
        backend_steps,
    ):
        """
        Supported Weight Types:
            - 1-D tensors: Bias vectors (Linear/Convolution layers)
            - 2-D tensors: Linear layer weights [D_out, D_in]
            - 3-D tensors: Grouped expert weights [G, D_in, D_out] or [G, D_out, D_in]
            - 4-D tensors: Conv2D weights [D_out, D_in, KH, KW] (forced to "conv_spectral")
            - 5-D tensors: Conv3D weights [D_out, D_in, KH, KW, KD] (forced to "conv_spectral")

        Limitations:
            - Does not support learnable RMS/Layer-norm parameters
            - Does not support shared experts in format [D_in, D_out * n_shared_experts], where n_shared_experts > 1
            - Does not support Conv1D layers

        Note:
            - For 3-D expert weights, the layout must be specified during optimizer initialization.
            - 0-D (scalar) weights is supported but should not appear in this function call
        """
        g = g.to_local() if isinstance(g, DTensor) else g

        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        def _lmo_for_2d_tensor(g, need_transpose=False):
            g = g if not need_transpose else g.transpose(0, 1)
            g = zeropower_backends[zeropower_backend](g, steps=backend_steps, eps=eps)
            g = self.normalise_grad(g, norm_factor=norm_factor, eps=eps)
            return g if not need_transpose else g.transpose(0, 1)

        if g.ndim == 2:
            g = _lmo_for_2d_tensor(g, need_transpose=False)
        elif g.ndim == 3:
            if g.shape[0] > 0:
                # When world_size [fsdp x EP] > Total number of experts,
                # some ranks may have 0 experts that shape will be [0, d-in, d-out]
                # We should return the original grad here and **do not** do stack
                g = torch.stack(
                    [
                        _lmo_for_2d_tensor(
                            g[i], need_transpose=self.experts_need_transpose
                        )
                        for i in range(g.shape[0])
                    ],
                    dim=0,
                )
            else:
                pass
        elif g.ndim == 1:
            if zeropower_backend != "bias_rms":
                g_diag = torch.diag_embed(g).contiguous()
                result_diag = _lmo_for_2d_tensor(g_diag)
                g = result_diag.diagonal().contiguous()
            else:
                g = self.normalise_grad(g, norm_factor="bias_rms", eps=eps)
                pass

        elif g.ndim == 4 or g.ndim == 5:
            g = zeropower_backends[zeropower_backend](
                g.reshape(len(g), -1), steps=backend_steps, eps=eps
            ).view(g.shape)
            g = self.normalise_grad(g, norm_factor="conv_spectral", eps=eps)

        else:
            raise ValueError(f"Unknown grad shape: {g.shape}")

        return g

    @torch.no_grad()
    def normalise_grad(self, g, norm_factor, eps):
        if norm_factor == "spectral":
            g = g * (g.size(0) / g.size(1)) ** 0.5
        elif norm_factor == "image_spectral":
            g = g * max((g.size(0) / g.size(1)) ** 0.5, 1)
        elif norm_factor.startswith("embed"):
            # NB: here assume shape [vocab_size, embed_dim]
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
            if norm_factor == "embed_linear":
                g = g * g.size(1)
            elif norm_factor == "embed_sqrt":
                g = g * g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith("unembed"):
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
            if norm_factor == "unembed_linear":
                g = g / g.size(1)
            elif norm_factor == "unembed_sqrt":
                g = g / g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor == "sign":
            g = torch.sign(g)
        elif norm_factor == "bias_rms":
            rms_value = torch.sqrt(g.pow(2).mean())
            g = g / (rms_value + eps)
        elif norm_factor == "conv_spectral":
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels) ** 0.5 / (k**2)
        elif norm_factor == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g

    def __getstate__(self):
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})["grad_state"] = param.grad

    def _load_grads_from_state(self):
        for param, state in self.state.items():
            if "grad_state" in state:
                param.grad = state["grad_state"]
            elif isinstance(param, torch.Tensor):
                param.grad = None

    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_group=None):
        # TODO(JSC): we could maybe use tesnor update rather than for-loop here
        # can be helpful for FSDP and EP params
        for idx_in_bucket in range(start_idx, end_idx):
            shift = idx_in_bucket - start_idx
            p = params[idx_in_bucket]
            u = updates[shift]
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]

            if wd != 0:
                # p.data.mul_(1 - wd*lr)
                p.mul_(1 - wd * lr)

            if isinstance(p, DTensor) and self.tp_enabled:
                original_placements = p.placements
                tp_mesh_dim = tp_axis(original_placements, p.shape == u.shape)

            if isinstance(p, DTensor):
                if tp_group is None or tp_mesh_dim is None:
                    p.to_local().add_(u, alpha=-lr)
                else:
                    tp_rank = tp_group.rank()
                    tp_sharded_dim = original_placements[tp_mesh_dim].dim
                    chunk_size = p.to_local().shape[tp_sharded_dim]
                    start_offset = tp_rank * chunk_size

                    slicer = [slice(None)] * u.dim()
                    slicer[tp_sharded_dim] = slice(
                        start_offset, start_offset + chunk_size
                    )
                    u_sliced = u[slicer]
                    p.to_local().add_(u_sliced, alpha=-lr)
            else:
                p.add_(u, alpha=-lr)

            if momentum != 1 and self.is_light and p.grad is not None:
                p.grad.mul_(1 - momentum)

    def step_scalar(
        self,
        scalar_params,
        scalar_param_names,
        skip_update=False,
    ):
        if len(scalar_params) == 0:
            return {}

        for param_idx in range(len(scalar_params)):
            p = scalar_params[param_idx]
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )
            g = g.to_local() if isinstance(g, DTensor) else g

            # the lmo of scalar is just sign
            u = torch.sign(g)

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

    def step_embedding(
        self,
        embed_params,
        embed_param_names,
        skip_update=False,
    ):
        if len(embed_params) == 0:
            return {}

        tp_group = None
        # if self.dp_replicate_enabled:
        #     dp_replicate_group = self.world_mesh["dp_replicate"].get_group()
        # else:
        #     dp_replicate_group = None

        if self.tp_enabled:
            tp_group = self.world_mesh["tp"].get_group()

        for param_idx in range(len(embed_params)):
            p = embed_params[param_idx]
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )

            u = self.lmo(g, **param_kwargs)

            #########################################################
            # # As of we use norm for Embedding, maybe we should not do Reduce here
            # if (
            #     dp_replicate_group is not None
            #     and self.extra_reduce_for_HSDP
            #     and self.fsdp_enabled
            # ):
            #     dist.all_reduce(u, group=dp_replicate_group, op=dist.ReduceOp.AVG)
            #     dist.barrier(group=dp_replicate_group)
            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1, tp_group=tp_group)

    def step_experts(
        self,
        expert_params,
        expert_param_names,
        skip_update=False,
    ):
        if len(expert_params) == 0:
            return {}

        device = expert_params[0].device
        fsdp_group = self.world_mesh["dp_shard_cp"].get_group()
        world_size = dist.get_world_size(fsdp_group)
        local_rank = dist.get_rank(fsdp_group)
        ep_per_rank = math.ceil(expert_params[0].shape[0] / world_size)

        # each rank will process `len(expert_params) * ep_per_rank` experts
        # each expert will have `self.norms_to_log` norms
        # so each rank will have `len(expert_params) * ep_per_rank * len(self.norms_to_log)`
        # norms
        # globally, its [[g0-ep0, g0-ep1, g0-ep2, ...], [g1-ep0, g1-ep1, g1-ep2, ...], ...] on each
        # rank

        for param_idx in range(len(expert_params)):
            p = expert_params[param_idx]
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )
            u = self.lmo(g, **param_kwargs)

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

    def step_ddp(
        self,
        ddp_params,
        ddp_param_names,
        skip_update=False,
    ):
        # Either we do DDP
        # or we do TP,  there is no [DDP + TP] case but for safety we add sevel checks
        # if len(ddp_params) == 0:
        #     return {}

        tp_group, dp_replicate_group = None, None

        rank = 0
        bucket_size = world_size = 1
        total_buckets = len(ddp_params)

        if self.dp_replicate_enabled:
            dp_replicate_group = self.world_mesh["dp_replicate"].get_group()
            world_size = dp_replicate_group.size()
            rank = dp_replicate_group.rank()

            bucket_size = world_size
            total_buckets = math.ceil(len(ddp_params) / bucket_size)

        if self.tp_enabled:
            tp_group = self.world_mesh["tp"].get_group()
            tp_world_size = dist.get_world_size(group=tp_group)

        device = ddp_params[0].device if len(ddp_params) > 0 else torch.device("cuda")
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.zeros, dtype=cast_dtype, device=device)

        # for DDP, we need to first update the buffer
        for param_idx in range(len(ddp_params)):
            p = ddp_params[param_idx]
            _, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, update_buffer=True, gather_to_local=False
            )

        #  then we do scion stuff
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(ddp_params))
            current_rank_idx = start_idx + rank
            if current_rank_idx < len(ddp_params):
                p = ddp_params[current_rank_idx]
                # Step 1: Get the gradient
                _, nesterov, momentum, wd, param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(p)]
                ]
                g = self.get_momentum_or_grad(
                    p, momentum, nesterov, update_buffer=False, gather_to_local=False
                )
                if isinstance(g, DTensor) and self.tp_enabled:
                    g = gather_tp_shard(
                        g.to_local(), tp_group, tp_world_size, g.placements
                    ).to(dtype=cast_dtype)

            else:
                # To avoid idle stream, we pad the last rank
                p = ddp_params[end_idx - 1]
                g = zero_tensor(p.shape)
                _, nesterov, momentum, wd, param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(p)]
                ]

            # step 2: lmo
            u = self.lmo(g, **param_kwargs)

            if not skip_update:
                # Step 3: FOR DDP, we do all-gather
                if self.dp_replicate_enabled:
                    # only gather params when we doing DDP + BUCKET
                    gather_lists = [None] * world_size
                    for i in range(world_size):
                        param_idx = start_idx + i
                        if i == rank or param_idx >= len(ddp_params):
                            gather_lists[i] = u.to(dtype=cast_dtype)
                        elif param_idx < len(ddp_params):
                            p = ddp_params[start_idx + i]
                            gather_lists[i] = zero_tensor(p.shape)
                    dist.all_gather(
                        gather_lists, u.to(dtype=cast_dtype), group=dp_replicate_group
                    )
                    if self.tp_enabled:
                        # only if DP+TP we need to barrier here other-wise its automatically synced
                        dist.barrier(group=dp_replicate_group)
                else:
                    # other wise (TP only), dp world_size is 1
                    gather_lists = [u.to(dtype=cast_dtype)]

                # Step 4: Update the parameters
                self.update_bucket_params(
                    ddp_params, gather_lists, start_idx, end_idx, tp_group=tp_group
                )

    def step_fsdp(
        self,
        fsdp_params,
        fsdp_param_names,
        skip_update=False,
    ):
        if len(fsdp_params) == 0:
            return {}
        tp_group, dp_replicate_group = None, None
        """
        To make FSDP+DP works, we lets step_fsdp work on each dp_replicate separately.
        Hence, we only care about the world size inside the dp_replicate.
        """

        # due to the werid implementation of parallel_dims.py (upstream)
        # here we should use `dp_shard_cp` rather then `dp_shard` as of
        # CP is also part of the dp_shard
        fsdp_group = self.world_mesh["dp_shard_cp"].get_group()

        if self.dp_replicate_enabled:
            dp_replicate_group = self.world_mesh["dp_replicate"].get_group()

        if self.tp_enabled:
            tp_group = self.world_mesh["tp"].get_group()
            tp_world_size = dist.get_world_size(group=tp_group)

        world_size = dist.get_world_size(fsdp_group)
        rank = dist.get_rank(fsdp_group)

        # @ THIS IS A HACK
        bucket_size = world_size
        total_buckets = math.ceil(len(fsdp_params) / bucket_size)

        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype
        zero_tensor = partial(torch.empty, dtype=cast_dtype, device=device)

        # Process each bucket
        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(fsdp_params))

            # Step 1: Prepare data for first all_to_all
            grads_send_list, send_shapes = [], []
            target_shape, param_kwargs = None, None

            for rank_idx in range(world_size):
                current_rank_idx = start_idx + rank_idx

                if current_rank_idx < len(fsdp_params):
                    p = fsdp_params[current_rank_idx]
                    _, nesterov, momentum, wd, param_kwargs = self.groups_info[
                        self.parameters_to_groups[id(p)]
                    ]

                    g = self.get_momentum_or_grad(
                        p,
                        momentum,
                        nesterov,
                        update_buffer=True,
                        gather_to_local=False,
                    )

                    original_placements = g.placements
                    tp_mesh_dim = tp_axis(original_placements)
                    if tp_group is not None and tp_mesh_dim is not None:
                        # the reason we need `tp_mesh_dim` is we want a flexible solution
                        # that Attention go TP and MLP go EP
                        g = gather_tp_shard(
                            g.to_local(), tp_group, tp_world_size, original_placements
                        ).to(dtype=cast_dtype)
                    else:
                        g = g.to_local().to(dtype=cast_dtype)

                    # Save the shape info for this parameter
                    if rank == rank_idx:
                        target_shape = p.shape
                else:
                    # Use a dummy shape for parameters beyond our range
                    p = fsdp_params[end_idx - 1]
                    g = zero_tensor(p.to_local().shape)

                grads_send_list.append(g)
                send_shapes.append(g.shape)

            # Make sure target_shape is initialized
            # (trigger by the padding of the last ranks)
            if target_shape is None and end_idx > 0:
                target_shape = fsdp_params[end_idx - 1].shape
                param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(fsdp_params[end_idx - 1])]
                ][-1]

            recv_shapes = [
                calculate_shard_shape(target_shape, rank_idx, world_size)
                for rank_idx in range(world_size)
            ]
            recv_list = [zero_tensor(shape) for shape in recv_shapes]
            # Step 3: First all_to_all - using ASYNC version
            dist.barrier()
            dist.all_to_all(recv_list, grads_send_list, group=fsdp_group)
            # Step 5: Concatenate received gradients along dimension 0 and perform NS5
            # All tensors in recv_list should have the same dimensions except for dim 0

            full_g = torch.cat(recv_list, dim=0)
            u = self.lmo(full_g, **param_kwargs)
            dist.barrier(group=fsdp_group)

            if dp_replicate_group is not None and self.extra_reduce_for_HSDP:
                dist.all_reduce(u, group=dp_replicate_group, op=dist.ReduceOp.AVG)
                dist.barrier(group=dp_replicate_group)
            # in case of FSDP+DP, we can do a All-Reduce here sync the grads
            if not skip_update:
                # Step 6: Split the processed tensor back for second all_to_all
                split_sizes = [shape[0] for shape in recv_shapes]

                grads_send_list = list(torch.split(u, split_sizes, dim=0))
                recv_list = [zero_tensor(shape) for shape in send_shapes]
                # Step 8: Second all_to_all - using ASYNC version
                dist.all_to_all(recv_list, grads_send_list, group=fsdp_group)
                del grads_send_list
                # Step 10: Update parameters using the results
                self.update_bucket_params(
                    fsdp_params,
                    recv_list,
                    start_idx,
                    end_idx,
                    tp_group=tp_group,
                )

    @torch.no_grad()
    def get_momentum_or_grad(
        self, p, momentum, nesterov, update_buffer=False, gather_to_local=False
    ):
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            state = self.state[p]
            if "momentum_buffer" not in state.keys():
                if update_buffer:
                    state["momentum_buffer"] = torch.zeros_like(g)
                else:
                    """
                    When you using DDP + Dist-muon,you might trieer an error here.
                    Because in the optimizer.log you try to log all gradient's norm.
                    But for DDP + Dist-muon, each rank only has a part of the gradient.

                    --
                    For debug, you can return None here.
                    """
                    raise ValueError(
                        "Momentum buffer not found in optimizer state. "
                        "Please check if the optimizer is initialized correctly."
                    )
            buf = state["momentum_buffer"]
            if update_buffer:
                buf.mul_(1 - momentum).add_(g, alpha=momentum)
            else:
                buf = buf.mul(1 - momentum).add(g, alpha=momentum)
            g = buf if not nesterov else buf.mul(1 - momentum).add(g, alpha=momentum)

        if gather_to_local and isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()
        return g
