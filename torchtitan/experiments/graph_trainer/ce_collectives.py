# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Copy-engine adapters for SimpleFSDP's functional and bucketed collectives."""

import functools
import logging
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import Work

from torchtitan.experiments.ce_collectives.collectives import (
    CEAllToAllReduceScatter,
    CEMulticastAllGather,
    CEWorkspace,
    domain_rank_groups,
)
from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META

logger = logging.getLogger(__name__)

_lib = torch.library.Library("graph_trainer_ce", "DEF")
_lib.define(
    "all_gather_into_tensor(Tensor input, int group_size, Any group_name, "
    "Dict(str, Any)? config=None, *, int workspace_size_mib=128) -> Tensor",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
_lib.define(
    "all_gather_into_tensor_out(Tensor input, int group_size, Any group_name, "
    "Dict(str, Any)? config=None, *, Tensor(a!) out, int workspace_size_mib=128) -> ()",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
_lib.define(
    "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, "
    "Any group_name, Dict(str, Any)? config=None, *, int ranks_per_domain, "
    "int workspace_size_mib=128) -> Tensor",
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _group(group_name: Any, group_size: int, config: dict | None) -> dist.ProcessGroup:
    if config:
        raise ValueError("CE collectives do not support NCCL-specific config options.")
    group = (
        group_name
        if isinstance(group_name, dist.ProcessGroup)
        else dist.distributed_c10d._resolve_process_group(group_name)
    )
    if group.size() != group_size:
        raise ValueError("CE collective group_size does not match its process group.")
    return group


@functools.cache
def _communication_stream(group: dist.ProcessGroup, device: torch.device):
    return torch.cuda.Stream(device=device)


@functools.cache
def _communication_workspace(
    group: dist.ProcessGroup, device: torch.device, workspace_size_mib: int
):
    # Large-pool requests below 10 MiB get a 20 MiB segment in PyTorch.
    return CEWorkspace(group, size_bytes=workspace_size_mib * 1024**2)


@functools.cache
def _all_gather(group: dist.ProcessGroup, device: torch.device):
    return CEMulticastAllGather(group)


@functools.cache
def _reduce_scatter(
    group: dist.ProcessGroup, device: torch.device, ranks_per_domain: int
):
    return CEAllToAllReduceScatter(group, ranks_per_domain=ranks_per_domain)


class _CollectiveWork(Work):
    def __init__(self, caller_stream, *communication_tensors):
        super().__init__()
        self._caller_stream = caller_stream
        self._communication_tensors = communication_tensors

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        super().wait(timeout)
        if torch.cuda.current_stream() != self._caller_stream:
            self.event.wait(self._caller_stream)
        # Like NCCL's tensor stash, retain buffers until the caller is ordered
        # after communication. record_stream() instead defers their reuse until
        # the end of CUDA graph capture, accumulating buffers across layers.
        self._communication_tensors = ()
        return True


def _enqueue_all_gather(
    input: torch.Tensor, out: torch.Tensor, group, *, workspace_size_mib: int
) -> None:
    if (
        not input.is_contiguous()
        or not out.is_contiguous()
        or input.dtype != out.dtype
        or input.device != out.device
        or out.numel() != input.numel() * group.size()
    ):
        raise ValueError("CE all-gather requires equal contiguous input/output shards.")
    comm = _all_gather(group, input.device)
    stream = _communication_stream(group, input.device)
    caller_stream = torch.cuda.current_stream(input.device)
    stream.wait_stream(caller_stream)
    with torch.cuda.stream(stream):
        workspace = _communication_workspace(group, input.device, workspace_size_mib)
        storage = workspace.get_buffer(dtype=input.dtype, device=input.device)
        chunk_capacity = storage.numel() // group.size()
        source = input.flatten()
        output = out.view(group.size(), input.numel())
        for offset in range(0, input.numel(), chunk_capacity):
            chunk_numel = min(chunk_capacity, input.numel() - offset)
            scratch = storage[: group.size() * chunk_numel]
            shard = scratch.narrow(0, group.rank() * chunk_numel, chunk_numel)
            shard.copy_(source.narrow(0, offset, chunk_numel))
            comm(scratch, shard, group, workspace=workspace)
            output[:, offset : offset + chunk_numel].copy_(
                scratch.view(group.size(), chunk_numel)
            )
        torch._C._distributed_c10d._register_work(
            out, _CollectiveWork(caller_stream, input, out)
        )


@torch.library.impl(_lib, "all_gather_into_tensor", "CUDA")
def all_gather_into_tensor(
    input, group_size, group_name, config=None, *, workspace_size_mib=128
):
    group = _group(group_name, group_size, config)
    out = input.new_empty((input.shape[0] * group_size, *input.shape[1:]))
    _enqueue_all_gather(input, out, group, workspace_size_mib=workspace_size_mib)
    return out


@torch.library.impl(_lib, "all_gather_into_tensor", "Meta")
def _all_gather_meta(
    input, group_size, group_name, config=None, *, workspace_size_mib=128
):
    return input.new_empty((input.shape[0] * group_size, *input.shape[1:]))


@torch.library.impl(_lib, "all_gather_into_tensor_out", "CUDA")
def all_gather_into_tensor_out(
    input, group_size, group_name, config=None, *, out, workspace_size_mib=128
):
    _enqueue_all_gather(
        input,
        out,
        _group(group_name, group_size, config),
        workspace_size_mib=workspace_size_mib,
    )


@torch.library.impl(_lib, "all_gather_into_tensor_out", "Meta")
def _all_gather_out_meta(
    input, group_size, group_name, config=None, *, out, workspace_size_mib=128
):
    return None


@torch.library.impl(_lib, "reduce_scatter_tensor", "CUDA")
def reduce_scatter_tensor(
    input,
    reduce_op,
    group_size,
    group_name,
    config=None,
    *,
    ranks_per_domain,
    workspace_size_mib=128,
):
    group = _group(group_name, group_size, config)
    if reduce_op not in ("sum", "avg") or input.shape[0] % group_size:
        raise ValueError("CE reduce-scatter requires equal shards and SUM or AVG.")
    comm = _reduce_scatter(group, input.device, ranks_per_domain)
    out = input.new_empty((input.shape[0] // group_size, *input.shape[1:]))
    stream = _communication_stream(group, input.device)
    caller_stream = torch.cuda.current_stream(input.device)
    stream.wait_stream(caller_stream)
    with torch.cuda.stream(stream):
        comm(
            out,
            input,
            group,
            dist.ReduceOp.SUM if reduce_op == "sum" else dist.ReduceOp.AVG,
            workspace=_communication_workspace(
                comm.local_group, input.device, workspace_size_mib
            ),
        )
        torch._C._distributed_c10d._register_work(
            out, _CollectiveWork(caller_stream, input, out)
        )
    return out


@torch.library.impl(_lib, "reduce_scatter_tensor", "Meta")
def _reduce_scatter_meta(
    input,
    reduce_op,
    group_size,
    group_name,
    config=None,
    *,
    ranks_per_domain,
    workspace_size_mib=128,
):
    return input.new_empty((input.shape[0] // group_size, *input.shape[1:]))


def ce_collectives_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    config: CECollectivesConfig,
    workspace_size_mib: int = 128,
    excluded_module_fqns: tuple[str, ...] = (),
) -> torch.fx.GraphModule:
    """Replace SimpleFSDP collectives after bucketing, retaining their waits.

    Reduction uses the CE implementation's fixed FP32 accumulation order and
    may differ from NCCL. TP/EP collectives without FSDP provenance are retained.
    """
    if workspace_size_mib <= 0:
        raise ValueError("CE workspace_size_mib must be positive.")
    c10d = torch.ops._c10d_functional
    ce = torch.ops.graph_trainer_ce
    replacements = {}
    if config.all_gather:
        replacements[
            c10d.all_gather_into_tensor.default
        ] = ce.all_gather_into_tensor.default
        replacements[
            c10d.all_gather_into_tensor_out.default
        ] = ce.all_gather_into_tensor_out.default
    if config.reduce_scatter:
        replacements[
            c10d.reduce_scatter_tensor.default
        ] = ce.reduce_scatter_tensor.default

    counts = {"all_gather": 0, "reduce_scatter": 0}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in replacements:
            continue
        sources = node.meta.get("bucketing_custom_sources") or [
            node.meta.get("custom", {})
        ]
        if not all(source.get(FSDP_PARAM_FQNS_META) for source in sources):
            continue
        if any(
            fqn.startswith(module + ".")
            for source in sources
            for fqn in source[FSDP_PARAM_FQNS_META]
            for module in excluded_module_fqns
        ):
            continue
        is_reduce_scatter = node.target is c10d.reduce_scatter_tensor.default
        arguments = node.normalized_arguments(gm, normalize_to_only_use_kwargs=True)
        assert arguments is not None
        kwargs = dict(arguments.kwargs)
        group_size = kwargs["group_size"]
        if group_size == 1:
            continue
        ranks_per_domain = config.ranks_per_domain or group_size
        domain_rank_groups(group_size, ranks_per_domain)
        if not is_reduce_scatter and ranks_per_domain != group_size:
            raise ValueError(
                "CE all-gather requires one NVLink domain; set all_gather=False "
                "for hierarchical reduce-scatter."
            )
        if kwargs.get("config"):
            raise ValueError(
                "CE collectives do not support NCCL-specific config options."
            )
        if is_reduce_scatter:
            if kwargs["reduce_op"] not in ("sum", "avg"):
                raise ValueError("CE reduce-scatter supports only SUM and AVG.")
            kwargs["ranks_per_domain"] = ranks_per_domain
        kwargs["workspace_size_mib"] = workspace_size_mib
        if node.target is c10d.all_gather_into_tensor_out.default:
            # A mutable op returning None remains compatible with functionalization.
            # Its existing wait now waits on the same caller-owned output storage.
            node.replace_all_uses_with(kwargs["out"])
            node.meta["val"] = None
        node.target = replacements[node.target]
        node.args = ()
        node.kwargs = kwargs
        counts["reduce_scatter" if is_reduce_scatter else "all_gather"] += 1
    if any(counts.values()):
        logger.info(
            "CE SimpleFSDP collectives: %d all-gathers, %d reduce-scatters",
            counts["all_gather"],
            counts["reduce_scatter"],
        )
        gm.graph.lint()
        gm.recompile()
    else:
        logger.warning(
            "CE collectives requested, but no eligible SimpleFSDP collectives found"
        )
    return gm
