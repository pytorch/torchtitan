# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.components.ft.manager import FTManager
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger


IGNORE_INDEX = -100  # Pytorch's default for F.cross_entropy


# WARNING: currently this does not take into account gradient accumulation
# and the gradient can still be biased toward grad accum step with less valid tokens
# See: https://github.com/pytorch/torchtitan/issues/1842
def token_imbalance_ce_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    token_mesh: DeviceMesh,
    ft_pg: dist.ProcessGroup | None,
) -> torch.Tensor:
    """
    Cross‑entropy loss that is *robust* to varying numbers of valid tokens across ranks.

    In a typical distributed training setup (data parallel + sequence parallel),
    each rank computes the loss over **only its local tokens** and returns an
    *average* over those tokens:

    Afterwards, when Fully‑Sharded Data Parallel (FSDP) averages the gradients
    across all ranks, the resulting update is equivalent to a **global sample
    average** *only if every rank contains the same number of tokens*.
    In practice that assumption is violated for many workloads:
    - Sequences are padded to a fixed length -> some ranks see fewer real tokens.
    - SFT finetuning where user's queries tokens are masked out.
    - Vision encoders often injects a large number of “ignored”
      tokens as context that are not trained with text tokens' loss.

    This function fixes the issue by **scaling the sum-of-loss** with the *average*
    number of non‑ignored tokens per rank, computed via an all-reduce over
    `token_mesh`.  The returned scalar therefore represents the loss that would
    be obtained if every token in the entire distributed batch contributed with
    equal weight to the global gradient, regardless of how many padded or
    ignored tokens each rank contains.

    Parameters
    ----------
    pred : torch.Tensor
    labels : torch.Tensor
    token_mesh : DeviceMesh
        A device mesh that contains all ranks participating in this training step's
        loss computation.  The function performs an ``all_reduce`` (mean) over the
        `num_tokens` tensor of a rank across this mesh.
    ft_pg: dist.ProcessGroup | None
        Optional pg for Fault Tolerance training.

    Returns
    -------
    torch.Tensor
        A scalar loss tensor,  ready for ``backward()`` and FSDP all-reduce mean

    Notes
    -----
    * The function internally uses :func:`torch.nn.functional.cross_entropy`
      with ``reduction="sum"`` so that each token contributes exactly once to
      the numerator.  The denominator is the **average** number of valid tokens
      per rank, not the local count.
    * If a rank contains no valid tokens (i.e., all labels are ``IGNORE_INDEX``),
      its contribution to the sum is zero and its `num_tokens` becomes zero.
      In that case the mean across ranks will still be well‑defined as long as
      at least one rank has non‑zero token count.
    """
    sum_loss = torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )
    num_tokens = (labels != IGNORE_INDEX).sum()
    avg_num_tokens_per_rank = funcol.all_reduce(
        num_tokens, reduceOp=c10d.ReduceOp.AVG.name, group=token_mesh
    )
    if ft_pg is not None:
        avg_num_tokens_per_rank = funcol.all_reduce(
            avg_num_tokens_per_rank, reduceOp=c10d.ReduceOp.AVG.name, group=ft_pg
        )
    return sum_loss / avg_num_tokens_per_rank


def build_token_imbalance_ce_loss(
    job_config: JobConfig, parallel_dims: ParallelDims, ft_manager: FTManager, **kwargs
):
    del kwargs  # delete any unused arguments
    # NOTE: The device mesh where the input tokens w/ shape BSD can be sliced:
    # DP split the batch dim B
    # CP split the sequence dim S
    token_mesh = parallel_dims.world_mesh["dp_cp"]
    ft_pg = ft_manager.loss_sync_pg
    loss_fn = partial(token_imbalance_ce_loss, token_mesh=token_mesh, ft_pg=ft_pg)
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn
