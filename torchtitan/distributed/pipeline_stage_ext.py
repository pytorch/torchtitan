from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.nn.parallel import DistributedDataParallel

from torchtitan.tools.logging import logger


class PipelineStageExt(PipelineStage):
    """
    Extends the PipelineStage to include additional functionality or overrides.
    This class can be used to customize the behavior of pipeline stages in a distributed setting.
    """

    def __init__(self, job_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize additional attributes based on job configuration
        self.reshard_after_backward = (
            job_config.parallelism.pipeline_parallel_reshard_after_backward
        )
        self.requires_gradient_sync = (
            job_config.parallelism.pipeline_parallel_requires_gradient_sync
        )

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        Compared to the base implementation, this version additionally frees the output chunk.
        """
        output = self.output_chunks[fwd_chunk_id]
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size(),
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        self.output_chunks[fwd_chunk_id] = torch.Tensor([])

        return ops

    # Override or add methods as necessary
    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            reshard_after_backward = (
                self.reshard_after_backward if not last_backward else True
            )
            requires_gradient_sync = (
                self.requires_gradient_sync if not last_backward else True
            )
            self.submod.set_is_last_backward(last_backward)
            self.submod.set_reshard_after_backward(reshard_after_backward)
            self.submod.set_requires_gradient_sync(requires_gradient_sync)
            self.submod.set_requires_all_reduce(last_backward)
            result = perform_backward(backward_type)()
        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups
