# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for GraphTrainer graph construction.

Flat calling convention and wrapping contract:
1. Extracted graphs execute on the flat values produced by
   ``minimal_fx_tracer``. Tensor subclasses are unwrapped into plain leaves by
   the tracer before FX execution.
2. Only values that cross the PP/runtime boundary are rewrapped: stage forward
   outputs, input gradients sent to the previous stage, and parameter gradients
   before assigning to live ``param.grad``.
3. Internal graph values stay flat because they never escape GraphPP graph
   execution: saved-for-backward values, unsharded FSDP params, raw grad
   leaves, reduce-grad inputs, and multiplexed intermediate outputs.
4. DTensor and other traceable tensor subclasses use the existing tracer layout
   metadata. GraphPP must not add a separate DTensor-specific wrapping path.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    BOXED_CODEGEN_META,
    ensure_boxed_graph_module,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    example_inputs_from_placeholders,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    PagedStashManager,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    canonicalize_graph_pass,
    compile_time_passes,
    construct_mandatory_graph_passes,
    deduplicate_fsdp_unshard_chains_pass,
    eliminate_dead_code_pass,
    final_inductor_compile_passes,
)
from torchtitan.protocols.model import BaseModel


if TYPE_CHECKING:
    from torchtitan.distributed import ParallelismContext
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class GraphTrainerConfigView:
    """Subset of ``GraphTrainer.Config`` read by PP graph construction.

    GraphPP is entered through TorchTitan's generic pipelining function API,
    which passes decomposed config fields instead of the full
    ``GraphTrainer.Config``. PP graph construction reads only ``compile``,
    ``parallelism``, and ``model``, so GraphPP exposes exactly those fields
    instead of synthesizing a fake full trainer config. Both SPMD paths pass
    the full ``GraphTrainer.Config``, which has the same fields.
    """

    compile: GraphTrainerCompileConfig
    parallelism: ParallelismConfig
    model: BaseModel.Config


def _find_fsdp_bucketing_pass(
    passes: list[Callable],
) -> Callable | None:
    for pass_fn in passes:
        if (
            isinstance(pass_fn, functools.partial)
            and pass_fn.func is joint_transformer_block_bucketing_reordering_pass
        ):
            return pass_fn
    return None


def _configure_fsdp_bucketing_pass(
    fsdp_bucketing_pass: Callable | None,
    *,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> Callable | None:
    """Restrict one FSDP bucketing pass to selected collective types."""
    if fsdp_bucketing_pass is None or not (
        bucket_all_gathers or bucket_reduce_scatters or bucket_all_reduces
    ):
        return None
    assert isinstance(fsdp_bucketing_pass, functools.partial)
    return functools.partial(
        fsdp_bucketing_pass.func,
        *fsdp_bucketing_pass.args,
        **dict(
            fsdp_bucketing_pass.keywords or {},
            bucket_all_gathers=bucket_all_gathers,
            bucket_reduce_scatters=bucket_reduce_scatters,
            bucket_all_reduces=bucket_all_reduces,
        ),
    )


def _apply_passes_with_extracted_fsdp_bucketing(
    traced: TracedResult,
    passes: list[Callable],
    fsdp_bucketing_pass: Callable | None,
    *,
    compile_config: GraphTrainerCompileConfig,
    bucket_all_gathers: bool,
    bucket_reduce_scatters: bool,
    bucket_all_reduces: bool,
) -> None:
    """Apply passes with bucketing limited to collectives kept in this graph.

    At the original bucketing-pass position, a configured copy processes only
    collective types that will remain in the joint graph. Collectives selected
    for extraction stay unbucketed until their action graphs are created.
    """
    configured_bucketing_pass: Callable | None = _configure_fsdp_bucketing_pass(
        fsdp_bucketing_pass,
        bucket_all_gathers=bucket_all_gathers,
        bucket_reduce_scatters=bucket_reduce_scatters,
        bucket_all_reduces=bucket_all_reduces,
    )
    configured_passes: list[Callable] = []
    for pass_fn in passes:
        if pass_fn is fsdp_bucketing_pass:
            if configured_bucketing_pass is not None:
                configured_passes.append(configured_bucketing_pass)
        else:
            configured_passes.append(pass_fn)
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        configured_passes,
        compile_config=compile_config,
    )


def _execute_graph_module(
    gm: fx.GraphModule,
    args: list[Any],
) -> tuple[Any, ...]:
    """Execute one boxed FX graph module and normalize its result to a tuple."""

    with torch.no_grad():
        outputs = gm(args)
    if args:
        raise ValueError(
            "GraphPP graph call expected boxed FX codegen to clear its mutable "
            f"argument list, but {len(args)} entries remain."
        )
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _pack_graph_args(
    *,
    graph_name: str,
    input_names: tuple[str, ...],
    flat_input_indices: tuple[int, ...],
    num_param_inputs: int,
    num_sharded_param_values: int,
    unshard_extracted: bool,
    unsharded_param_values: list[Any],
    flat_non_param_inputs: list[Any],
    runtime_validate: bool,
) -> list[Any]:
    """Pack flat runtime values in graph placeholder order."""

    expected_num_param_inputs = (
        num_param_inputs if unshard_extracted else num_sharded_param_values
    )
    if runtime_validate and len(unsharded_param_values) != expected_num_param_inputs:
        raise ValueError(
            f"{graph_name} parameter input count mismatch: "
            f"{len(unsharded_param_values)} != {expected_num_param_inputs}"
        )

    flat_inputs = [*unsharded_param_values, *flat_non_param_inputs]
    graph_args = list(unsharded_param_values[:num_param_inputs])
    for name, flat_index in zip(
        input_names[num_param_inputs:],
        flat_input_indices,
        strict=True,
    ):
        runtime_flat_index = flat_index
        if unshard_extracted:
            # The traced parameter prefix may contain unused aliases from
            # parametrized modules. The unshard graph omits those leaves,
            # shifting every following buffer and user input to the left.
            runtime_flat_index -= num_sharded_param_values - num_param_inputs
        if runtime_validate and (
            runtime_flat_index < 0 or runtime_flat_index >= len(flat_inputs)
        ):
            raise ValueError(
                f"{graph_name} placeholder index is out of range: "
                f"{name} indexes {runtime_flat_index}, but runtime has "
                f"{len(flat_inputs)} flattened inputs"
            )
        graph_args.append(flat_inputs[runtime_flat_index])
    return graph_args


def _compile_graph_pp_module(
    gm: fx.GraphModule,
    *,
    compile_config: GraphTrainerCompileConfig,
    graph_name: str,
) -> fx.GraphModule:
    """Compile one extracted GraphPP callable with GraphTrainer Inductor passes."""
    if compile_config is None or not compile_config.enable_passes:
        return ensure_boxed_graph_module(gm)

    example_inputs = example_inputs_from_placeholders(gm)
    gm = apply_graph_passes(
        gm,
        example_inputs,
        final_inductor_compile_passes(
            compile_config,
            use_cuda_graph=False,
            boxed_codegen=True,
        ),
        compile_config=compile_config,
    )
    if gm.meta.get(BOXED_CODEGEN_META) is not True:
        raise ValueError(
            "GraphPP compiled graph did not use boxed codegen. Check that the "
            "terminal Inductor pass was not disabled."
        )
    logger.info(
        "GraphPP compiled %s with %s inductor",
        graph_name,
        compile_config.inductor_compilation,
    )
    return gm


def _apply_graph_pp_pre_partition_or_extraction_passes(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    parallelism_context: ParallelismContext | None,
    split_fsdp_param_unshard: bool,
    split_fsdp_grad_reduction: bool,
) -> Callable | None:
    """Apply graph invariants before GraphPP partitioning or extraction.

    Required normalization is not controlled by ``enable_passes`` or
    ``disable_passes`` because partitioning and extraction assume canonical FX
    structure: dead code is gone, no-op patterns are collapsed, and every flat
    FSDP parameter has at most one unshard chain. ``enable_passes`` only gates
    the optional GraphTrainer optimization passes that run after normalization.
    When FSDP collectives are extracted, bucketing is split into two steps:

    1. Select the bucketing pass for reuse on extracted action graphs.
    2. Apply the pass pipeline, bucketing only collectives kept in this graph.

    The returned pass is later configured for each extracted action graph.
    """
    # Paged stash sizes its buffers by replaying the PP schedule over a
    # per-stage footprint, so the slots this stage's passes register have to be
    # attributed to this stage.
    PagedStashManager.get_instance().current_stage_index = stage.stage_index

    compile_config: GraphTrainerCompileConfig = config.compile
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        [
            eliminate_dead_code_pass,
            canonicalize_graph_pass,
            deduplicate_fsdp_unshard_chains_pass,
        ],
        compile_config=compile_config,
        respect_disable_passes=False,
    )

    if not compile_config.enable_passes:
        traced.gm = apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            construct_mandatory_graph_passes(),
            compile_config=compile_config,
            respect_disable_passes=False,
        )
        return None

    passes = compile_time_passes(
        traced,
        config,
        use_cuda_graph=False,
        parallelism_context=parallelism_context,
        include_inductor=False,
        include_mandatory_normalization=False,
    )

    fsdp_bucketing_pass: Callable | None = None
    if split_fsdp_param_unshard or split_fsdp_grad_reduction:
        # Step 1: retain the original pass for extracted action graphs.
        fsdp_bucketing_pass = _find_fsdp_bucketing_pass(passes)

    # Step 2: apply the pipeline without bucketing extracted collectives.
    _apply_passes_with_extracted_fsdp_bucketing(
        traced,
        passes,
        compile_config=compile_config,
        fsdp_bucketing_pass=fsdp_bucketing_pass,
        bucket_all_gathers=not split_fsdp_param_unshard,
        bucket_reduce_scatters=not split_fsdp_grad_reduction,
        bucket_all_reduces=not split_fsdp_grad_reduction,
    )
    return fsdp_bucketing_pass
