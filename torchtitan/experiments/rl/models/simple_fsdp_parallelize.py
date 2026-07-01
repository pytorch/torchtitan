# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleFSDP parallelize path for the RL PolicyTrainer's aot_fx_trace mode.

RL's default parallelize (``torchtitan.models.qwen3.parallelize_qwen3``) uses
eager FSDP2, whose per-module forward/backward hooks cannot be captured by
``make_fx``. The aot_fx_trace / precompile path instead needs SimpleFSDP, which
expresses the data-parallel all-gather and reduce-scatter as traceable DTensor
``redistribute`` ops so the collectives show up as graph nodes.

This module reuses graph_trainer's SimpleFSDP helpers (both experiments) and
provides:

- ``parallelize_qwen3_simple_fsdp``: an RL-owned parallelize_fn (no PP, no CP,
  which precompile does not support).
- ``to_simple_fsdp_spec``: rewrites an already-built RL qwen3 ``ModelSpec`` to
  the SimpleFSDP path while keeping its converters. Converters (the fp32
  ``LMHeadCastConverter`` every RL config relies on, and
  ``BatchInvariantFlexConverter``) are applied at registry build time, so the
  spec's ``model`` config already carries them; wrapping it in
  ``GraphTrainerQwen3Model.Config`` preserves those fields and only adds the
  ``init_states`` override that runs weight init under
  ``disable_active_parametrization`` (required so init writes the underlying
  sharded parameters, not the replicated compute view).
"""

from __future__ import annotations

import dataclasses

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.qwen3.parallelize import annotate_qwen3
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_qwen3_simple_fsdp(
    model: GraphTrainerQwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> GraphTrainerQwen3Model:
    """Annotate, apply TP/EP + SimpleFSDP, and (no-op for aot_fx_trace) compile.

    The model must be on the meta device (or already fit in memory); the caller
    materializes and initializes weights afterwards.
    """
    if parallel_dims.pp_enabled:
        raise ValueError(
            "RL SimpleFSDP parallelize does not support pipeline parallelism "
            "(RL has no PP and aot_fx_trace does not support it)."
        )
    if parallel_dims.cp_enabled:
        raise ValueError(
            "RL SimpleFSDP parallelize does not support context parallelism "
            "(CooR precompile does not support CP; set "
            "--parallelism.context_parallel_degree 1)."
        )

    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product of "
        f"TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), "
        f"i.e. {parallel_dims.seq_len_divisor}."
    )

    annotate_qwen3(model)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # Applied unconditionally: the `fsdp` mesh always exists with a real backend
    # (even at degree 1), so MixedPrecisionPolicy's param_dtype cast (the bf16
    # forward that batch-invariant mode relies on) still applies in single-GPU
    # runs.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    # No-op for aot_fx_trace (graph capture happens at training time); kept so
    # the jit compile mode still works if ever selected.
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model


def to_simple_fsdp_spec(spec: ModelSpec) -> ModelSpec:
    """Rewrite an RL qwen3 ``ModelSpec`` for the SimpleFSDP aot_fx_trace path.

    Wraps ``spec.model`` (with converters already applied) in
    ``GraphTrainerQwen3Model.Config`` and swaps ``parallelize_fn`` to the
    SimpleFSDP one. All other spec fields (state_dict_adapter, etc.) are
    preserved so checkpoint load and weight sync are unchanged.
    """
    base_model = spec.model
    graph_model = GraphTrainerQwen3Model.Config(
        **{f.name: getattr(base_model, f.name) for f in dataclasses.fields(base_model)}
    )
    return dataclasses.replace(
        spec,
        model=graph_model,
        parallelize_fn=parallelize_qwen3_simple_fsdp,
    )
