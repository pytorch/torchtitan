# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Matched four-GPU MTP1 recipes for deterministic eager/GraphTrainer checks."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Literal

from torchtitan.components.data import GrainDataLoader
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import MTPDecoder, MTPLoss
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from .mast_configs import (
    _base_local_cat_verification,
    _configure_dist_moe_runtime,
    _configure_mtp1,
    _configure_round_robin_routing,
    deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mtp1_pp1_verification_4gpu,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu,
)
from .paired_numerics_runtime import install_gradient_sha256_capture


_OUTPUT_ROOT_ENV = "TORCHTITAN_PAIRED_NUMERICS_OUTPUT_ROOT"
_CUDA_GRAPHS_ENV = "TORCHTITAN_PAIRED_NUMERICS_CUDA_GRAPHS"
_PP1_ACCUMULATION_STEPS_ENV = "TORCHTITAN_PAIRED_NUMERICS_PP1_ACCUMULATION_STEPS"


def _read_bool_env(name: str) -> bool:
    value = os.environ.get(name)
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be set to 0 or 1")
    return value == "1"


def _stable_repr(value: object) -> str:
    """Remove process-local function addresses from config representations."""
    return re.sub(r" at 0x[0-9a-fA-F]+>", ">", repr(value))


def _pp1_accumulation_steps() -> int:
    raw_value = os.environ.get(_PP1_ACCUMULATION_STEPS_ENV)
    if raw_value is None:
        raise RuntimeError(
            f"{_PP1_ACCUMULATION_STEPS_ENV} is required; " "use run_paired_numerics.py"
        )
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{_PP1_ACCUMULATION_STEPS_ENV} must be a positive integer"
        ) from error
    if value <= 0:
        raise ValueError(f"{_PP1_ACCUMULATION_STEPS_ENV} must be a positive integer")
    return value


def _set_deterministic_gate(config: Trainer.Config) -> Trainer.Config:
    config.training.disable_cuda_graphs = not _read_bool_env(_CUDA_GRAPHS_ENV)
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.profiler = Profiler.Config(enable_profiling=False)
    return config


def _eager_pp1_config() -> Trainer.Config:
    reference = (
        graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mtp1_pp1_verification_4gpu()
    )
    config = _base_local_cat_verification(seq_len=512)
    _configure_dist_moe_runtime(
        config,
        inplace_wgrad_accum=False,
        activation_slot_capacity_factor=0.5,
    )
    _configure_round_robin_routing(config)
    _configure_mtp1(config)
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.num_tokens_per_train_step = 512 * 4 * _pp1_accumulation_steps()
    config.model.local_compile_regions = []
    config.activation_checkpoint = copy.deepcopy(reference.activation_checkpoint)
    config.dataloader = copy.deepcopy(reference.dataloader)
    config.hf_assets_path = reference.hf_assets_path
    config.comm.trace_buf_size = reference.comm.trace_buf_size
    return _set_deterministic_gate(config)


def _graph_trainer_pp1_config() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mtp1_pp1_verification_4gpu()
    config.parallelism.expert_parallel_degree = 2
    config.training.num_tokens_per_train_step = 512 * 4 * _pp1_accumulation_steps()
    _set_deterministic_gate(config)
    return config


def _eager_pp2_config() -> Trainer.Config:
    config = deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu()
    config.model.local_compile_regions = []
    config.parallelism.fsdp_defer_gradient_reduction = True
    return _set_deterministic_gate(config)


def _graph_trainer_pp2_config() -> GraphTrainer.Config:
    config = (
        graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu()
    )
    _set_deterministic_gate(config)
    return config


def _common_contract(config: Trainer.Config, topology: str) -> dict[str, object]:
    parallelism = config.parallelism
    training = config.training
    model = config.model
    if not isinstance(model, MTPDecoder.Config):
        raise TypeError("MTP1 paired numerics requires MTPDecoder.Config")
    dataloader = config.dataloader
    if not isinstance(dataloader, GrainDataLoader.Config):
        raise TypeError("MTP1 paired numerics requires GrainDataLoader.Config")
    loss = config.loss
    if not isinstance(loss, MTPLoss.Config):
        raise TypeError("MTP1 paired numerics requires MTPLoss.Config")
    routed_aux_coeffs = []
    for _, moe, _, _ in model.traverse(MoE.Config):
        if moe.router.aux_loss is None:
            raise ValueError("MTP1 paired numerics requires router auxiliary loss")
        routed_aux_coeffs.append(moe.router.aux_loss.coeff)
    if not routed_aux_coeffs or any(coeff != 0.01 for coeff in routed_aux_coeffs):
        raise ValueError(
            "MTP1 paired numerics requires auxiliary coefficient 0.01 "
            "at every routed depth"
        )
    if len(model.mtp_layers) != 1 or dataloader.num_mtp_layers != 1:
        raise ValueError("paired numerics requires exactly one MTP depth")
    if loss.mtp_scale != 0.1:
        raise ValueError("paired numerics requires MTP loss scale 0.1")
    return {
        "topology": topology,
        "debug": {
            "seed": config.debug.seed,
            "deterministic": config.debug.deterministic,
            "deterministic_warn_only": config.debug.deterministic_warn_only,
        },
        "training": {
            "cuda_graphs_enabled": not training.disable_cuda_graphs,
            "dtype": training.dtype,
            "mixed_precision_param": training.mixed_precision_param,
            "max_context_length": training.max_context_length,
            "tokens_per_microbatch_per_dp_rank": (
                training.num_tokens_per_microbatch_per_dp_rank
            ),
            "tokens_per_train_step": training.num_tokens_per_train_step,
        },
        "parallelism": {
            "dp_replicate": parallelism.data_parallel_replicate_degree,
            "dp_shard": parallelism.data_parallel_shard_degree,
            "ep": parallelism.expert_parallel_degree,
            "tp": parallelism.tensor_parallel_degree,
            "cp": parallelism.context_parallel_degree,
            "pp": parallelism.pipeline_parallel_degree,
            "pp_schedule": parallelism.pipeline_parallel_schedule,
            "num_pp_microbatches": parallelism.num_pp_microbatches,
            "stage_modules": (parallelism.pipeline_parallel_module_fqns_per_model_part),
            "fsdp_reshard_after_forward": (parallelism.fsdp_reshard_after_forward),
        },
        "data": {
            "dataloader_type": type(dataloader).__qualname__,
            "dataset": _stable_repr(dataloader.dataset),
            "shuffle": dataloader.shuffle,
            "repeat": dataloader.repeat,
            "max_num_documents": dataloader.max_num_documents,
            "hf_assets_path": config.hf_assets_path,
        },
        "model_objective": {
            "num_mtp_layers": len(model.mtp_layers),
            "num_data_mtp_layers": dataloader.num_mtp_layers,
            "mtp_scale": loss.mtp_scale,
            "num_routed_depths": len(routed_aux_coeffs),
            "routed_aux_coeffs": routed_aux_coeffs,
        },
        "activation_checkpoint": _stable_repr(config.activation_checkpoint),
    }


def _implementation_contract(
    config: Trainer.Config,
    scenario: str,
    topology: Literal["pp1", "pp2"],
) -> dict[str, object]:
    inplace_wgrad = []
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        inplace_wgrad.append(experts.inplace_wgrad_accum)
    contract: dict[str, object] = {
        "scenario": scenario,
        "config_type": type(config).__qualname__,
        "fsdp_defer_gradient_reduction": (
            config.parallelism.fsdp_defer_gradient_reduction
        ),
        "dist_moe_inplace_wgrad": inplace_wgrad,
    }
    if scenario == "eager":
        if not config.parallelism.fsdp_defer_gradient_reduction:
            raise ValueError("the eager paired control requires deferred reduction")
        if config.model.local_compile_regions != []:
            raise ValueError(
                "the eager paired control requires model.local_compile_regions=[]"
            )
        expected_inplace_wgrad = topology == "pp2"
        if any(value != expected_inplace_wgrad for value in inplace_wgrad):
            raise ValueError(
                f"the eager {topology} control requires DistMoE "
                f"inplace_wgrad_accum={expected_inplace_wgrad}"
            )
        contract["local_compile_regions"] = []
    else:
        if not isinstance(config, GraphTrainer.Config):
            raise TypeError("the GraphTrainer scenario requires GraphTrainer.Config")
        if config.parallelism.fsdp_defer_gradient_reduction:
            raise ValueError("GraphTrainer must own gradient reduction explicitly")
        if any(inplace_wgrad):
            raise ValueError("GraphTrainer paired numerics requires graph-owned WGrad")
        if config.compile.numerics_changing_optim:
            raise ValueError("paired numerics forbids numerics-changing graph passes")
        wgrad_fusion = (
            config.compile.spmd_gradient_accumulation.fuse_wgrad_accumulation
        )
        if wgrad_fusion != "enabled":
            raise ValueError("paired numerics requires GraphTrainer WGrad fusion")
        if (
            config.compile.spmd_gradient_accumulation.fsdp_grad_reduce_mode
            != "last_microbatch"
        ):
            raise ValueError(
                "paired numerics requires last-microbatch GraphTrainer reduction"
            )
        contract.update(
            {
                "fuse_wgrad_accumulation": wgrad_fusion,
                "inductor_compilation": config.compile.inductor_compilation,
                "numerics_changing_optim": (config.compile.numerics_changing_optim),
                "fsdp_grad_reduce_mode": (
                    config.compile.spmd_gradient_accumulation.fsdp_grad_reduce_mode
                ),
            }
        )
    return contract


def _validate_topology(config: Trainer.Config, topology: str) -> None:
    parallelism = config.parallelism
    expected = {
        "pp1": (4, 2, 1),
        "pp2": (2, 2, 2),
    }[topology]
    actual = (
        parallelism.data_parallel_shard_degree,
        parallelism.expert_parallel_degree,
        parallelism.pipeline_parallel_degree,
    )
    if actual != expected:
        raise ValueError(
            f"{topology} requires (dp_shard, ep, pp)={expected}, got {actual}"
        )
    if topology == "pp2" and parallelism.num_pp_microbatches != 32:
        raise ValueError("PP2 paired numerics requires 32 pipeline microbatches")


def _write_contract(
    config: Trainer.Config,
    *,
    topology: Literal["pp1", "pp2"],
    scenario: Literal["eager", "graph_trainer"],
) -> None:
    if _OUTPUT_ROOT_ENV not in os.environ:
        raise RuntimeError(
            f"{_OUTPUT_ROOT_ENV} is required; use run_paired_numerics.py"
        )
    _validate_topology(config, topology)
    common = _common_contract(config, topology)
    if common["debug"] != {
        "seed": 42,
        "deterministic": True,
        "deterministic_warn_only": False,
    }:
        raise ValueError("paired numerics requires deterministic seed 42")
    rank = int(os.environ.get("RANK", "0"))
    output_path = (
        Path(os.environ[_OUTPUT_ROOT_ENV])
        / "config_contracts"
        / scenario
        / f"rank_{rank}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "rank": rank,
                "common": common,
                "implementation": _implementation_contract(config, scenario, topology),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _finalize_recipe(
    config: Trainer.Config,
    *,
    topology: Literal["pp1", "pp2"],
    scenario: Literal["eager", "graph_trainer"],
) -> Trainer.Config:
    _write_contract(config, topology=topology, scenario=scenario)
    install_gradient_sha256_capture(scenario)
    return config


def eager_mtp1_pp1_paired_numerics_4gpu() -> Trainer.Config:
    """Return the deterministic eager PP1/DP4/EP2 MTP1 control."""
    return _finalize_recipe(
        _eager_pp1_config(),
        topology="pp1",
        scenario="eager",
    )


def graph_trainer_mtp1_pp1_paired_numerics_4gpu() -> GraphTrainer.Config:
    """Return the matched GraphTrainer PP1/DP4/EP2 MTP1 recipe."""
    config = _graph_trainer_pp1_config()
    _finalize_recipe(config, topology="pp1", scenario="graph_trainer")
    return config


def eager_mtp1_pp2_paired_numerics_4gpu() -> Trainer.Config:
    """Return the deterministic eager PP2/VPP8/DP2/EP2 MTP1 control."""
    return _finalize_recipe(
        _eager_pp2_config(),
        topology="pp2",
        scenario="eager",
    )


def graph_trainer_mtp1_pp2_paired_numerics_4gpu() -> GraphTrainer.Config:
    """Return the matched GraphTrainer PP2/VPP8/DP2/EP2 MTP1 recipe."""
    config = _graph_trainer_pp2_config()
    _finalize_recipe(config, topology="pp2", scenario="graph_trainer")
    return config
