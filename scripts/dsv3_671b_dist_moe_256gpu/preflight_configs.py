#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate the report and full-scale numerics recipes without a CUDA device.

This script replaces the CUDA capability probe only while constructing config
objects. It does not initialize a model or authorize launching MXFP8 on an
unsupported GPU. Runtime preflight must still validate a GB300 device.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from torchtitan.config.transform import quantization


CONFIG_NAMES = (
    "deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance",
    "deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile",
    "deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance",
    "deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile",
    "deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics",
    "deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics",
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics",
)


def _routed_moes(config: Any) -> Iterable[Any]:
    for layer in [*config.model.layers, *config.model.mtp_layers]:
        if layer.moe is not None:
            yield layer.moe


def _summary(config_name: str, config: Any) -> dict[str, Any]:
    routed_moes = list(_routed_moes(config))
    data_source = config.dataloader.dataset.dataset.source
    pp_degree = config.parallelism.pipeline_parallel_degree
    effective_microbatches = (
        config.training.num_tokens_per_train_step
        // config.training.num_tokens_per_microbatch_per_dp_rank
        // config.parallelism.data_parallel_shard_degree
    )
    stage_parts = config.parallelism.pipeline_parallel_module_fqns_per_model_part
    virtual_stages = len(stage_parts) // pp_degree if pp_degree > 1 else 1
    is_graph_trainer = config_name.startswith("graph_trainer_")
    summary = {
        "config_name": config_name,
        "data_parallel_shard_degree": config.parallelism.data_parallel_shard_degree,
        "expert_parallel_degree": config.parallelism.expert_parallel_degree,
        "pipeline_parallel_degree": pp_degree,
        "pipeline_parallel_virtual_stages": virtual_stages,
        "configured_pp_microbatches": config.parallelism.num_pp_microbatches,
        "effective_microbatches": effective_microbatches,
        "tokens_per_step": config.training.num_tokens_per_train_step,
        "seed": config.debug.seed,
        "deterministic": config.debug.deterministic,
        "deterministic_warn_only": config.debug.deterministic_warn_only,
        "mtp_depth": len(config.model.mtp_layers),
        "data_target_depth": config.dataloader.num_mtp_layers,
        "mtp_loss_scale": config.loss.mtp_scale,
        "routed_moe_depths": len(routed_moes),
        "routed_aux_coefficients": sorted(
            {moe.router.aux_loss.coeff for moe in routed_moes}
        ),
        "router_types": sorted({type(moe.router).__qualname__ for moe in routed_moes}),
        "outer_cuda_graphs_enabled": not config.training.disable_cuda_graphs,
        "fsdp_defer_gradient_reduction": (
            config.parallelism.fsdp_defer_gradient_reduction
        ),
        "graph_trainer_wgrad_fusion": (
            config.compile.spmd_gradient_accumulation.fuse_wgrad_accumulation
            if is_graph_trainer
            else None
        ),
        "local_compile_regions": config.model.local_compile_regions,
        "metrics_log_frequency": config.metrics.log_freq,
        "tensorboard_enabled": config.metrics.enable_tensorboard,
        "override_imports": config.override.imports,
        "profiling_enabled": config.profiler.enable_profiling,
        "profile_frequency": config.profiler.profile_freq,
        "profile_warmup": config.profiler.profiler_warmup,
        "profile_active": config.profiler.profiler_active,
        "memory_snapshot_enabled": config.profiler.enable_memory_snapshot,
        "memory_snapshot_frequency": config.profiler.memory_snapshot_freq,
        "steps": config.training.steps,
        "tokenizer_path": config.hf_assets_path,
        "data_source_type": type(data_source).__qualname__,
        "data_source_path": data_source.path,
        "data_source_name": data_source.name,
        "data_source_kwargs": data_source.load_dataset_kwargs,
        "data_shuffle": config.dataloader.shuffle,
    }

    is_pp2 = "sanket" in config_name
    is_numerics = config_name.endswith("_numerics")
    expected = {
        "data_parallel_shard_degree": 128 if is_pp2 else 256,
        "expert_parallel_degree": 64,
        "pipeline_parallel_degree": 2 if is_pp2 else 1,
        "pipeline_parallel_virtual_stages": 8 if is_pp2 else 1,
        "effective_microbatches": 120 if is_pp2 else 16,
        "tokens_per_step": 62_914_560 if is_pp2 else 16_777_216,
        "seed": 14_536 if is_pp2 else 42,
        "mtp_depth": 1,
        "data_target_depth": 1,
        "mtp_loss_scale": 0.1,
        "routed_moe_depths": 59,
        "routed_aux_coefficients": [0.01],
        "outer_cuda_graphs_enabled": True,
        "fsdp_defer_gradient_reduction": not is_graph_trainer,
        "local_compile_regions": [],
        "metrics_log_frequency": 1 if is_numerics else 10,
        "override_imports": [
            "torchtitan_recipes.overrides.fused_mla.fused_mla",
            "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
        ],
        "steps": 60,
    }
    for field_name, expected_value in expected.items():
        actual_value = summary[field_name]
        assert actual_value == expected_value, (
            config_name,
            field_name,
            actual_value,
            expected_value,
        )
    assert summary["router_types"] == ["_RoundRobinDeepSeekV3Router.Config"]
    if is_graph_trainer:
        assert summary["graph_trainer_wgrad_fusion"] == "enabled"
    assert summary["deterministic"] is is_numerics
    assert summary["deterministic_warn_only"] is False
    assert summary["tensorboard_enabled"] is is_numerics
    assert summary["profiling_enabled"] is not is_numerics
    assert summary["memory_snapshot_enabled"] is not is_numerics
    if is_numerics:
        return summary
    profile_expected = {
        "profile_frequency": 43 if is_pp2 else 41,
        "profile_warmup": 3 if is_pp2 else 0,
        "profile_active": 2 if is_pp2 else 1,
        "memory_snapshot_frequency": 40 if is_pp2 else 41,
    }
    for field_name, expected_value in profile_expected.items():
        actual_value = summary[field_name]
        assert actual_value == expected_value, (
            config_name,
            field_name,
            actual_value,
            expected_value,
        )
    return summary


def main() -> None:
    quantization.has_cuda_capability = lambda *_args, **_kwargs: True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.dsv3_671b_dist_moe_256gpu import mast_configs

    summaries = [
        _summary(config_name, getattr(mast_configs, config_name)())
        for config_name in CONFIG_NAMES
    ]
    print(json.dumps(summaries, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
