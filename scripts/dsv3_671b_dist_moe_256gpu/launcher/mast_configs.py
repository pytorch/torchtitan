# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose and specialize recipes archived in the pinned TorchTitan runtime."""

from scripts.dsv3_671b_dist_moe_256gpu import mast_configs as _mast_configs
from scripts.dsv3_671b_dist_moe_256gpu.mast_configs import *  # noqa: F401, F403


def _configure_single_step_pp2_profile(config):
    """Record one final-geometry PP2 step without a Kineto warmup cycle.

    A 120-microbatch PP2 step can fill Kineto's default activity buffer before
    a multi-step warmup finishes. The launcher pairs this recipe with a larger
    profiling-only buffer through ``TORCHTITAN_KINETO_MAX_GPU_BUFFER_SIZE_MB``.
    """
    config.profiler.profile_freq = 41
    config.profiler.profiler_warmup = 0
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_single_step_profile():
    """Profile one eager final-geometry PP2 MTP1 training step."""
    config = (
        _mast_configs.deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()
    )
    return _configure_single_step_pp2_profile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_single_step_profile():
    """Profile one GraphTrainer final-geometry PP2 MTP1 training step."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()
    )
    return _configure_single_step_pp2_profile(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_full_lookahead_performance():
    """Measure GraphTrainer PP2 MTP1 with the complete residency lookahead."""
    config = (
        _mast_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance()
    )
    config.parallelism.pp_num_unshard_lookahead_factor = "full"
    return config
