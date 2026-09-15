# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scale tests that require GB300-class hardware and a Slurm launcher."""

from tests.integration_tests import OverrideDefinitions


_ARTIFACT_OPTIONS = (
    "--debug.save_config_file=resolved_config.json",
    "--profiler.enable_profiling",
    "--profiler.profile_freq=10",
    "--profiler.profiler_warmup=3",
    "--profiler.profiler_active=1",
    "--profiler.enable_memory_snapshot",
    "--profiler.memory_snapshot_freq=10",
)


def build_gb300_dsv3_tests_list() -> list[OverrideDefinitions]:
    """Build the staged DeepSeek V3 GB300 scale-validation suite.

    B200 CI covers the components on one node. This suite adds real Slurm,
    cross-node PP/FSDP/EP communication, and golden numerics coverage before an
    expensive 256-GPU run is allowed to proceed.
    """
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_671b_pp4_ep32_mxfp8,
        deepseek_v3_debugmodel_mxfp8_fsdp8_pp2_ep8,
    )

    return [
        OverrideDefinitions(
            test_name="deepseek_v3_mxfp8_pp2_ep8_loss_compile",
            test_descr="DeepSeek V3 MXFP8 with PP2, FSDP8, EP8, and loss compile.",
            configs=(deepseek_v3_debugmodel_mxfp8_fsdp8_pp2_ep8,),
            override_args=(_ARTIFACT_OPTIONS,),
            ngpu=16,
            golden_numerics_path="deepseek_v3_mxfp8_16gpu.txt",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            test_name="deepseek_v3_671b_pp4_ep32_mxfp8",
            test_descr="DeepSeek V3 671B starting configuration on 256 GPUs.",
            configs=(deepseek_v3_671b_pp4_ep32_mxfp8,),
            override_args=(_ARTIFACT_OPTIONS,),
            ngpu=256,
            golden_numerics_path="deepseek_v3_671b_pp4_ep32_mxfp8.txt",
            use_real_pg=True,
        ),
    ]
