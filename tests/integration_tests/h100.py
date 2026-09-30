# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan_recipes.tests.suites.h100 as recipes

from tests.integration_tests import IntegrationTestDefinition


def build_h100_tests_list() -> list[IntegrationTestDefinition]:
    """
    Build the list of integration tests that need H100-class hardware.

    Each entry names one configuration per run; see ``torchtitan_recipes.tests.suites.h100``.
    """
    return [
        IntegrationTestDefinition(
            configs=[recipes.llama3_debugmodel_fsdp_symm_mem],
            test_descr="FSDP symmetric memory",
            test_name="fsdp_symm_mem",
            ngpu=2,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[recipes.deepseek_v3_debugmodel_hybridep_fsdp4_ep2],
            test_descr="DeepSeek V3 FSDP+HybridEP",
            test_name="deepseek_v3_fsdp+hybridep",
            ngpu=4,
            # deep_ep/NVSHMEM is CUDA-only, so skip on ROCm.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[recipes.llama3_debugmodel_dist_gemm_tp2],
            test_descr="Dist GEMM: fuse the TP collectives into the attention "
            "and FFN projections (FSDP2 + TP2)",
            test_name="dist_gemm",
            ngpu=4,
            # symmetric memory is CUDA-only.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[recipes.qwen3_moe_deepep_fsdp4_ep4],
            test_descr="Qwen3 FSDP+DeepEP",
            test_name="qwen3_fsdp+deepep",
            ngpu=4,
            skip_rocm_test=True,
        ),
    ]
