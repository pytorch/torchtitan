# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan_recipes.tests.h100 as recipes
from torchtitan.models.llama3.config_registry import llama3_debugmodel_float8

from tests.integration_tests import OverrideDefinitions


def build_h100_tests_list() -> list[OverrideDefinitions]:
    """
    Build the list of integration tests that need H100-class hardware.

    Each entry names one configuration per run; see ``torchtitan_recipes.tests.h100``.
    """
    return [
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_tp2_asynctp_compile],
            test_descr="2D async TP compile",
            test_name="2d_asynctp_compile",
        ),
        OverrideDefinitions(
            configs=[llama3_debugmodel_float8],
            test_descr="Float8 test",
            test_name="float8",
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp_symm_mem],
            test_descr="FSDP symmetric memory",
            test_name="fsdp_symm_mem",
            ngpu=2,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_float8_fsdp2_tp2_pp2_asynctp_compile],
            test_descr="FSDP+async TP+PP+torch.compile+Float8",
            test_name="fsdp+tp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_float8_hsdp2x2_cp2_compile],
            test_descr="HSDP+CP+torch.compile+Float8",
            test_name="hsdp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_hybridep_fsdp4_ep2_compile],
            test_descr="DeepSeek V3 FSDP+HybridEP+compile",
            test_name="deepseek_v3_fsdp+hybridep+compile",
            ngpu=4,
            # deep_ep/NVSHMEM is CUDA-only, so skip on ROCm.
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_minimal_async_ep_fsdp2_tp2_cp2_ep8],
            test_descr="DeepSeek V3 FSDP+CP+TP+MinimalAsyncEP",
            test_name="deepseek_v3_fsdp+cp+tp+minimal_async_ep",
            ngpu=8,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_dist_gemm_tp2],
            test_descr="Dist GEMM: fuse the TP collectives into the attention "
            "and FFN projections (FSDP2 + TP2)",
            test_name="dist_gemm",
            ngpu=4,
            # symmetric memory is CUDA-only.
            skip_rocm_test=True,
        ),
    ]
