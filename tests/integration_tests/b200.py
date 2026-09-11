# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan_recipes.tests.suites.b200 as recipes

from tests.integration_tests import IntegrationTestDefinition


def build_b200_tests_list() -> list[IntegrationTestDefinition]:
    """Build integration tests that require B200-class hardware."""
    return [
        IntegrationTestDefinition(
            configs=[recipes.kimi_k3_debugmodel_mm],
            test_descr="Kimi K3 multimodal SPMD-typed FSDP, TP and EP",
            test_name="kimi_k3_mm",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[recipes.kimi_k3_debugmodel_mm_muon],
            test_descr="Kimi K3 multimodal per-head DistMuon FSDP+EP numerics",
            test_name="kimi_k3_mm_muon",
            ngpu=2,
            golden_numerics_path="tests/assets/losses/real_pg/kimi_k3_b200.txt",
            loss_compare_seed_config=recipes.kimi_k3_debugmodel_mm,
        ),
        IntegrationTestDefinition(
            configs=[recipes.kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4],
            test_descr="Kimi K3 FSDP, TP, EP and PP with 4 VPP stages",
            test_name="kimi_k3_fsdp2_tp2_ep2_pp2_vpp4",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[recipes.llama3_debugmodel_mxfp8_fsdp2],
            test_descr="MXFP8 linear with an FSDP-managed weight cache",
            test_name="mxfp8_linear_fsdp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[recipes.llama3_debugmodel_nvfp4_fsdp2],
            test_descr="NVFP4 linear with an FSDP-managed weight cache",
            test_name="nvfp4_linear_fsdp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[recipes.kimi_k3_debugmodel_mm_allgather_kv_cp2],
            test_descr="Kimi K3 multimodal K/V all-gather context parallelism",
            test_name="kimi_k3_mm_allgather_kv_cp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[recipes.kimi_k3_debugmodel_mm_ulysses_cp2],
            test_descr="Kimi K3 multimodal Ulysses context parallelism",
            test_name="kimi_k3_mm_ulysses_cp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[
                recipes.deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2,
                recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2,
                recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm,
            ],
            test_descr="Eager BF16, MXFP8, and VMM Dist-MoE with FSDP and EP",
            test_name="dist_moe_eager_fsdp_ep_cudagraph",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2],
            test_descr="Eager MXFP8 Dist-MoE PP with BF16 reduction",
            test_name="dist_moe_eager_fsdp_ep_pp_cudagraph",
            ngpu=4,
            use_real_pg=True,
        ),
    ]
