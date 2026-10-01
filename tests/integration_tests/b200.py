# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan_recipes.tests.b200 as recipes

from tests.integration_tests import OverrideDefinitions


def build_b200_tests_list() -> list[OverrideDefinitions]:
    """Build integration tests that require B200-class hardware."""
    return [
        OverrideDefinitions(
            configs=[recipes.kimi_k3_debugmodel_mm],
            test_descr="Kimi K3 multimodal SPMD-typed FSDP, TP and EP",
            test_name="kimi_k3_mm",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.kimi_k3_debugmodel_mm_muon],
            test_descr="Kimi K3 multimodal per-head DistMuon FSDP and EP",
            test_name="kimi_k3_mm_muon",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4],
            test_descr="Kimi K3 FSDP, TP, EP and PP with 4 VPP stages",
            test_name="kimi_k3_fsdp2_tp2_ep2_pp2_vpp4",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_mxfp8_fsdp2],
            test_descr="MXFP8 linear with an FSDP-managed weight cache",
            test_name="mxfp8_linear_fsdp",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_nvfp4_fsdp2],
            test_descr="NVFP4 linear with an FSDP-managed weight cache",
            test_name="nvfp4_linear_fsdp",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2],
            test_descr=(
                "BF16 Dist-MoE in-place WGRAD accumulation with FSDP, EP, "
                "and CUDA graphs"
            ),
            test_name="dist_moe_bf16_fsdp_ep_cudagraph",
            ngpu=2,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2],
            test_descr=(
                "MXFP8 Dist-MoE in-place WGRAD accumulation with FSDP, EP, "
                "and CUDA graphs"
            ),
            test_name="dist_moe_mxfp8_fsdp_ep_cudagraph",
            ngpu=2,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2
            ],
            test_descr="BF16 Dist-MoE with non-pipeline GraphTrainer",
            test_name="graph_trainer_dist_moe_bf16_fsdp_ep",
            ngpu=2,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2
            ],
            test_descr="MXFP8 Dist-MoE with non-pipeline GraphTrainer",
            test_name="graph_trainer_dist_moe_mxfp8_fsdp_ep",
            ngpu=2,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2],
            test_descr="MXFP8 Dist-MoE with eager PP activation-slot reuse",
            test_name="dist_moe_mxfp8_fsdp_ep_pp_cudagraph",
            ngpu=4,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2_fp32_reduce
            ],
            test_descr=("MXFP8 Dist-MoE with eager PP, BF16 WGrad, and FP32 reduction"),
            test_name="dist_moe_mxfp8_fsdp_ep_pp_fp32_reduce_cudagraph",
            ngpu=4,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2
            ],
            test_descr="MXFP8 Dist-MoE with GraphPP activation-slot reuse",
            test_name="graph_trainer_dist_moe_mxfp8_fsdp_ep_pp_cudagraph",
            ngpu=4,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm],
            test_descr="MXFP8 Dist-MoE with host-backed VMM scratch preallocation",
            test_name="dist_moe_mxfp8_fsdp_ep_cudagraph_vmm",
            ngpu=2,
            use_real_pg=True,
        ),
    ]
