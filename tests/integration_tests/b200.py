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
            test_descr="Kimi K3 multimodal per-head DistMuon FSDP and EP",
            test_name="kimi_k3_mm_muon",
            ngpu=2,
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
    ]
