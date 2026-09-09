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
            configs=[recipes.kimi_k3_debugmodel_mm_fsdp2],
            test_descr="Kimi K3 multimodal SPMD-typed FSDP",
            test_name="kimi_k3_mm_fsdp",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_mxfp8_fsdp2],
            test_descr="MXFP8 linear with an FSDP-managed weight cache",
            test_name="mxfp8_linear_fsdp",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.graph_trainer_llama3_debugmodel_mxfp8_fsdp2_pp2],
            test_descr="GraphTrainer MXFP8 with FSDP and pipeline parallelism",
            test_name="graph_trainer_mxfp8_fsdp+pp",
            ngpu=4,
            use_real_pg=True,
        ),
    ]
