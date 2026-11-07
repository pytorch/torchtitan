# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remote numerical fixtures validated on GB300 hosts."""

from torchtitan_recipes.tests.models.qwen3_8_flash_next import (
    qwen38_flash_next_debugmodel,
    qwen38_flash_next_debugmodel_tp2,
)

from tests.integration_tests import IntegrationTestDefinition


def build_gb300_tests_list() -> list[IntegrationTestDefinition]:
    return [
        IntegrationTestDefinition(
            configs=[qwen38_flash_next_debugmodel_tp2],
            test_descr="Qwen3.8-Flash-Next TP2+EP2+SP strict numerical regression",
            test_name="qwen3_8_flash_next_tp+ep_numerics",
            ngpu=2,
            use_real_pg=True,
            skip_rocm_test=True,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/"
                "qwen3_8_flash_next.txt"
            ),
            loss_compare_seed_config=qwen38_flash_next_debugmodel,
        ),
    ]
