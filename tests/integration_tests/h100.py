# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from tests.integration_tests import OverrideDefinitions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_h100_tests_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "2D async TP compile",
            "2d_asynctp_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_float8",
                ],
            ],
            "Float8 test",
            "float8",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.spmd_backend full_dtensor",
                    "--parallelism.enable-fsdp-symm-mem",
                ],
            ],
            "FSDP symmetric memory",
            "fsdp_symm_mem",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_float8",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "FSDP+async TP+PP+torch.compile+Float8",
            "fsdp+tp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_float8",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "HSDP+CP+torch.compile+Float8",
            "hsdp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel_hybridep",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                    "--compile.components model,loss",
                ],
            ],
            "DeepSeek V3 FSDP+HybridEP+compile",
            "deepseek_v3_fsdp+hybridep+compile",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors
