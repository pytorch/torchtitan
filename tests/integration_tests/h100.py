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
        # TODO: re-enable this test once the async TP issue is fixed
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
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.converters quantize.linear.float8",
                    "--quantize.linear.float8.enable_fsdp_float8_all_gather",
                    "--quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp",
                ],
            ],
            "Float8 test",
            "float8",
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                    "--model.converters quantize.linear.float8",
                    "--quantize.linear.float8.enable_fsdp_float8_all_gather",
                    "--quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp",
                ],
            ],
            "FSDP+async TP+PP+torch.compile+Float8",
            "fsdp+tp+cp+compile+float8",
            ngpu=8,
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--model.converters quantize.linear.float8",
                    "--quantize.linear.float8.enable_fsdp_float8_all_gather",
                    "--quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp",
                ]
            ],
            "HSDP+CP+torch.compile+Float8",
            "hsdp+cp+compile+float8",
            ngpu=8,
        ),
    ]
    return integration_tests_flavors
