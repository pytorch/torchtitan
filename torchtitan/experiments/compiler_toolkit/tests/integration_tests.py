# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_compiler_toolkit_test_list() -> list[OverrideDefinitions]:
    """
    returns a list of OverrideDefinitions that is used to generate
    variations of integration tests based on the same root config file.
    """
    integration_tests_flavors = [
        # llama3 tests
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "llama3 FSDP+TP",
            "llama3_fsdp_tp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes autobucketing_reordering",
                ],
            ],
            "llama3 FSDP+TP autobucketing",
            "llama3_fsdp_tp_autobucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing",
                ],
            ],
            "llama3 FSDP+TP manualbucketing",
            "llama3_fsdp_tp_manualbucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes cudagraph",
                ],
            ],
            "llama3 FSDP+TP+cudagraph",
            "llama3_fsdp_tp_cudagraph",
            ngpu=4,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "llama3 FSDP+TP+FlexAttn",
            "llama3_fsdp_tp_flexattn",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes autobucketing_reordering,regional_inductor",
                ],
            ],
            "llama3 FSDP+TP+FlexAttn autobucketing regional_inductor",
            "llama3_fsdp_tp_flexattn_autobucketing_regional_inductor",
            ngpu=4,
        ),
        # TODO: enable this when cudagraph is fixed
        # OverrideDefinitions(
        #     [
        #         [
        #             "--module compiler_toolkit.llama3",
        #             "--config compiler_toolkit_llama3_debugmodel_flex_attn",
        #             "--parallelism.data_parallel_shard_degree 2",
        #             "--parallelism.tensor_parallel_degree 2",
        #             "--compile.passes autobucketing_reordering,regional_inductor,cudagraph",
        #         ],
        #     ],
        #     "llama3 FSDP+TP+FlexAttn autobucketing regional_inductor+cudagraph",
        #     "llama3_fsdp_tp_flexattn_autobucketing_regional_inductor_cudagraph",
        #     ngpu=4,
        # ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.joint_passes inductor_decomposition",
                    "--compile.passes full_inductor_compilation",
                ],
            ],
            "llama3 full_inductor_compilation",
            "llama3_full_inductor_compilation",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.llama3",
                    "--config compiler_toolkit_llama3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing,regional_inductor",
                ],
            ],
            "llama3 FSDP+TP+FlexAttn manualbucketing regional_inductor",
            "llama3_fsdp_tp_flexattn_manualbucketing_regional_inductor",
            ngpu=4,
        ),
        # deepseek_v3 tests
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.deepseek_v3",
                    "--config compiler_toolkit_deepseek_v3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--activation_checkpoint.mode none",
                ],
            ],
            "deepseek_v3 FSDP+TP+EP",
            "deepseekv3_fsdp_tp_ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module compiler_toolkit.deepseek_v3",
                    "--config compiler_toolkit_deepseek_v3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--activation_checkpoint.mode none",
                ],
            ],
            "deepseek_v3 FSDP+TP+EP+FlexAttention",
            "deepseekv3_fsdp_tp_ep_flexattention",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "compiler_toolkit": build_compiler_toolkit_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--gpu_arch_type",
        default="cuda",
        choices=["cuda", "rocm"],
        help="GPU architecture type. Must be specified as either 'cuda' or 'rocm'.",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION["compiler_toolkit"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
