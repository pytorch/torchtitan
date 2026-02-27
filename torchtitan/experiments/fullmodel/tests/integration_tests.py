# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_fullmodel_test_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=auto_bucketing",
                ],
            ],
            "JIT 1D+auto_bucketing",
            "jit_1d_auto_bucketing",
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=transformer_block_bucketing",
                ],
            ],
            "JIT 1D+transformer_block_bucketing",
            "jit_1d_transformer_block_bucketing",
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--activation_checkpoint.mode selective",
                    "--activation_checkpoint.selective_ac_option op",
                ],
            ],
            "JIT 1D with selective op AC",
            "jit_1d_sac_op",
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--activation_checkpoint.mode full",
                ],
            ],
            "JIT 1D with full AC",
            "jit_1d_full_ac",
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "JIT 2D async TP",
            "jit_2d_asynctp",
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--training.steps 20",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "JIT PP+DP+TP 3D test with save/load resume ckpt",
            "jit_pp_dp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ]
            ],
            "JIT HSDP+TP",
            "jit_hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "JIT HSDP+CP (with dp_shard)",
            "jit_hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "JIT FSDP+TP+CP",
            "jit_fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
                # load at [tp:4].
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 4",
                    "--training.steps 30",
                ],
            ],
            "JIT Optional checkpoint",
            "jit_optional_checkpoint",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.deepseek_v3",
                    "--config fullmodel_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "JIT FSDP+EP",
            "jit_fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.deepseek_v3",
                    "--config fullmodel_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                ],
            ],
            "JIT FSDP+TP+EP+ETP",
            "jit_fsdp+tp+ep+etp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.deepseek_v3",
                    "--config fullmodel_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "JIT FSDP+CP",
            "jit_fsdp+cp",
            ngpu=4,
        ),
        # === AOT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes auto_bucketing",
                ],
            ],
            "AOT llama3 FSDP+TP autobucketing",
            "aot_llama3_fsdp_tp_autobucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing",
                ],
            ],
            "AOT llama3 FSDP+TP manualbucketing",
            "aot_llama3_fsdp_tp_manualbucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes cudagraph",
                ],
            ],
            "AOT llama3 FSDP+TP+cudagraph",
            "aot_llama3_fsdp_tp_cudagraph",
            ngpu=4,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn",
            "aot_llama3_fsdp_tp_flexattn",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes auto_bucketing,regional_inductor",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn autobucketing regional_inductor",
            "aot_llama3_fsdp_tp_flexattn_autobucketing_regional_inductor",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.joint_passes inductor_decomposition",
                    "--compile.passes full_inductor_compilation",
                ],
            ],
            "AOT llama3 full_inductor_compilation",
            "aot_llama3_full_inductor_compilation",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.llama3",
                    "--config fullmodel_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing,regional_inductor",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn manualbucketing regional_inductor",
            "aot_llama3_fsdp_tp_flexattn_manualbucketing_regional_inductor",
            ngpu=4,
        ),
        # deepseek_v3 AOT tests
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.deepseek_v3",
                    "--config fullmodel_deepseek_v3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--activation_checkpoint.mode none",
                ],
            ],
            "AOT deepseek_v3 FSDP+TP+EP",
            "aot_deepseekv3_fsdp_tp_ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module fullmodel.deepseek_v3",
                    "--config fullmodel_deepseek_v3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--activation_checkpoint.mode none",
                ],
            ],
            "AOT deepseek_v3 FSDP+TP+EP+FlexAttention",
            "aot_deepseekv3_fsdp_tp_ep_flexattention",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "fullmodel": build_fullmodel_test_list,
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

    test_list = _TEST_SUITES_FUNCTION["fullmodel"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
