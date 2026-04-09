# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def _build_llama3_tests() -> list[OverrideDefinitions]:
    """Llama3-based integration tests (run on default A10 machines)."""
    return [
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=transformer_block_bucketing",
                ],
            ],
            "JIT 1D+transformer_block_bucketing",
            "jit_1d_transformer_block_bucketing",
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
                # load at [tp:4].
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
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
        # === AOT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "AOT llama3 FSDP+TP",
            "aot_llama3_fsdp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes auto_bucketing",
                ],
            ],
            "AOT llama3 FSDP+TP autobucketing",
            "aot_llama3_fsdp_tp_autobucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing",
                ],
            ],
            "AOT llama3 FSDP+TP manualbucketing",
            "aot_llama3_fsdp_tp_manualbucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes cudagraph",
                ],
            ],
            "AOT llama3 FSDP+TP+cudagraph",
            "aot_llama3_fsdp_tp_cudagraph",
            ngpu=8,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn",
            "aot_llama3_fsdp_tp_flexattn",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes auto_bucketing,regional_inductor",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn autobucketing regional_inductor",
            "aot_llama3_fsdp_tp_flexattn_autobucketing_regional_inductor",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.joint_passes inductor_decomposition",
                    "--compile.passes auto_bucketing,full_inductor_compilation",
                ],
            ],
            "AOT llama3 auto_bucketing+full_inductor_compilation",
            "aot_llama3_auto_bucketing_full_inductor_compilation",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.passes transformer_block_bucketing,regional_inductor",
                ],
            ],
            "AOT llama3 FSDP+TP+FlexAttn manualbucketing regional_inductor",
            "aot_llama3_fsdp_tp_flexattn_manualbucketing_regional_inductor",
            ngpu=8,
        ),
        # === aot_fx_trace mode tests ===
        # Note: aot_fx_trace applies cudagraph by default, so skip_rocm_test=True.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP+cudagraph",
            "aot_fx_trace_llama3_fsdp_tp",
            ngpu=8,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel_flex_attn",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP+FlexAttn",
            "aot_fx_trace_llama3_fsdp_tp_flexattn",
            ngpu=8,
        ),
    ]


def _build_deepseek_v3_tests() -> list[OverrideDefinitions]:
    """DeepSeek-v3-based integration tests (require H100 machines)."""
    return [
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 8",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "JIT FSDP+EP",
            "jit_fsdp+ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                ],
            ],
            "JIT FSDP+TP+EP+ETP",
            "jit_fsdp+tp+ep+etp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "JIT FSDP+CP",
            "jit_fsdp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend inductor",
                    "--parallelism.tensor_parallel_degree 1",
                    "--parallelism.expert_parallel_degree 8",
                    "--compile.passes=auto_bucketing",
                ]
            ],
            "jit_deepseekv3_auto_bucketing",
            ngpu=8,
        ),
        # === AOT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                ],
            ],
            "AOT deepseek_v3 FSDP+TP+EP",
            "aot_deepseekv3_fsdp_tp_ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel_flex_attn",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                ],
            ],
            "AOT deepseek_v3 FSDP+TP+EP+FlexAttention",
            "aot_deepseekv3_fsdp_tp_ep_flexattention",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--compile.joint_passes inductor_decomposition",
                ],
            ],
            "AOT deepseek_v3 inductor_decomposition",
            "aot_deepseekv3_inductor_decomposition",
            ngpu=8,
        ),
        # === aot_fx_trace mode tests ===
        # Note: cudagraph is auto-skipped for DSv3 because MoE load-balancing
        # introduces CUDA→CPU transfers incompatible with CUDA graph capture.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+EP",
            "aot_fx_trace_deepseek_v3_fsdp_tp_ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel_flex_attn",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+EP+FlexAttn",
            "aot_fx_trace_deepseek_v3_fsdp_tp_ep_flexattn",
            ngpu=8,
        ),
    ]


def build_graph_trainer_test_list() -> list[OverrideDefinitions]:
    """All graph_trainer integration tests (Llama3 + DeepSeek-v3)."""
    return _build_llama3_tests() + _build_deepseek_v3_tests()


def build_graph_trainer_default_test_list() -> list[OverrideDefinitions]:
    """Llama3 tests only (for default A10 machines)."""
    return _build_llama3_tests()


def build_graph_trainer_h100_test_list() -> list[OverrideDefinitions]:
    """DeepSeek-v3 tests only (for H100 machines)."""
    return _build_deepseek_v3_tests()


_TEST_SUITES_FUNCTION = {
    "graph_trainer": build_graph_trainer_test_list,
    "graph_trainer_default": build_graph_trainer_default_test_list,
    "graph_trainer_h100": build_graph_trainer_h100_test_list,
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
        "--test_suite",
        default="graph_trainer",
        choices=list(_TEST_SUITES_FUNCTION.keys()),
        help="Which test suite to run (default: graph_trainer, which runs all tests)",
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

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
