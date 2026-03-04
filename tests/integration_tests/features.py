# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

from tests.integration_tests import OverrideDefinitions

# Use RUNNER_TEMP if defined (GitHub Actions variable), else fallback to old path
runner_temp = os.getenv("RUNNER_TEMP")
if runner_temp:
    checkpoint_path = os.path.join(
        runner_temp,
        "artifacts-to-be-uploaded/model_only_hf_checkpoint/hf_checkpoint/step-10/",
    )
else:
    checkpoint_path = (
        "artifacts-to-be-uploaded/model_only_hf_checkpoint/hf_checkpoint/step-10/"
    )


def build_features_test_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [
                    "--profiling.enable_profiling",
                    "--metrics.enable_tensorboard",
                ],
            ],
            "default",
            "default",
        ),
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                ],
            ],
            "1D compile",
            "1d_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--activation_checkpoint.mode selective",
                    "--activation_checkpoint.selective_ac_option op",
                ],
            ],
            "1D compile with selective op AC",
            "1d_compile_sac_op",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D eager",
            "2d_eager",
        ),
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D compile",
            "2d_compile",
        ),
        # TODO: re-enable this test once the async TP CI issue is fixed
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
                    "--checkpoint.enable",
                ],
                [
                    "--checkpoint.enable",
                    "--training.steps 20",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
            "full_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.folder hf_checkpoint",
                    "--checkpoint.last_save_model_only",
                    "--checkpoint.last_save_in_hf",
                ],
                [
                    "--checkpoint.enable",
                    f"--checkpoint.initial_load_path {checkpoint_path}",
                    "--checkpoint.initial_load_model_only",
                    "--checkpoint.initial_load_in_hf",
                ],
            ],
            "Checkpoint Integration Test - save load model only checkpoint in HF definition and format",
            "model_only_hf_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.last_save_model_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only fp32",
            "last_save_model_only_fp32",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.last_save_model_only",
                    "--checkpoint.export_dtype bfloat16",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only bf16",
            "last_save_model_only_bf16",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule InterleavedZeroBubble",
                ],
            ],
            "PP looped zero bubble test",
            "pp_looped_zero_bubble",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
                ],
            ],
            "PP zero bubble test (v shaped)",
            "pp_zbv",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test 1F1B",
            "pp_1f1b",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test GPipe",
            "pp_gpipe",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 4",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP 1F1B 2D test",
            "pp_dp_1f1b",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP GPipe 2D test",
            "pp_dp_gpipe",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+TP 2D test",
            "pp_tp",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--training.steps 20",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "PP+DP+TP 3D test with torch.compile",
            "3d_compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 1",
                ],
            ],
            "PP looped 1F1B test",
            "pp_looped_1f1b",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule PipelineScheduleMulti",
                    "--parallelism.pipeline_parallel_schedule_csv ./tests/assets/custom_schedule.csv",
                ],
            ],
            "PP with custom pipeline schedule loaded from CSV file",
            "pp_custom_csv",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--optimizer.name AdamW --optimizer.implementation foreach",
                ]
            ],
            "Foreach Optimizer Test",
            "optimizer_foreach",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=4",
                ]
            ],
            "DDP",
            "ddp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='allgather'",
                ]
            ],
            "CP (allgather)",
            "cp_allgather",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='alltoall'",
                ]
            ],
            "CP (alltoall)",
            "cp_alltoall",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                ]
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+CP",
            "fsdp+cp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (without dp_shard)",
            "hsdp+cp_without_dp_shard",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (with dp_shard)",
            "hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+TP+CP",
            "fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                ],
                [
                    # placeholder for the generation script's generate step
                ],
            ],
            "Generation script test",
            "test_generate",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.fsdp_reshard_after_forward always",
                ],
            ],
            "Test always resharding after forward pass",
            "fsdp_reshard_always",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
            ],
            "Optional checkpoint",
            "optional_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    # Local batch size = 8, and `ngpu=2`, so default
                    # global batch size = 8 * 2 = 16.
                    # To achieve 2 gradient accumulation steps, multiply
                    # default global batch size by 2. 16 * 2 = 32.
                    "--training.local_batch_size 8",
                    "--training.global_batch_size 32",
                ],
            ],
            "Gradient accumulation",
            "gradient_accumulation",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--validator.enable",
                    "--validator.dataloader.dataset c4_test",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--parallelism.pipeline_parallel_degree=2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
            ],
            "Validation test with tp, cp, pp",
            "validation_tp_cp_pp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--dataloader.num_workers",
                    "2",
                    "--dataloader.pin_memory",
                    "--dataloader.persistent_workers",
                    "--dataloader.prefetch_factor",
                    "4",
                ],
            ],
            "Dataloader kwargs (via CLI args)",
            "dataloader_kwargs",
            ngpu=2,
        ),
        # NOTE: below are tests which require config change that cannot be done
        #       via CLI overrides, so remain llama3 specific
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree=4",
                    "--activation_checkpoint.mode='full'",
                ]
            ],
            "FSDP+FLEX_ATTN",
            "fsdp+flex_attn",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree=4",
                    "--activation_checkpoint.mode=selective",
                    "--activation_checkpoint.selective_ac_option=op",
                ]
            ],
            "FSDP + FLEX + per op SAC",
            "fsdp+flex_attn+per_op_sac",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_varlen_attn",
                    "--parallelism.data_parallel_shard_degree=4",
                    "--activation_checkpoint.mode=selective",
                    "--activation_checkpoint.selective_ac_option=op",
                ]
            ],
            "FSDP+VARLEN_ATTN + per op SAC",
            "fsdp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_opt_in_bwd",
                    "--checkpoint.enable",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--training.enable_cpu_offload",
                ],
                [
                    "--module llama3 --config llama3_debugmodel_opt_in_bwd",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--training.enable_cpu_offload",
                ],
            ],
            "Enable CPU Offload, Optimizer in backward with TP, DP, CP",
            "cpu_offload+opt_in_bwd+TP+DP+CP",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_float8_emulate",
                ],
            ],
            "Float8 emulation test",
            "float8_emulation",
        ),
    ]

    return integration_tests_flavors
