# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import os

from tests.integration_tests import OverrideDefinitions


def _is_pp_only(variant: tuple[str, ...], ngpu: int) -> bool:
    """True when the variant has PP > 1 and no other SPMD parallelism > 1.

    full_dtensor requires at least one SPMD axis > 1; PP-only runs collapse
    every dense SPMD axis to size 1 and trip DTensor's reshape / flatten
    rejection of Shard-on-degenerate-axis. Detected by parsing the explicit
    ``--parallelism.*_degree`` flags and back-computing ``dp_shard`` (default
    -1, "fill remaining").
    """
    degrees = {
        "pipeline_parallel_degree": 1,
        "data_parallel_replicate_degree": 1,
        "data_parallel_shard_degree": -1,
        "tensor_parallel_degree": 1,
        "context_parallel_degree": 1,
        "expert_parallel_degree": 1,
    }
    for arg in variant:
        for key in degrees:
            if key in arg:
                try:
                    degrees[key] = int(arg.split()[-1])
                except ValueError:
                    pass
                break
    if degrees["data_parallel_shard_degree"] == -1:
        denom = (
            degrees["pipeline_parallel_degree"]
            * degrees["data_parallel_replicate_degree"]
            * degrees["context_parallel_degree"]
            * degrees["tensor_parallel_degree"]
        )
        degrees["data_parallel_shard_degree"] = max(1, ngpu // denom)
    return degrees["pipeline_parallel_degree"] > 1 and all(
        v <= 1 for k, v in degrees.items() if k != "pipeline_parallel_degree"
    )


def _enable_spmd_backend(t: OverrideDefinitions, backend: str) -> OverrideDefinitions:
    """Inject ``--parallelism.spmd_backend`` into every variant.

    All features.py tests run under SPMD backends except PP-only variants
    (see ``_is_pp_only``) and CP + compile variants (upstream symint
    limitation); legacy non-full_dtensor coverage lives in models.py.
    """
    test_name = t.test_name if backend == "full_dtensor" else f"{t.test_name}_{backend}"
    new_args = []
    for variant in t.override_args:
        prefix: list[str] = []
        has_cp = any("context_parallel_degree" in arg for arg in variant)
        has_compile = any("compile.enable" in arg for arg in variant)
        if not _is_pp_only(variant, t.ngpu) and not (has_cp and has_compile):
            prefix.append(f"--parallelism.spmd_backend {backend}")
        if test_name != t.test_name:
            variant = tuple(
                arg.replace(f"{t.test_name}/", f"{test_name}/")
                for arg in variant
            )
        new_args.append(tuple(prefix) + tuple(variant))
    return dataclasses.replace(
        t,
        override_args=tuple(new_args),
        test_name=test_name,
    )


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
                    "--profiler.enable_profiling",
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
                    "activation-checkpoint:selective",
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
                [
                    "--module llama3 --config llama3_debugmodel_ce_loss",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D eager (ChunkedCELoss + standard CE loss with TP+loss_parallel)",
            "2d_eager",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.no-enable-sequence-parallel",
                ],
            ],
            "2D eager (SP disabled)",
            "2d_eager_no_sp",
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
                    "--checkpoint.export_dtype bfloat16",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only bf16",
            "last_save_model_only_bf16",
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
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+TP GPipe 2D test",
            "pp_tp_gpipe",
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
        # TODO: Disabled with the FlexAttention default (SDPA is no longer a
        # language-model backend). Zero-bubble / multi schedules split backward
        # and call torch's stage_backward_input, which runs
        # _get_grad_fn_or_grad_acc (t.requires_grad) over every stage input —
        # including the forwarded FlexAttention BlockMask, which is not a Tensor
        # ("'BlockMask' object has no attribute 'requires_grad'"). Full-backward
        # schedules (1F1B/GPipe/Interleaved1F1B) are unaffected. Re-enable once
        # stage_backward_input skips non-tensor stage inputs upstream.
        # (VarlenAttention's tensor-based metadata would sidestep this, but
        # varlen requires flash_attn_interface/FA3, which the core integration
        # CI does not install; SDPA is no longer a core LM backend. So the
        # upstream stage_backward_input fix is the path here.)
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule InterleavedZeroBubble",
                    "activation-checkpoint:full",
                ],
            ],
            "PP looped zero bubble test",
            "pp_looped_zero_bubble",
            ngpu=4,
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
                    "activation-checkpoint:full",
                ],
            ],
            "PP zero bubble test (v shaped)",
            "pp_zbv",
            ngpu=2,
            disabled=True,
        ),
        # TODO: Disabled for the same reason as the zero-bubble PP tests above:
        # the custom CSV schedule splits backward (separate input-grad step),
        # so stage_backward_input chokes on the forwarded FlexAttention
        # BlockMask. Re-enable once stage_backward_input skips non-tensor inputs.
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule PipelineScheduleMulti",
                    "--parallelism.pipeline_parallel_schedule_csv ./tests/assets/custom_schedule.csv",
                    "activation-checkpoint:full",
                ],
            ],
            "PP with custom pipeline schedule loaded from CSV file",
            "pp_custom_csv",
            ngpu=2,
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--optimizer.implementation fused_opt_states_bf16",
                ]
            ],
            "BF16 Optimizer States Test",
            "optimizer_bf16_states",
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
                ]
            ],
            "CP",
            "cp",
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
                    "--override.imports torchtitan.overrides.fused_swiglu",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "Override: swap FeedForward with fused SwiGLU (FSDP2 + TP2)",
            "override_fused_swiglu",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--override.imports torchtitan.overrides.fused_swiglu",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "Override: fuse grouped experts + FFNs on deepseek_v3 "
            "(FSDP2 + TP2 dense, EP4 sparse)",
            "override_fused_grouped_experts",
            ngpu=4,
        ),
        # NOTE: below are tests which require config change that cannot be done
        #       via CLI overrides, so remain llama3 specific
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_varlen_attn",
                    "--parallelism.data_parallel_shard_degree=4",
                    "activation-checkpoint:selective",
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
                    "--module llama3 --config llama3_debugmodel_float8_emulate_lora",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                ],
            ],
            "Float8 emulate + LoRA training test",
            "float8_emulate_lora",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--comm.mode torchcomms",
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "FSDP+CP+PP+compile with torchcomms",
            "torchcomms_3d_dp+cp+pp+compile",
            ngpu=8,
            skip_rocm_test=True,
            # NotImplementedError: new_group cannot delegate to split_group
            # with use_local_synchronization=True; split_group requires all
            # ranks in the parent group to participate.
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config llama3_debugmodel_ce_loss",
                    "--comm.mode torchcomms",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "FSDP+TP+PP+compile with torchcomms",
            "torchcomms_3d_dp+tp+pp+compile",
            ngpu=8,
            skip_rocm_test=True,
            # torchcomms-managed TP PG not registered in c10d;
            # resolve fails under compile
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module llama3 --config sft_debugmodel",
                ],
            ],
            "SFT ChatDataset integration test",
            "sft",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.create_seed_checkpoint",
                ],
            ],
            "Seed checkpoint creation",
            "seed_checkpoint",
            ngpu=1,
            timeout=30,
        ),
    ]

    return [
        *[_enable_spmd_backend(t, "full_dtensor") for t in integration_tests_flavors],
        *[_enable_spmd_backend(t, "spmd_types") for t in integration_tests_flavors],
    ]
