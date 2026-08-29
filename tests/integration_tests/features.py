# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchtitan_recipes.tests.features as recipes

from tests.integration_tests import OverrideDefinitions


def build_features_test_list() -> list[OverrideDefinitions]:
    """
    Build the list of integration tests covering the core features of torchtitan.

    Each entry names one configuration per run; see ``torchtitan_recipes.tests.features``.
    """
    return [
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_default],
            test_descr="default",
            test_name="default",
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_compile],
            test_descr="1D compile",
            test_name="1d_compile",
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_compile_sac_op],
            test_descr="1D compile with selective op AC",
            test_name="1d_compile_sac_op",
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_tp2,
                recipes.llama3_debugmodel_ce_loss_tp2,
            ],
            test_descr=(
                "2D eager (ChunkedLossWrapper + standard CE loss with TP+loss_parallel)"
            ),
            test_name="2d_eager",
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_tp2_no_sp],
            test_descr="2D eager (SP disabled)",
            test_name="2d_eager_no_sp",
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_tp2_compile],
            test_descr="2D compile",
            test_name="2d_compile",
        ),
        # TODO: re-enable this test once the async TP CI issue is fixed
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_tp2_asynctp_compile_spmd_types],
            test_descr="2D async TP compile",
            test_name="2d_asynctp_compile",
            disabled=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_full_checkpoint_save,
                recipes.llama3_debugmodel_full_checkpoint_load,
            ],
            test_descr="Checkpoint Integration Test - Save Load Full Checkpoint",
            test_name="full_checkpoint",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_hf_checkpoint_save,
                recipes.llama3_debugmodel_hf_checkpoint_load,
            ],
            test_descr=(
                "Checkpoint Integration Test - save load model only checkpoint in "
                "HF definition and format"
            ),
            test_name="model_only_hf_checkpoint",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_last_save_model_only_bf16],
            test_descr="Checkpoint Integration Test - Save Model Only bf16",
            test_name="last_save_model_only_bf16",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_pp2_1f1b],
            test_descr="PP 1D test 1F1B",
            test_name="pp_1f1b",
            ngpu=2,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_fsdp2_pp2_1f1b,
                recipes.llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage,
            ],
            test_descr="PP+DP 1F1B 2D test",
            test_name="pp_dp_1f1b",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_tp2_pp2_gpipe],
            test_descr="PP+TP GPipe 2D test",
            test_name="pp_tp_gpipe",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_fsdp2_tp2_pp2_save,
                recipes.llama3_debugmodel_fsdp2_tp2_pp2_load,
            ],
            test_descr="PP+DP+TP 3D test with save/load resume ckpt",
            test_name="pp_dp_tp",
            ngpu=8,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_pp2_compile],
            test_descr="PP+DP+TP 3D test with torch.compile",
            test_name="3d_compile",
            ngpu=8,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_pp4_interleaved_1f1b,
                recipes.llama3_debugmodel_pp4_interleaved_1f1b_layers_per_stage,
            ],
            test_descr="PP looped 1F1B test",
            test_name="pp_looped_1f1b",
            ngpu=4,
            use_real_pg=True,
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
            configs=[recipes.llama3_debugmodel_pp4_zero_bubble],
            test_descr="PP looped zero bubble test",
            test_name="pp_looped_zero_bubble",
            ngpu=4,
            disabled=True,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_pp2_zbv],
            test_descr="PP zero bubble test (v shaped)",
            test_name="pp_zbv",
            ngpu=2,
            disabled=True,
            use_real_pg=True,
        ),
        # TODO: Disabled for the same reason as the zero-bubble PP tests above:
        # the custom CSV schedule splits backward (separate input-grad step),
        # so stage_backward_input chokes on the forwarded FlexAttention
        # BlockMask. Re-enable once stage_backward_input skips non-tensor inputs.
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_pp2_custom_csv],
            test_descr="PP with custom pipeline schedule loaded from CSV file",
            test_name="pp_custom_csv",
            ngpu=2,
            disabled=True,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_optimizer_bf16_states],
            test_descr="BF16 Optimizer States Test",
            test_name="optimizer_bf16_states",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_ddp4],
            test_descr="DDP",
            test_name="ddp",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_hsdp2x2],
            test_descr="HSDP",
            test_name="hsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_cp4],
            test_descr="CP",
            test_name="cp",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_hsdp2x2_tp2],
            test_descr="HSDP+TP",
            test_name="hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_cp2],
            test_descr="FSDP+CP",
            test_name="fsdp+cp",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_ddp2_cp2],
            test_descr="HSDP+CP (without dp_shard)",
            test_name="hsdp+cp_without_dp_shard",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_hsdp2x2_cp2],
            test_descr="HSDP+CP (with dp_shard)",
            test_name="hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_cp2],
            test_descr="FSDP+TP+CP",
            test_name="fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp_reshard_always],
            test_descr="Test always resharding after forward pass",
            test_name="fsdp_reshard_always",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[
                recipes.llama3_debugmodel_optional_checkpoint_save,
                recipes.llama3_debugmodel_optional_checkpoint_load_tp2,
            ],
            test_descr="Optional checkpoint",
            test_name="optional_checkpoint",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_gradient_accumulation],
            test_descr="Gradient accumulation",
            test_name="gradient_accumulation",
            ngpu=2,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_validation_tp2_cp2_pp2],
            test_descr="Validation test with tp, cp, pp",
            test_name="validation_tp_cp_pp",
            ngpu=8,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fused_swiglu_tp2],
            test_descr="Override: swap FeedForward with fused SwiGLU (FSDP2 + TP2)",
            test_name="override_fused_swiglu",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fused_grouped_experts_tp2_ep4],
            test_descr=(
                "Override: fuse grouped experts + FFNs on deepseek_v3 "
                "(FSDP2 + TP2 dense, EP4 sparse)"
            ),
            test_name="override_fused_grouped_experts",
            ngpu=4,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_varlen_attn_fsdp4_sac],
            test_descr="FSDP+VARLEN_ATTN + per op SAC",
            test_name="fsdp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_float8_emulate_lora_tp2_pp2],
            test_descr="Float8 emulate + LoRA training test",
            test_name="float8_emulate_lora",
            ngpu=8,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_sft],
            test_descr="SFT ChatDataset integration and numerics test",
            test_name="sft",
            ngpu=2,
            golden_numerics_path=("tests/assets/losses/{execution_mode}/sft_a10g.txt"),
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_seed_checkpoint],
            test_descr="Seed checkpoint creation",
            test_name="seed_checkpoint",
            ngpu=1,
            timeout=30,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.kimi_k3_debugmodel_cp2],
            test_descr="Kimi K3, context parallel cp2",
            test_name="kimi_k3_cp2",
            ngpu=2,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
    ]
