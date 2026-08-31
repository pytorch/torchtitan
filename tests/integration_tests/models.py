# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchtitan_recipes.tests.models as recipes

from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build the list of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will
    broadly cover all the supported model parallelism patterns on all the supported
    models.
    """
    return [
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_cp2],
            test_descr="Llama 3 FSDP+TP+CP",
            test_name="llama3_fsdp+tp+cp",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/llama3_a10g.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_pp2],
            test_descr="Llama 3 FSDP+TP+PP",
            test_name="llama3_fsdp+tp+pp",
            ngpu=8,
            golden_numerics_path="tests/assets/losses/real_pg/llama3_pp_a10g.txt",
            use_real_pg=True,
        ),
        # Integration Test Cases for DeepSeek V3
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_mtp_fsdp4_ep2_compile],
            test_descr="DeepSeek V3 MTP FSDP+EP+compile",
            test_name="deepseek_v3_mtp_fsdp+ep+compile",
            ngpu=4,
            # The Helion fused RoPE kernels are CUDA-only and tuned for NVIDIA
            # H100/GB200; skip on ROCm where they are unvalidated.
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fsdp8_ep8],
            test_descr="DeepSeek V3 FSDP+EP",
            test_name="deepseek_v3_fsdp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/deepseek_v3_a10g.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4],
            test_descr="DeepSeek V3 FSDP+CP+PP+EP",
            test_name="deepseek_v3_fsdp+cp+pp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/real_pg/deepseek_v3_cp_pp_a10g.txt"
            ),
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_hsdp2x2_ep2],
            test_descr="DeepSeek V3 HSDP+EP",
            test_name="deepseek_v3_hsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2],
            test_descr="DeepSeek V3 fused MLA+SwiGLU FSDP+EP",
            test_name="deepseek_v3_fused_mla_swiglu_fsdp+ep",
            ngpu=4,
            skip_rocm_test=True,
        ),
        # Integration Test Cases for Qwen3 dense and MoE model
        OverrideDefinitions(
            configs=[recipes.qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8],
            test_descr="Qwen3 MoE FSDP+TP+CP+EP (param groups)",
            test_name="qwen3_moe_fsdp+tp+cp+ep_param_groups",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/qwen3_a10g.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[
                recipes.qwen3_debugmodel_fsdp2_tp2_cp2_no_sp,
                recipes.qwen3_debugmodel_fsdp2_tp2_cp2,
            ],
            test_descr="Qwen3 FSDP+TP+CP (SP disabled)",
            test_name="qwen3_fsdp+tp+cp_no_sp",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.qwen3_debugmodel_fsdp2_tp2_cp2_compile_helion_rope],
            test_descr="Qwen3 fused QKV FSDP+TP+CP + compile + Helion RoPE override",
            test_name="qwen3_fused_qkv_fsdp+tp+cp_compile_helion_rope",
            ngpu=8,
            # The Helion fused cos/sin RoPE kernel is CUDA-only and its autotuned
            # configs are tuned for NVIDIA H100; skip on ROCm where it is
            # unvalidated (see torchtitan/overrides/helion_rope.py).
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.qwen3_debugmodel_non_fused_qkv_fsdp2_tp2_cp2],
            # Reverse test: fused QKV is the debugmodel default, so exercise the
            # separate wq/wk/wv projection path under FSDP+TP+CP.
            test_descr="Qwen3 non-fused QKV FSDP+TP+CP",
            test_name="qwen3_non_fused_qkv_fsdp+tp+cp",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3.5
        OverrideDefinitions(
            configs=[recipes.qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4],
            test_descr="Qwen3.5 MoE FSDP+TP+EP+PP",
            test_name="qwen3_5_moe_fsdp+tp+ep+pp",
            ngpu=8,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.qwen35_debugmodel_moe_fsdp4_tp2_ep4],
            test_descr="Qwen3.5 MoE FSDP+TP+EP",
            test_name="qwen3_5_moe_fsdp+tp+ep",
            ngpu=8,
            # NOTE: This topology is not bitwise deterministic with Real PG on
            # A10G, so this case provides end-to-end coverage without a golden.
        ),
        OverrideDefinitions(
            configs=[recipes.qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac],
            test_descr="Qwen3.5 FSDP+TP+VARLEN_ATTN + per op SAC",
            test_name="qwen3_5_fsdp+tp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
            use_real_pg=True,
        ),
        # Integration Test Cases for gpt-oss
        OverrideDefinitions(
            configs=[recipes.gpt_oss_debugmodel_fsdp4_tp2_ep4_compile],
            test_descr="Gpt-oss FSDP+TP+EP+compile",
            test_name="gpt_oss_fsdp+tp+ep+compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.gpt_oss_debugmodel_fsdp4_tp2_ep4],
            test_descr="GPT-OSS FSDP+TP+EP",
            test_name="gpt_oss_fsdp+tp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/gpt_oss_a10g.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[recipes.gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac],
            test_descr="GPT-OSS PP+FSDP+CP+EP+SACOP",
            test_name="gpt_oss_pp+fsdp+cp+ep+sacop",
            ngpu=8,
            golden_numerics_path="tests/assets/losses/real_pg/gpt_oss_pp_a10g.txt",
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.gpt_oss_debugmodel_fsdp4_pp2_ep4_sac],
            test_descr="Gpt-oss PP+FSDP+EP+SACOP with VarlenAttention",
            test_name="gpt_oss_pp+fsdp+ep+sacop",
            ngpu=8,
            use_real_pg=True,
        ),
        # Integration Test Cases for Kimi K2.7
        OverrideDefinitions(
            configs=[recipes.kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2],
            test_descr="Kimi K2.7 DistMuon PP+FSDP+EP",
            test_name="kimi_k2_5_muon_pp+fsdp+ep",
            ngpu=4,
            timeout=600,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.kimi_k2_5_debugmodel_muon_fsdp8_ep8],
            test_descr="Kimi K2.5 DistMuon FSDP+EP",
            test_name="kimi_k2_5_muon_fsdp+ep",
            ngpu=8,
        ),
        # Integration Test Cases for Muse Glimmer
        OverrideDefinitions(
            configs=[recipes.muse_glimmer_debugmodel_fsdp8],
            test_descr="Muse Glimmer text FSDP",
            test_name="muse_glimmer_text_fsdp",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/muse_glimmer_a10g.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[recipes.muse_glimmer_debugmodel_mm_fsdp2_tp2],
            test_descr="Muse Glimmer multimodal FSDP+TP+SP",
            test_name="muse_glimmer_mm_fsdp+tp+sp",
            ngpu=4,
        ),
    ]
