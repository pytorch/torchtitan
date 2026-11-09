# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lift passes of the Qwen3 235B-A22B fusion plan (plans/plan_iter2.txt).

G-01 .. G-91 are the Qwen3 30B-A3B regions (agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8), reused
unchanged: their matchers lift the expected 94 (or 188 / 8) sites in the 235B capture. The 30B
G-02, G-33 and G-42 do not match at FSDP 256 x EP 32; G-102, G-115 and G-129 are the 235B finder's
regions for the same ops. G-105, G-111, G-125 and G-133 are regions only the 235B layout has
(FSDP bucket packing for sharded experts, the LM head gradient, wqkv buckets).
Importing this package registers each region's lift pass (<folder>/lift_pass.py); the
directory of the skill's lift_regions.py must be on PYTHONPATH.
"""

import importlib

# Region id -> (folder, pass name).
PASSES = {
    "G-01": ("g01_wqkv", "lift_fusion_region_g01"),
    "G-102": ("g102_k_norm", "lift_fusion_region_g102"),
    "G-07": ("g07_loss", "lift_fusion_region_g07"),
    "G-06": ("g06_aux_loss", "lift_fusion_region_g06"),
    "G-17": ("g17_aux_loss", "lift_fusion_region_g17"),
    "G-19": ("g19_wo", "lift_fusion_region_g19"),
    "G-15": ("g15_wqkv", "lift_fusion_region_g15"),
    "G-115": ("g115_routed_experts", "lift_fusion_region_g115"),
    "G-129": ("g129_routed_experts", "lift_fusion_region_g129"),
    "G-41": ("g41_wqkv", "lift_fusion_region_g41"),
    "G-45": ("g45_ffn_norm", "lift_fusion_region_g45"),
    "G-24": ("g24_wo", "lift_fusion_region_g24"),
    "G-28": ("g28_wo", "lift_fusion_region_g28"),
    "G-47": ("g47_router", "lift_fusion_region_g47"),
    # Hand-built: G-09 (all 8 loss chunks, unsafe to lift as one instance) split per chunk.
    "G-90": ("g90_loss_fwd", "lift_fusion_region_g90"),
    # From the re-analysis of the iter4 profile (its G-09): router aux-loss forward.
    "G-91": ("g91_aux_loss", "lift_fusion_region_g91"),
    # Only in the 235B layout (FSDP-sharded experts, 256-way FSDP buckets).
    "G-105": ("g105_routed_experts", "lift_fusion_region_g105"),
    "G-111": ("g111_lm_head", "lift_fusion_region_g111"),
    "G-125": ("g125_wqkv_bkt", "lift_fusion_region_g125"),
    "G-133": ("g133_wqkv", "lift_fusion_region_g133"),
}

# The plan, in the order the passes run.
DEFAULT_SELECTION = [name for _, name in PASSES.values()]

for _folder, _ in PASSES.values():
    importlib.import_module(f"{__name__}.{_folder}.lift_pass")
