# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What K3 actually quantizes -- one definition, shared by QAT and QLoRA.

The released ``quantization_config`` targets ``["Linear"]`` but carries an
ignore list that removes almost all of it::

    format:            mxfp4-pack-quantized
    weights:           num_bits 4, group_size 32, symmetric, scale uint8
    input_activations: null
    ignore:            self_attn, shared_experts, mlp.{gate,up,gate_up,down}_proj,
                       lm_head, vision_tower, mm_projector

and report sec 4.1.4 states the intent directly: "quantize the MoE expert
weights -- which dominate the model's parameter memory -- to MXFP4, with
activations computed in MXFP8, while all non-expert components (attention
projections, latent MoE projections, shared experts, and MoE routers) remain in
higher precision."

So the scope is the ROUTED EXPERTS ONLY. In our module tree those are the
``GroupedExperts`` 3-D parameters, not ``nn.Linear`` at all -- meaning the
name-based target lists that ``apply_mxfp4_qat`` and ``quantize_lora_bases``
grew before the release quantized precisely the set K3 keeps in high precision,
and skipped the only set it quantizes. :func:`is_quantizable` is the single
predicate both now consult.

The ``input_activations: null`` in the checkpoint config is not a contradiction
of MXFP8 activations: the checkpoint stores weights only, and activation
precision is a runtime property of the QAT/serving path.
"""

from __future__ import annotations

import re

import torch.nn as nn

# Verbatim from the released config's quantization_config.ignore. Kept as the
# official regexes rather than paraphrased substrings so a diff against a future
# checkpoint is mechanical.
OFFICIAL_IGNORE_PATTERNS: tuple[str, ...] = (
    r".*self_attn.*",
    r".*shared_experts.*",
    r".*mlp\.(gate|up|gate_up|down)_proj.*",
    r".*lm_head.*",
    r".*vision_tower.*",
    r".*mm_projector.*",
)

# Our module names differ from the HF checkpoint's in two places, so the
# official patterns alone would under-match. Both additions are non-expert
# components the report explicitly lists as staying in higher precision.
_EXTRA_IGNORE_PATTERNS: tuple[str, ...] = (
    # HF calls the dense/shared FFN "mlp"; ours is feed_forward.
    r".*feed_forward\.(gate|up|down)_proj.*",
    # latent MoE projections ("latent MoE projections" in report sec 4.1.4)
    r".*moe\.latent\..*",
    # MoE router ("MoE routers", ibid). Ours is router.gate.
    r".*router\.gate.*",
    # HF calls both attention types self_attn, so the official pattern above
    # stopped covering ours when MLA moved to "attention" and KDA to
    # "delta_attention". The trailing dot is what keeps this off
    # attention_res_proj, which is a graft parameter and not attention.
    r".*attention\..*",
)

_IGNORE_RE = re.compile(
    "|".join(f"(?:{p})" for p in OFFICIAL_IGNORE_PATTERNS + _EXTRA_IGNORE_PATTERNS)
)

MXFP4_GROUP_SIZE = 32
MXFP4_NUM_BITS = 4


def is_ignored(fqn: str) -> bool:
    """True when the official ignore list keeps ``fqn`` in higher precision."""
    return _IGNORE_RE.fullmatch(fqn) is not None


def is_quantizable(fqn: str, module: nn.Module) -> bool:
    """True when K3 quantizes this module's weights to MXFP4.

    Under the official scope this is only ever a routed-expert module. The
    check is deliberately positive rather than "not ignored": a new module name
    we have not classified should default to higher precision, since wrongly
    quantizing a component K3 keeps in bf16 is a silent quality regression
    while wrongly skipping one only costs memory.
    """
    from torchtitan.models.common.moe import GroupedExperts

    return isinstance(module, GroupedExperts) and not is_ignored(fqn)


def quantizable_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every module in ``model`` that K3's scope puts in MXFP4."""
    return [(fqn, m) for fqn, m in model.named_modules() if is_quantizable(fqn, m)]
