# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Knowledge distillation loss combining KL divergence with cross-entropy."""

import torch
import torch.nn.functional as F

from torchtitan.components.loss import cross_entropy_loss, IGNORE_INDEX


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Compute knowledge distillation loss.

    Args:
        student_logits: [B, L, V] student model output logits.
        teacher_logits: [B, L, V] teacher model output logits (detached).
        labels: [B, L] target token ids.
        temperature: Softmax temperature for soft targets.
        alpha: Weight for distillation loss vs hard-label loss.
            1.0 = pure distillation, 0.0 = pure cross-entropy.

    Returns:
        Scalar loss (sum reduction, to be divided by global_valid_tokens).
    """
    # Mask for valid (non-padding) tokens
    valid_mask = labels != IGNORE_INDEX  # [B, L]
    flat_mask = valid_mask.flatten()  # [B*L]

    # Hard-label cross-entropy loss (sum reduction)
    ce = cross_entropy_loss(student_logits, labels)

    # Soft-target KL divergence loss
    flat_student = student_logits.flatten(0, 1).float()  # [B*L, V]
    flat_teacher = teacher_logits.flatten(0, 1).float()  # [B*L, V]

    # Only compute KL on valid tokens
    flat_student = flat_student[flat_mask]
    flat_teacher = flat_teacher[flat_mask]

    student_log_probs = F.log_softmax(flat_student / temperature, dim=-1)
    teacher_probs = F.softmax(flat_teacher / temperature, dim=-1)

    # KL(teacher || student) with sum reduction, scaled by T^2
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (
        temperature**2
    )

    return alpha * kl + (1 - alpha) * ce
