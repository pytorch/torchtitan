# This code is from:
# https://github.com/microsoft/dion/blob/main/optimizers/scalar_utils.py


# @article{ahn2025dion,
#  title={Dion: Distributed Orthonormalized Updates},
#  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
#  journal={arXiv preprint: 2504.05295},
#  year={2025}
# }

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List

import torch
from torch import Tensor


#@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


#@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


#@torch.compile(fullgraph=True)
def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdamW optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    M_div = torch._foreach_div(M, denom)
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


#@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)
