# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat

import torch


def zeropower_via_svd(G, **kwargs):
    original_dtype = G.dtype
    G = G.to(torch.float32)
    # SVD does not support bfloat16
    if G.size(0) > G.size(1):
        G = G.T
        transpose = True
    else:
        transpose = False
    U, S, V = G.svd()
    X = U @ V.T
    if transpose:
        X = X.T
    return X.to(original_dtype).contiguous()


# Polar Express
@torch.compile
def zeropower_via_polar_express(G, steps=5, eps=1e-7):
    # https://arxiv.org/abs/2505.16932
    coeffs_base = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.948690853482295, -2.908902115962949, 0.5518191394370137),
        (3.318419657370602, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.668903984574749, 0.4188073119525673),
        (1.891301407787398, -1.267995827194587, 0.3768040894852483),
        (1.875001480853448, -1.250001645399949, 0.3750001645474248),
        (1.875000000000000, -1.250000000000000, 0.375000000000000),  # limit
    ]

    # apply the 1/1.01 stabiliser **only** to the first seven triples
    coeffs_base = [
        (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_base[:-1]
    ] + [coeffs_base[-1]]

    # extend the list so that coeffs[k] is defined for every k < steps
    coeffs = coeffs_base + list(
        repeat(coeffs_base[-1], max(0, steps - len(coeffs_base)))
    )

    original_dtype = G.dtype
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (torch.linalg.norm(X) + eps)  # ensure top singular value <= 1

    # main loop
    for k in range(steps):
        a, b, c = coeffs[k]
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(original_dtype)


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert (
        len(G.shape) == 2
    ), f"Please make sure gradients are 2D tensors to use NS, got shape: {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    #     for a, b, c in [ # updated coefficients from @leloykun
    #     (4.0848, -6.8946, 2.9270),
    #     (3.9505, -6.3029, 2.6377),
    #     (3.7418, -5.5913, 2.3037),
    #     (2.8769, -3.1427, 1.2046),
    #     (2.8366, -3.0525, 1.2012),
    # ]:
    original_dtype = G.dtype
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (torch.linalg.norm(X) + eps)  # ensure top singular value <= 1

    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(original_dtype)


zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    polar_express=zeropower_via_polar_express,
    identity=lambda x, **kwargs: x,
)
