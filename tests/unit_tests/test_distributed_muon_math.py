# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from torchtitan.components.distributed_optimizers.muon import _compute_muon_update


class TestDistributedMuonMath(unittest.TestCase):
    def test_two_steps_match_torch_muon(self):
        optimizer_kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": True,
            "ns_coefficients": (3.4445, -4.7750, 2.0315),
            "eps": 1e-7,
            "ns_steps": 3,
            "adjust_lr_fn": "original",
        }

        for dtype, shape in itertools.product(
            (torch.bfloat16, torch.float32), ((3, 5), (5, 3))
        ):
            with self.subTest(dtype=dtype, shape=shape):
                generator = torch.Generator().manual_seed(4)
                initial = torch.randn(shape, generator=generator, dtype=dtype)
                gradients = [
                    torch.randn(shape, generator=generator, dtype=dtype)
                    for _ in range(2)
                ]

                reference_param = torch.nn.Parameter(initial.clone())
                reference = torch.optim.Muon(
                    [reference_param], **optimizer_kwargs
                )
                actual_param = initial.clone()
                actual_momentum = torch.zeros_like(actual_param)

                for gradient in gradients:
                    reference_param.grad = gradient.clone()
                    reference.step()

                    actual_momentum.lerp_(
                        gradient, 1 - optimizer_kwargs["momentum"]
                    )
                    prepared = torch.lerp(
                        gradient,
                        actual_momentum,
                        optimizer_kwargs["momentum"],
                    )
                    update, adjusted_lr = _compute_muon_update(
                        prepared,
                        out=torch.empty_like(prepared),
                        lr=optimizer_kwargs["lr"],
                        ns_coefficients=optimizer_kwargs["ns_coefficients"],
                        ns_steps=optimizer_kwargs["ns_steps"],
                        eps=optimizer_kwargs["eps"],
                        adjust_lr_fn=optimizer_kwargs["adjust_lr_fn"],
                    )
                    actual_param.mul_(
                        1
                        - optimizer_kwargs["lr"]
                        * optimizer_kwargs["weight_decay"]
                    )
                    actual_param.add_(update, alpha=-adjusted_lr)

                self.assertTrue(torch.equal(actual_param, reference_param))
                self.assertTrue(
                    torch.equal(
                        actual_momentum,
                        reference.state[reference_param]["momentum_buffer"],
                    )
                )

    def test_batched_update_matches_independent_matrices(self):
        kwargs = {
            "lr": 0.03,
            "ns_coefficients": (3.4445, -4.7750, 2.0315),
            "ns_steps": 3,
            "eps": 1e-7,
            "adjust_lr_fn": "match_rms_adamw",
        }

        for shape in ((4, 3, 5), (4, 5, 3)):
            with self.subTest(shape=shape):
                generator = torch.Generator().manual_seed(5)
                prepared = torch.randn(shape, generator=generator)
                batched, _ = _compute_muon_update(
                    prepared, out=torch.empty_like(prepared), **kwargs
                )
                independent = torch.stack(
                    [
                        _compute_muon_update(
                            matrix, out=torch.empty_like(matrix), **kwargs
                        )[0]
                        for matrix in prepared
                    ]
                )

                self.assertTrue(torch.equal(batched, independent))
