# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

from torchtitan.components.loss import ChunkedLoss, IGNORE_INDEX
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWithParamGrads,
)


class _FakeDecoder(nn.Module):
    """Minimal Decoder-like model for testing ChunkedLossWithParamGrads."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.layers = nn.ModuleDict()
        self.tok_embeddings = None
        self.norm = None

    def forward(self, tokens, skip_lm_head=False):
        if skip_lm_head:
            return tokens
        return self.output(tokens)


def _make_model_and_loss(dim, vocab_size, num_chunks=4, with_param_grads=False):
    model = _FakeDecoder(dim, vocab_size)
    loss_cls = ChunkedLossWithParamGrads if with_param_grads else ChunkedLoss
    chunked_loss = loss_cls(loss_cls.Config(num_chunks=num_chunks))
    chunked_loss.lm_head = model.output
    return model, chunked_loss


def _chunked_loss_and_grads(model, chunked_loss, hidden_states, labels, gvt):
    h = hidden_states.detach().requires_grad_(True)
    loss = chunked_loss(h, labels, gvt)
    if isinstance(chunked_loss, ChunkedLossWithParamGrads):
        h_grad, w_grad = torch.autograd.grad(loss, [h, model.output.weight])
    else:
        loss.backward()
        h_grad = h.grad
        w_grad = model.output.weight.grad
    return loss, h_grad.clone(), w_grad.clone()


class TestChunkedLossWithParamGrads(TestCase):
    def test_bitwise_equal_with_chunked_loss(self):
        torch.manual_seed(42)
        B, L, D, V = 2, 8, 32, 64
        labels = torch.randint(0, V, (B, L))
        global_valid_tokens = float((labels != IGNORE_INDEX).sum().item())
        hidden_states = torch.randn(B, L, D)

        model_a, loss_a_fn = _make_model_and_loss(D, V)
        model_b, loss_b_fn = _make_model_and_loss(D, V, with_param_grads=True)
        model_b.output.load_state_dict(model_a.output.state_dict())

        loss_a, h_grad_a, w_grad_a = _chunked_loss_and_grads(
            model_a, loss_a_fn, hidden_states, labels, global_valid_tokens
        )
        loss_b, h_grad_b, w_grad_b = _chunked_loss_and_grads(
            model_b, loss_b_fn, hidden_states, labels, global_valid_tokens
        )

        self.assertEqual(loss_b, loss_a)
        self.assertEqual(h_grad_b, h_grad_a)
        self.assertEqual(w_grad_b, w_grad_a)

    def test_does_not_touch_dot_grad(self):
        torch.manual_seed(0)
        B, L, D, V = 2, 8, 32, 64
        model, chunked_loss = _make_model_and_loss(D, V, with_param_grads=True)
        h = torch.randn(B, L, D, requires_grad=True)
        labels = torch.randint(0, V, (B, L))
        loss = chunked_loss(h, labels)
        torch.autograd.grad(loss, [h, model.output.weight])
        self.assertIsNone(h.grad)  # pyrefly: ignore[missing-attribute]
        self.assertIsNone(
            model.output.weight.grad
        )  # pyrefly: ignore[missing-attribute]


if __name__ == "__main__":
    unittest.main()
