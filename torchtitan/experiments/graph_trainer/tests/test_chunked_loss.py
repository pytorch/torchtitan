# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

from torchtitan.components.loss import (
    BaseLoss,
    ChunkedLossWrapper,
    cross_entropy_loss,
    IGNORE_INDEX,
    LossTerm,
)
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)


class _FakeDecoder(nn.Module):
    """Minimal Decoder-like model for testing ChunkedLossWrapperWithParamGrads."""

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


class _WeightedTwoOutputLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        auxiliary_weight: float = 0.25

    def __init__(self, config: Config, *, compile_config=None):
        del compile_config
        self.fn = cross_entropy_loss
        self.auxiliary_weight = config.auxiliary_weight

    def _build_loss_terms(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        **loss_inputs,
    ) -> tuple[LossTerm, ...]:
        del loss_inputs
        assert isinstance(pred, tuple) and len(pred) == 2
        return (
            LossTerm(pred[0], labels),
            LossTerm(pred[1], labels, weight=self.auxiliary_weight),
        )


def _make_model_and_loss(dim, vocab_size, num_chunks=4, with_param_grads=False):
    model = _FakeDecoder(dim, vocab_size)
    loss_cls = (
        ChunkedLossWrapperWithParamGrads if with_param_grads else ChunkedLossWrapper
    )
    chunked_loss = loss_cls(loss_cls.Config(num_chunks=num_chunks))
    chunked_loss.lm_head = model.output
    return model, chunked_loss


def _chunked_loss_and_grads(model, chunked_loss, hidden_states, labels, gvt):
    h = hidden_states.detach().requires_grad_(True)
    loss, _ = chunked_loss(h, labels, gvt)
    if isinstance(chunked_loss, ChunkedLossWrapperWithParamGrads):
        h_grad, w_grad = torch.autograd.grad(loss, [h, model.output.weight])
    else:
        loss.backward()
        h_grad = h.grad
        w_grad = model.output.weight.grad
    return loss, h_grad.clone(), w_grad.clone()


class TestChunkedLossWrapperWithParamGrads(TestCase):
    def test_config_builds_param_grads_loss(self):
        loss = ChunkedLossWrapperWithParamGrads.Config(num_chunks=4).build()
        self.assertIsInstance(loss, ChunkedLossWrapperWithParamGrads)
        self.assertEqual(loss.num_chunks, 4)

    def test_bitwise_equal_with_chunked_loss(self):
        for num_tokens, num_chunks in ((16, 4), (8, 4)):
            with self.subTest(num_tokens=num_tokens, num_chunks=num_chunks):
                torch.manual_seed(42)
                D, V = 32, 64
                labels = torch.randint(0, V, (num_tokens,))
                global_valid_tokens = float((labels != IGNORE_INDEX).sum().item())
                hidden_states = torch.randn(num_tokens, D)

                model_a, loss_a_fn = _make_model_and_loss(D, V, num_chunks)
                model_b, loss_b_fn = _make_model_and_loss(
                    D, V, num_chunks, with_param_grads=True
                )
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
        num_tokens, D, V = 16, 32, 64
        model, chunked_loss = _make_model_and_loss(D, V, with_param_grads=True)
        h = torch.randn(num_tokens, D, requires_grad=True)
        labels = torch.randint(0, V, (num_tokens,))
        loss, _ = chunked_loss(h, labels)
        torch.autograd.grad(loss, [h, model.output.weight])
        self.assertIsNone(h.grad)  # pyrefly: ignore[missing-attribute]
        self.assertIsNone(
            model.output.weight.grad
        )  # pyrefly: ignore[missing-attribute]

    def test_multi_output_matches_base_wrapper(self):
        torch.manual_seed(42)
        num_tokens, dim, vocab_size, num_chunks = 16, 8, 32, 4
        labels = torch.randint(0, vocab_size, (num_tokens,))
        loss_config = _WeightedTwoOutputLoss.Config()

        model_a = _FakeDecoder(dim, vocab_size)
        model_b = _FakeDecoder(dim, vocab_size)
        model_b.output.load_state_dict(model_a.output.state_dict())
        loss_a = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(
                num_chunks=num_chunks,
                loss_fn=loss_config,
            )
        )
        loss_b = ChunkedLossWrapperWithParamGrads(
            ChunkedLossWrapperWithParamGrads.Config(
                num_chunks=num_chunks,
                loss_fn=loss_config,
            )
        )
        loss_a.set_lm_head(model_a.output)
        loss_b.set_lm_head(model_b.output)

        hidden = torch.randn(2, num_tokens, dim)
        hidden_a = tuple(item.detach().clone().requires_grad_(True) for item in hidden)
        hidden_b = tuple(item.detach().clone().requires_grad_(True) for item in hidden)

        value_a, _ = loss_a(hidden_a, labels)
        value_a.backward()
        value_b, _ = loss_b(hidden_b, labels)
        grads_b = torch.autograd.grad(value_b, (*hidden_b, model_b.output.weight))

        self.assertEqual(value_b, value_a)
        for hidden_grad_a, hidden_grad_b in zip(
            (item.grad for item in hidden_a), grads_b[:-1], strict=True
        ):
            self.assertEqual(hidden_grad_b, hidden_grad_a)
        self.assertEqual(grads_b[-1], model_a.output.weight.grad)


if __name__ == "__main__":
    unittest.main()
