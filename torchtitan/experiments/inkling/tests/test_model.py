# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.transform.lora import LoRATransform
from torchtitan.experiments.inkling import build_model_config


def test_revisited_attention_adapter_accumulates_both_visit_gradients():
    torch.manual_seed(42)
    cfg = LoRATransform(
        rank=4, alpha=4, target_modules=["layers.1.attn.wq_du"]
    ).transform(build_model_config("debugmodel"))
    model = cfg.build()
    model.init_states()
    projection = model.layers[1].attn.wq_du
    tokens = torch.tensor([12, 2, 45, 67, 8, 10])
    logits = model(tokens[:-1])
    F.cross_entropy(logits, tokens[1:]).backward()
    grad = projection.lora_b.weight.grad.clone()

    # Substitute independent, identically initialized adapters at each visit. The
    # tied projection's gradient must be the sum, not just the last pass's update.
    visits = [copy.deepcopy(projection) for _ in range(2)]
    for visit in visits:
        visit.zero_grad()
    untied = iter(visits)
    with projection.register_forward_hook(
        lambda module, args, out: next(untied)(*args)
    ):
        independent = model(tokens[:-1])
        F.cross_entropy(independent, tokens[1:]).backward()
    torch.testing.assert_close(independent, logits, rtol=0, atol=0)
    grads = [visit.lora_b.weight.grad for visit in visits]
    assert all(g is not None and g.count_nonzero() for g in grads)
    torch.testing.assert_close(grad, grads[0] + grads[1], rtol=0, atol=0)


def test_batch_and_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(42)
    model = build_model_config("debugmodel").build()
    model.init_states()
    # Cross both the short convolution and sliding-window boundary.
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [42, 31, 12, 8, 10, 9, 11]])
    with torch.no_grad():
        logits = model(tokens)
        changed = tokens.clone()
        changed[0, 5:] = 11
        changed[1] = 20
        actual = model(changed)
    torch.testing.assert_close(actual[0, :5], logits[0, :5], rtol=0, atol=0)
    assert not torch.equal(actual[0, 5:], logits[0, 5:])


def test_padding_cannot_split_a_training_document():
    model = build_model_config("debugmodel").build()
    tokens = torch.tensor([[1, 0, 3]])
    batch = {
        "input": tokens,
        "labels": torch.tensor([[3, -100, 2]]),
        "positions": torch.tensor([[0, 0, 2]]),
        "padding_mask": torch.tensor([[False, True, False]]),
    }
    with pytest.raises(ValueError, match="contiguous document"):
        model.preprocess_inputs(batch)
