# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.rl.trainer.losses.dapo import _normalize, DAPOLoss


@pytest.fixture
def reset_local_compile():
    LocalCompileConfig().apply()
    torch.compiler.reset()
    yield
    LocalCompileConfig().apply()


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = _normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


def test_dapo_loss_matches_vllm_tempered_logprobs_bitwise() -> None:
    torch.manual_seed(0)
    logits = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 128, (64,))
    temperature = torch.full((64,), 0.7)
    # vLLM's sampler: logits.to(float32).div_(temperature), then log_softmax(dtype=float32).
    generator_logprobs = (
        logits.to(torch.float32)
        .div_(temperature[:, None])
        .log_softmax(dim=-1, dtype=torch.float32)
        .gather(-1, labels[:, None])
        .squeeze(-1)
    )

    _, metrics = DAPOLoss.Config().build()(
        logits=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def test_compiled_dapo_loss_matches_eager(reset_local_compile, monkeypatch) -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 128)
    labels = torch.randint(0, 128, (64,))
    generator_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None])
    # Ratios spread around 1, so both clip bounds change the loss.
    generator_logprobs = generator_logprobs.squeeze(-1) + 0.4 * torch.randn(64)
    generator_logprobs[0] = float("nan")  # dropped from the loss, as the trainer does
    loss_mask = torch.rand(64) > 0.2
    # The trainer's denominator: one int64 count per objective.
    global_loss_token_counts = torch.stack(
        [(loss_mask & generator_logprobs.isfinite()).sum()]
    )
    loss_inputs = {
        "generator_logprobs": generator_logprobs,
        "temperature": torch.linspace(0.7, 1.3, 64),
        "advantages": torch.randn(64),
        "loss_mask": loss_mask,
    }
    # Record what torch.compile receives, so the test fails if the region skips __call__.
    compiled_names = []
    real_compile = torch.compile

    def recording_compile(reference, **kwargs):
        compiled_names.append(reference.__qualname__)
        return real_compile(reference, **kwargs)

    monkeypatch.setattr(torch, "compile", recording_compile)
    loss_fn = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28).build()
    results = []
    for regions in ([], ["loss"]):
        LocalCompileConfig(regions=regions).apply()
        logits_with_grad = logits.clone().requires_grad_(True)
        loss, metrics = loss_fn(
            logits=logits_with_grad,
            labels=labels,
            global_loss_token_counts=global_loss_token_counts,
            **loss_inputs,
        )
        loss.backward()
        results.append((loss, metrics, logits_with_grad.grad))
    eager, compiled = results
    torch.testing.assert_close(compiled, eager)
    assert DAPOLoss.__call__.__qualname__ in compiled_names
