# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.losses.dapo import _normalize, DAPOLoss


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
        pred=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def _rl_loss_inputs(num_tokens: int, vocab: int, loss_token_frac: float) -> tuple:
    loss_mask = torch.rand(num_tokens) < loss_token_frac
    generator_logprobs = -5 * torch.rand(num_tokens)
    # -inf logprobs: the loss drops these even inside loss_mask.
    generator_logprobs[:8] = float("-inf")
    loss_kwargs = {
        "generator_logprobs": generator_logprobs,
        "temperature": torch.full((num_tokens,), 0.7),
        "advantages": torch.randn(num_tokens) * loss_mask,
        "loss_mask": loss_mask,
    }
    labels = torch.randint(0, vocab, (num_tokens,))
    return labels, loss_mask.sum().clamp_min(1).float(), loss_kwargs


@pytest.mark.parametrize("loss_token_frac", [0.15, 0.004, 0.0, 1.0])
@pytest.mark.parametrize(
    "loss_config",
    [DAPOLoss.Config(ratio_clip_high=0.28), GRPOLoss.Config()],
    ids=["dapo", "grpo"],
)
def test_chunked_loss_runs_lm_head_on_loss_mask_tokens(
    loss_config, loss_token_frac
) -> None:
    # Same loss, metrics and gradients as the unchunked loss. 0.15 runs 3 chunks, 0.004
    # keeps fewer tokens than chunks, 0.0 runs one masked row, 1.0 runs every token.
    torch.manual_seed(42)
    num_tokens, dim, vocab, num_chunks = 1024, 16, 64, 16
    labels, global_loss_token_counts, loss_kwargs = _rl_loss_inputs(
        num_tokens, vocab, loss_token_frac
    )
    hidden = torch.randn(num_tokens, dim)
    lm_head = nn.Linear(dim, vocab, bias=False)
    chunked_loss = ChunkedLossWrapper.Config(
        num_chunks=num_chunks, loss_fn=loss_config
    ).build()
    chunked_loss.set_lm_head(lm_head)

    hidden_ref = hidden.clone().requires_grad_()
    loss_ref, metrics_ref = chunked_loss.loss_fn(
        lm_head(hidden_ref), labels, global_loss_token_counts, **loss_kwargs
    )
    loss_ref.backward()
    grad_weight_ref, lm_head.weight.grad = lm_head.weight.grad, None

    lm_head_rows = []
    lm_head.register_forward_hook(
        lambda module, args, output: lm_head_rows.append(args[0].shape[0])
    )
    hidden_input = hidden.clone().requires_grad_()
    loss, metrics = chunked_loss(
        hidden_input, labels, global_loss_token_counts, **loss_kwargs
    )
    loss.backward()

    num_kept = int(loss_kwargs["loss_mask"].sum())
    assert sum(lm_head_rows) == max(num_kept, 1)
    assert max(lm_head_rows) <= num_tokens // num_chunks
    assert (hidden_input.grad[~loss_kwargs["loss_mask"]] == 0).all()
    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(metrics, metrics_ref)
    torch.testing.assert_close(hidden_input.grad, hidden_ref.grad)
    torch.testing.assert_close(lm_head.weight.grad, grad_weight_ref)


class TestChunkedLossFSDPLossMask(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_ranks_with_different_chunk_counts(self):
        """Rank 0 has no loss tokens (1 chunk), rank 1 runs 5: no hang, unchunked numbers."""
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("dp",))
        num_tokens, dim, vocab = 4096, 32, 128
        torch.manual_seed(42)
        reference_lm_head = nn.Linear(dim, vocab, bias=False)
        torch.manual_seed(self.rank)
        hidden = torch.randn(num_tokens, dim)
        labels, global_loss_token_counts, loss_kwargs = _rl_loss_inputs(
            num_tokens, vocab, (0.0, 0.6)[self.rank]
        )
        chunked_loss = ChunkedLossWrapper.Config(
            num_chunks=8, loss_fn=DAPOLoss.Config()
        ).build()

        results = []
        lm_head_rows = []
        for chunked in (False, True):
            lm_head = fully_shard(copy.deepcopy(reference_lm_head), mesh=mesh)
            hidden_input = hidden.clone().requires_grad_()
            if chunked:
                lm_head.register_forward_hook(
                    lambda module, args, output: lm_head_rows.append(args[0].shape[0])
                )
                chunked_loss.set_lm_head(lm_head)
                loss, _ = chunked_loss(
                    hidden_input, labels, global_loss_token_counts, **loss_kwargs
                )
            else:
                loss, _ = chunked_loss.loss_fn(
                    lm_head(hidden_input),
                    labels,
                    global_loss_token_counts,
                    **loss_kwargs,
                )
            loss.backward()
            results.append((loss, hidden_input.grad, lm_head.weight.grad.full_tensor()))

        self.assertEqual(len(lm_head_rows), (1, 5)[self.rank])
        for chunked_value, value in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(chunked_value, value)
