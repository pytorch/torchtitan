# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.models.deepseek_v4_1.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        seq_len=32,
        attention_backend="reference",
        indexer_loss_coeff=0.01,
    )
    for layer in config.layers:
        layer.feed_forward.routed_experts.compute_dtype = "float32"
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    model.eval()
    yield model
    torch.set_num_threads(previous)


def test_packed_documents_match_separate_forwards(model):
    tokens = torch.tensor([9, 15, 23, 19, 6, 2, 8, 7, 5, 10, 3, 4, 1])
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5])
    metadata = model._get_attention_metadata(positions=positions)
    packed = model(tokens, positions=positions, attention_metadata=metadata)
    separate = torch.cat((model(tokens[:7]), model(tokens[7:])))
    torch.testing.assert_close(packed, separate, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("prefix_length,token_count", [(10, 17), (1, 1)])
def test_forward_is_causal_with_shared_decoder_kv(model, prefix_length, token_count):
    first = torch.arange(token_count)
    second = first.clone()
    second[prefix_length:] += 30
    # Future routing changes grouped GEMM row counts and their FP32 rounding.
    torch.testing.assert_close(
        model(first)[:prefix_length], model(second)[:prefix_length], rtol=0, atol=1e-6
    )
    embeddings = []

    def retain_embeddings(module, args, output):
        output.retain_grad()
        embeddings.append(output)

    hook = model.tok_embeddings.register_forward_hook(retain_embeddings)
    model(first)[:prefix_length].square().sum().backward()
    hook.remove()
    assert embeddings[0].grad is not None
    assert embeddings[0].grad[:prefix_length].norm() > 0
    assert torch.count_nonzero(embeddings[0].grad[prefix_length:]) == 0


def test_checkpoint_roundtrip_is_exact(model):
    adapter = DeepSeekV41StateDictAdapter(model.config)
    original = model.state_dict()
    hf = adapter.to_hf(original)
    restored = adapter.from_hf(hf)
    assert original.keys() == restored.keys()
    assert "layers.2.attn.compressor.wgate.weight" in hf
    assert "layers.3.attn.compressor.wkv.weight" not in hf
    assert "layers.10.attn.indexer.wk.weight" not in hf
    assert "layers.1.engram.embed.weight" in hf
    for key in original:
        torch.testing.assert_close(restored[key], original[key], rtol=0, atol=0)


def test_forward_rejects_images_without_a_vision_encoder(model):
    with pytest.raises(ValueError, match="vision-enabled model"):
        model(
            torch.arange(18),
            pixel_values=torch.randn(36, 12),
            grid_hw=((6, 6),),
            token_types=torch.tensor([-1, -1, 0, 1, 1, 2, 1, 1, 2, 3] + [-1] * 8),
        )


# T: packed tokens; V: vocabulary.
def test_training_remat_preserves_gradients_and_routing_state(model):
    model.train()
    rematerialized = copy.deepcopy(model)
    DefaultActivationRematConfig().apply(rematerialized)
    positions_T = torch.tensor([*range(7), *range(12)])
    padding_T = torch.zeros(19, dtype=torch.bool)
    padding_T[-2:] = True
    denominator = torch.tensor([34.0])
    metadata = model._get_attention_metadata(
        positions=positions_T, padding_mask=padding_T, max_context_length=32
    )
    outputs = [[], []]
    labels = []
    for offset in (0, 40):
        tokens_T = torch.arange(19) + offset
        labels.append((tokens_T + 1).masked_fill(padding_T, -100))
        kwargs = dict(
            positions=positions_T,
            padding_mask=padding_T,
            attention_metadata=metadata,
            aux_loss_denominators=denominator,
        )
        for output, candidate in zip(outputs, (model, rematerialized), strict=True):
            output.append(candidate(tokens_T, **kwargs))
    losses = []
    for logits in outputs:
        losses.append(
            sum(
                F.cross_entropy(value, targets, reduction="sum")
                for value, targets in zip(logits, labels, strict=True)
            )
            / denominator[0]
        )
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    actual_parameters = dict(rematerialized.named_parameters())
    for name, parameter in model.named_parameters():
        actual_grad = actual_parameters[name].grad
        assert (actual_grad is None) == (parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(actual_grad, parameter.grad, rtol=0, atol=0)
    actual_buffers = dict(rematerialized.named_buffers(remove_duplicate=False))
    for name, value in model.named_buffers(remove_duplicate=False):
        torch.testing.assert_close(actual_buffers[name], value, rtol=0, atol=0)
    assert model.layers["8"].attention.compressor.wkv.weight.grad.norm() > 0
    assert model.layers["10"].attention.indexer.wq_b.weight.grad.norm() > 0
