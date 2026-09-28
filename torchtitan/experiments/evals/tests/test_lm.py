# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TrainstationLM scores must match an unpacked, per-request reference."""

import copy
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager, ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.evals.checkpoint import (
    build_model,
    cast_parameters,
    eval_parallelism,
    load_weights,
)
from torchtitan.experiments.evals.lm import TrainstationLM
from torchtitan.models.llama3 import model_registry

TOKENIZER_PATH = "./tests/assets/tokenizer"
MAX_CONTEXT_LENGTH = 64

PAIRS = [
    ("The capital of France is", " Paris."),
    ("Question: what is two plus two?\nAnswer:", " four"),
    ("", "An empty context is conditioned on BOS."),
    # Longer than MAX_CONTEXT_LENGTH, so the context is left-truncated.
    ("lorem ipsum " * 20, " dolor sit amet"),
]
DOCUMENTS = [
    "A short document.",
    # Spans several rolling windows.
    "Rolling windows cover long documents. " * 10,
]


@pytest.fixture(scope="module", autouse=True)
def process_group():
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmp}/rdzv", rank=0, world_size=1
        )
        yield
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(), tokenizer_path=TOKENIZER_PATH
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = model_registry("debugmodel", seq_len=MAX_CONTEXT_LENGTH).model.build()
    model.init_weights()
    return model.eval()


PARALLEL_DIMS = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
)


def _make_lm(model, tokenizer, num_tokens_per_batch=4 * MAX_CONTEXT_LENGTH):
    return TrainstationLM(
        model,
        tokenizer,
        parallel_dims=PARALLEL_DIMS,
        parallelism=ParallelismConfig(),
        max_context_length=MAX_CONTEXT_LENGTH,
        num_tokens_per_batch=num_tokens_per_batch,
        device=torch.device("cpu"),
    )


@torch.no_grad()
def _reference_logprob(model, context: list[int], continuation: list[int]):
    """Score one sequence on its own: no packing, no padding."""
    tokens = (context + continuation)[-(MAX_CONTEXT_LENGTH + 1) :]
    length = len(tokens) - 1
    inputs, _, extra_kwargs = model.preprocess_inputs(
        {
            "input": torch.tensor(tokens[:-1]),
            "labels": torch.tensor(tokens[1:]),
            "positions": torch.arange(length),
            "padding_mask": torch.zeros(length, dtype=torch.bool),
        },
        parallel_dims=PARALLEL_DIMS,
        parallelism=ParallelismConfig(),
    )
    logits = model(inputs, **extra_kwargs)
    logprobs = torch.log_softmax(logits[-len(continuation) :].float(), dim=-1)
    targets = torch.tensor(continuation)
    total = logprobs.gather(-1, targets.unsqueeze(-1)).sum().item()
    return total, bool((logprobs.argmax(-1) == targets).all())


def _encode_pair(tokenizer, context: str, continuation: str):
    if not context:
        return [tokenizer.bos_id], tokenizer.encode(
            continuation, add_bos=False, add_eos=False
        )
    whole = tokenizer.encode(context + continuation, add_bos=True, add_eos=False)
    context_enc = tokenizer.encode(context, add_bos=True, add_eos=False)
    return context_enc, whole[len(context_enc) :]


def test_loglikelihood_matches_reference(model, tokenizer):
    lm = _make_lm(model, tokenizer)
    requests = [
        Instance("loglikelihood", {}, pair, idx) for idx, pair in enumerate(PAIRS)
    ]
    results = lm.loglikelihood(requests, disable_tqdm=True)

    for (context, continuation), (logprob, is_greedy) in zip(
        PAIRS, results, strict=True
    ):
        expected, expected_greedy = _reference_logprob(
            model, *_encode_pair(tokenizer, context, continuation)
        )
        assert logprob == pytest.approx(expected, abs=1e-3)
        assert is_greedy == expected_greedy


def test_loglikelihood_rolling_matches_reference(model, tokenizer):
    lm = _make_lm(model, tokenizer)
    requests = [
        Instance("loglikelihood_rolling", {}, (doc,), idx)
        for idx, doc in enumerate(DOCUMENTS)
    ]
    results = lm.loglikelihood_rolling(requests, disable_tqdm=True)

    for doc, logprob in zip(DOCUMENTS, results, strict=True):
        tokens = tokenizer.encode(doc, add_bos=False, add_eos=False)
        # Disjoint windows of MAX_CONTEXT_LENGTH targets, each conditioned on
        # the preceding tokens (BOS for the first window).
        expected = 0.0
        for start in range(0, len(tokens), MAX_CONTEXT_LENGTH):
            prefix = [tokenizer.bos_id] + tokens[:start]
            window = tokens[start : start + MAX_CONTEXT_LENGTH]
            expected += _reference_logprob(model, prefix, window)[0]
        assert logprob == pytest.approx(expected, abs=1e-3)


def test_packing_matches_one_request_per_buffer(model, tokenizer):
    requests = [
        Instance("loglikelihood", {}, pair, idx) for idx, pair in enumerate(PAIRS)
    ]
    packed = _make_lm(model, tokenizer).loglikelihood(requests, disable_tqdm=True)
    one_per_buffer = _make_lm(
        model, tokenizer, num_tokens_per_batch=MAX_CONTEXT_LENGTH
    ).loglikelihood(requests, disable_tqdm=True)
    for (a, _), (b, _) in zip(packed, one_per_buffer, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_pack_respects_limits(model, tokenizer):
    lm = _make_lm(model, tokenizer, num_tokens_per_batch=MAX_CONTEXT_LENGTH)
    lm.max_num_documents = 2
    sequences = [([0] * n, [0]) for n in (40, 30, 20, 20, 20, 64)]
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i][0]))
    batches = lm._pack(order, sequences)
    assert sorted(i for b in batches for i in b) == list(range(len(sequences)))
    for batch in batches:
        assert len(batch) <= 2
        assert sum(len(sequences[i][0]) for i in batch) <= MAX_CONTEXT_LENGTH


def test_cast_parameters_keeps_complex_rope_cache(model, tokenizer):
    bf16_model = copy.deepcopy(model)
    cast_parameters(bf16_model, torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in bf16_model.parameters())
    assert all(
        a.dtype == b.dtype
        for a, b in zip(model.buffers(), bf16_model.buffers(), strict=True)
    )

    requests = [
        Instance("loglikelihood", {}, pair, idx) for idx, pair in enumerate(PAIRS)
    ]
    fp32 = _make_lm(model, tokenizer).loglikelihood(requests, disable_tqdm=True)
    bf16 = _make_lm(bf16_model, tokenizer).loglikelihood(requests, disable_tqdm=True)
    for (a, _), (b, _) in zip(fp32, bf16, strict=True):
        assert b == pytest.approx(a, rel=0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_model applies FSDP, which needs CUDA"
)
def test_load_weights_round_trips_dcp(model, tmp_path):
    dcp.save(ModelWrapper(model).state_dict(), checkpoint_id=str(tmp_path))
    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    config.parallelism = eval_parallelism(
        config.parallelism, world_size=1, tensor_parallel_degree=1
    )
    torch.manual_seed(1)
    loaded = build_model(
        config,
        parallel_dims=ParallelDims.from_config(config.parallelism, 1),
        device=torch.device("cuda"),
        dtype="float32",
    )
    load_weights(loaded, str(tmp_path))
    expected = model.state_dict()
    for key, value in ModelWrapper(loaded).state_dict().items():
        torch.testing.assert_close(
            value.full_tensor().cpu(), expected[key], rtol=0, atol=0
        )
