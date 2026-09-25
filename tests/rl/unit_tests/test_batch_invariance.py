# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

import torchtitan.components.loss as loss_module


def test_batch_invariance_gathers_vocab_shards(monkeypatch):
    full_logits = torch.tensor([[2.0, -1.0, 0.5, 3.0], [-2.0, 0.25, 1.0, 0.0]])
    local_logits = full_logits[:, :2]
    labels = torch.tensor([3, 1])
    calls = []
    monkeypatch.setattr(loss_module, "is_in_batch_invariant_mode", lambda: True)
    tp_group = object()

    def gather(logits, *args, **kwargs):
        calls.append((logits, args, kwargs))
        return full_logits

    monkeypatch.setattr(loss_module.spmd, "redistribute", gather)
    logprobs, entropy = loss_module.compute_logprobs(
        local_logits,
        labels,
        vocab_parallel_group=tp_group,
        return_entropy=True,
        global_vocab_size=full_logits.shape[-1],
    )

    assert len(calls) == 1
    assert calls[0][0] is local_logits
    expected_logprobs = -F.cross_entropy(full_logits, labels, reduction="none")
    expected_entropy = torch.logsumexp(full_logits, dim=-1) - (
        torch.softmax(full_logits, dim=-1) * full_logits
    ).sum(dim=-1)
    torch.testing.assert_close(logprobs, expected_logprobs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_vocab_parallel_policy_stats_require_global_vocab_size(monkeypatch):
    monkeypatch.setattr(loss_module, "is_in_batch_invariant_mode", lambda: False)
    tp_group = object()

    with pytest.raises(
        ValueError,
        match="global_vocab_size is required for vocab-parallel policy statistics",
    ):
        loss_module.compute_logprobs(
            torch.randn(2, 4),
            torch.tensor([0, 1]),
            vocab_parallel_group=tp_group,
        )


def test_replicated_policy_stats_do_not_infer_layout_from_spmd_context(monkeypatch):
    monkeypatch.setattr(
        loss_module.spmd,
        "redistribute",
        lambda *args, **kwargs: pytest.fail("replicated logits must not be gathered"),
    )
    monkeypatch.setattr(loss_module, "spmd_mesh_size", lambda _axis: 2)

    logits = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    actual = loss_module.compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=None,
    )

    expected = -F.cross_entropy(logits, labels, reduction="none")
    torch.testing.assert_close(actual, expected)


def test_vllm_logprob_patch_keeps_trainer_fallback_path(monkeypatch):
    """The vLLM patch must retain the trainer's batch-invariant op sequence."""
    # Import by path so this CPU test does not load the RL package initializer,
    # which intentionally requires the optional vLLM runtime.
    module_path = (
        Path(__file__).resolve().parents[3] / "torchtitan/rl/model/batch_invariance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "torchtitan_batch_invariance_test", module_path
    )
    assert spec is not None and spec.loader is not None
    batch_invariance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_invariance)

    module_names = (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.sample",
        "vllm.v1.worker.gpu.sample.logprob",
    )
    modules = {name: ModuleType(name) for name in module_names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if name != module_names[-1]:
            module.__path__ = []
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(modules[parent_name], child_name, module)

    calls = []
    compute_logprobs = loss_module.compute_logprobs

    def recording_compute_logprobs(logits, labels, **kwargs):
        calls.append((logits, labels.clone(), kwargs))
        return compute_logprobs(logits, labels, **kwargs)

    monkeypatch.setattr(loss_module, "compute_logprobs", recording_compute_logprobs)
    batch_invariance.force_logprobs_fn_for_batch_invariance()

    logits = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.0, 0.0, 1.5, 0.5]],
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([[3, 0, 2], [2, 1, 3]], dtype=torch.int32)
    actual = modules[module_names[-1]].compute_token_logprobs(logits, token_ids)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, token_ids.to(torch.int64)
    )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == token_ids.shape[1]
    for column, (call_logits, call_labels, call_kwargs) in enumerate(calls):
        assert call_logits is logits
        torch.testing.assert_close(call_labels, token_ids[:, column].to(torch.int64))
        assert call_kwargs == {"vocab_parallel_group": None}
