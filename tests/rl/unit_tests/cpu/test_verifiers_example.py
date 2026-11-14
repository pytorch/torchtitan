# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers math example."""

from __future__ import annotations

import asyncio
import os
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf
from renderers import Qwen3RendererConfig
from tokenizers import Tokenizer

from torchtitan.components.renderer import from_renderers, RenderersConfigAdapter

from torchtitan.config import ConfigLoader
from torchtitan.rl.examples.dapo_math import DapoMathSample
from torchtitan.rl.examples.verifiers.dapo_math import data
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.rollouter import VerifiersRollouter
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.types import Completion, RolloutStatus
from torchtitan_recipes.rl.verifiers_dapo_math import _verifiers_math_rollouter_config
from verifiers.v1.harnesses.null import NullHarnessConfig as VerifiersNullHarnessConfig


def test_verifiers_task_scores_math_response() -> None:
    math_task = data.VerifiersMathTask(
        data.VerifiersMathData(
            idx=0,
            prompt="problem",
            ground_truth=r"336^\circ",
        )
    )
    assert (
        asyncio.run(
            math_task.math_verify(SimpleNamespace(last_reply=r"Answer: \boxed{336}"))
        )
        == 1.0
    )
    assert (
        asyncio.run(
            math_task.math_verify(SimpleNamespace(last_reply=r"Answer: \boxed{335}"))
        )
        == 0.0
    )


def test_verifiers_task_dataset_is_resumable(monkeypatch) -> None:
    samples = [
        DapoMathSample(prompt="problem 1", ground_truth="34"),
        DapoMathSample(prompt="problem 2", ground_truth="113"),
        DapoMathSample(prompt="problem 3", ground_truth="7"),
    ]
    monkeypatch.setattr(data, "_load_math_dataset", lambda name: (iter(samples), 3))
    config = VerifiersTaskDataset.Config(
        verifiers_taskset=data.VerifiersMathTasksetConfig(
            id="torchtitan_verifiers_dapo_math",
            dataset="dapo_math",
        ),
        seed=7,
    )
    first = config.build()

    second = config.build()
    assert [next(first) for _ in range(3)] == [next(second) for _ in range(3)]

    checkpoint = first.state_dict()
    expected = [next(first) for _ in range(3)]
    resumed = config.build()
    resumed.load_state_dict(checkpoint)
    assert [next(resumed) for _ in range(3)] == expected


def test_verifiers_task_dataset_keeps_the_first_num_tasks(monkeypatch) -> None:
    samples = [
        DapoMathSample(prompt=f"problem {i}", ground_truth="1") for i in range(3)
    ]
    monkeypatch.setattr(data, "_load_math_dataset", lambda name: (iter(samples), 3))
    dataset = VerifiersTaskDataset.Config(
        verifiers_taskset=data.VerifiersMathTasksetConfig(
            id="torchtitan_verifiers_dapo_math",
            dataset="dapo_math",
        ),
        num_tasks=2,
        shuffle=False,
    ).build()

    task_indices = [next(dataset).verifiers_task_data["idx"] for _ in range(4)]
    assert task_indices == [0, 1, 0, 1]


def test_verifiers_task_dataset_rejects_a_config_its_id_does_not_load() -> None:
    """A config subclass that kept its parent's id would run the parent's Taskset."""

    class SubclassMathTasksetConfig(data.VerifiersMathTasksetConfig):
        pass

    with pytest.raises(ValueError, match="SubclassMathTasksetConfig"):
        VerifiersTaskDataset.Config(
            verifiers_taskset=SubclassMathTasksetConfig(
                id="torchtitan_verifiers_dapo_math", dataset="dapo_math"
            )
        )


def test_verifiers_environment_uses_no_sandbox() -> None:
    rollouter_config = _verifiers_math_rollouter_config(max_rollout_tokens=10240)
    config = rollouter_config.worker

    assert isinstance(config.environment, vf.SingleAgentEnvConfig)
    assert isinstance(config.environment.agent.runtime, vf.SubprocessConfig)
    assert isinstance(config.environment.agent.harness, VerifiersNullHarnessConfig)
    assert (
        config.environment.taskset
        == rollouter_config.training_dataloader.dataset.verifiers_taskset
    )


def test_a_group_runs_end_to_end_in_a_rollout_worker(monkeypatch) -> None:
    """One group through a real `VerifiersRolloutWorker`: the Verifiers env and agent, the
    renderer client, the generation server, the trace decode and the reward. Only the
    generator is fake: rollout 0 answers right, rollout 1 wrong.

    Fails when a Verifiers or renderers upgrade changes what the generation server receives
    or what a trace holds; the other tests use fake traces.
    """
    # The null harness installs its chat program's packages through uv, from uv's index:
    # PyPI unless UV_DEFAULT_INDEX or UV_INDEX_URL names another. CI and the release check
    # install Verifiers just before, so this skips only off-network.
    index = (
        os.environ.get("UV_DEFAULT_INDEX")
        or os.environ.get("UV_INDEX_URL")
        or "https://pypi.org/simple"
    )
    try:
        urllib.request.urlopen(f"{index.rstrip('/')}/openai/", timeout=10).close()
    except OSError as error:
        pytest.skip(
            f"uv's package index is unreachable ({error}); the null harness needs it"
        )
    tokenizer = Tokenizer.from_file("tests/assets/tokenizer/tokenizer.json")
    end_of_turn = tokenizer.token_to_id("<|im_end|>")
    answers = [r"The answer is \boxed{2}.", r"I think \boxed{3}."]

    async def generate_fn(prompt, *, request_id, group_id, **kwargs):
        # run_group gives rollout i the seed 100 + i.
        answer = answers[kwargs["sampling_config"].seed - 100]
        token_ids = tokenizer.encode(answer, add_special_tokens=False).ids
        token_ids.append(end_of_turn)
        return Completion(
            min_policy_version=3,
            max_policy_version=4,
            request_id=request_id,
            token_ids=token_ids,
            token_logprobs=[-0.5] * len(token_ids),
            finish_reason="stop",
            metrics=[],
        )

    # setup_async edits these for the whole process; set here, monkeypatch restores them.
    for name, value in (("VF_RUN_ID", "test"), ("no_proxy", ""), ("NO_PROXY", "")):
        monkeypatch.setenv(name, value)
    worker = _verifiers_math_rollouter_config(max_rollout_tokens=2048).worker.build()

    async def run():
        # Monarch runs setup_async and close in different tasks: an endpoint, then __cleanup__.
        await asyncio.create_task(
            worker.setup_async(
                tokenizer_config=None,
                renderer_config=from_renderers(
                    Qwen3RendererConfig(enable_thinking=False)
                ),
                hf_assets_path="tests/assets/tokenizer",
            )
        )
        try:
            return await worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=AsyncMock(),
                sample=VerifiersTaskSample(
                    verifiers_task_data={
                        "idx": 0,
                        "prompt": "What is 1 + 1?",
                        "ground_truth": "2",
                    }
                ),
                group_id=5,
                group_size=2,
                sampling=SamplingConfig(
                    temperature=1.0, top_p=1.0, max_tokens=64, seed=100
                ),
            )
        finally:
            await asyncio.create_task(worker.close())

    group = asyncio.run(run())
    right, wrong = sorted(group.rollouts, key=lambda rollout: rollout.rollout_id)
    assert [right.status, wrong.status] == [RolloutStatus.COMPLETED] * 2
    assert [right.reward, wrong.reward] == [1.0, 0.0]
    [turn] = right.turns
    assert turn.completion_token_ids.tolist() == tokenizer.encode(
        answers[0], add_special_tokens=False
    ).ids + [end_of_turn]
    assert turn.completion_logprobs.tolist() == [-0.5] * len(turn.completion_token_ids)
    assert (turn.min_policy_version, turn.max_policy_version) == (3, 4)
    assert turn.completion_message["content"].endswith(r"\boxed{2}.")


def test_verifiers_config_keeps_dapo_training_recipe() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.rl.verifiers_dapo_math",
            "--config",
            "rl_dapo_qwen3_4b_verifiers_8k",
        ]
    )
    assert isinstance(config.rollouter, VerifiersRollouter.Config)
    assert config.generator.sampling.max_tokens == 8192
    assert config.rollouter.worker.generation_server.max_rollout_tokens == 10240
    assert config.dump_folder == "outputs/rl/qwen3_4b_verifiers_8k"
    assert isinstance(config.renderer, RenderersConfigAdapter)
    assert config.renderer.renderers_config.name == "qwen3"
    assert config.renderer.renderers_config.enable_thinking
