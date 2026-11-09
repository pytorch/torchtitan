# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json

import numpy as np
import pytest

from torchtitan.rl.examples.slake import (
    RewardSlakeAnswer,
    SlakeDataset,
    SlakeEnv,
    SlakeSample,
)
from torchtitan.rl.examples.slake.rubric import is_correct, normalize_answer
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID


def _row(qid: int, **overrides) -> dict:
    row = {
        "qid": qid,
        "img_name": f"xmlab{qid}/source.jpg",
        "question": f"Question {qid}?",
        "answer": "Yes",
        "q_lang": "en",
        "answer_type": "CLOSED",
        "content_type": "Organ",
    }
    row.update(overrides)
    return row


@pytest.fixture
def data_dir(tmp_path):
    rows = [
        _row(0),
        _row(1, answer="Liver", content_type="Size"),
        _row(2, answer_type="OPEN", answer="CT"),
        _row(3, content_type="KG"),
        _row(4, q_lang="zh"),
    ]
    (tmp_path / "train.json").write_text(json.dumps(rows))
    return tmp_path


def test_dataset_keeps_english_image_questions(data_dir) -> None:
    dataset = SlakeDataset(SlakeDataset.Config(data_dir=str(data_dir)))
    # Closed and open questions; not the knowledge-graph or Chinese ones.
    assert len(dataset) == 3
    samples = [next(dataset) for _ in range(6)]
    # Two reshuffled epochs, each a permutation of the kept questions.
    assert {s.qid for s in samples[:3]} == {s.qid for s in samples[3:]} == {0, 1, 2}
    assert samples[0].image_path == str(
        data_dir / "imgs" / f"xmlab{samples[0].qid}/source.jpg"
    )


def test_dataset_state_dict_resumes_the_stream(data_dir) -> None:
    config = SlakeDataset.Config(data_dir=str(data_dir), seed=7)
    dataset = SlakeDataset(config)
    next(dataset)
    state = dataset.state_dict()
    expected = [next(dataset).qid for _ in range(5)]

    resumed = SlakeDataset(config)
    resumed.load_state_dict(state)
    assert [next(resumed).qid for _ in range(5)] == expected


def test_env_asks_one_question() -> None:
    sample = SlakeSample(
        qid=0,
        image_path="/data/imgs/a.jpg",
        question="Is the lung healthy?",
        answer="Yes",
        content_type="Abnormality",
    )
    env = SlakeEnv(SlakeEnv.Config(), env_input=sample)
    init = asyncio.run(env.init())
    (message,) = init.init_prompt_messages
    image_part, text_part = message["content"]
    assert image_part == {"type": "image", "path": "/data/imgs/a.jpg"}
    assert text_part["text"].startswith("Is the lung healthy?\n")
    step = asyncio.run(
        env.step({"role": "assistant", "content": "<answer>Yes</answer>"})
    )
    assert step.done


@pytest.mark.parametrize(
    ("response", "expected", "correct"),
    [
        ("<answer>Yes</answer>", "Yes", True),
        ("<answer> yes, it does. </answer>", "Yes", True),
        ("<answer>No</answer>", "Yes", False),
        ("<answer>The axial plane</answer>", "Transverse Plane", True),
        ("<answer>X ray</answer>", "X-Ray", True),
        ("<answer>left lung</answer>", "Lung", False),
        ("<answer>No</answer> wait <answer>Liver</answer>", "Liver", True),
        # The SFT-style short answer, with no answer block.
        ("Yes", "Yes", True),
        ("Axial.", "Transverse Plane", True),
        ("Left lung", "Lung", False),
    ],
)
def test_is_correct(response: str, expected: str, correct: bool) -> None:
    assert is_correct(response, expected) is correct


def test_normalize_answer() -> None:
    assert normalize_answer(" The Axial plane. ") == "transverse plane"
    assert normalize_answer("T2-weighted") == "t2"
    assert normalize_answer("Small Bowel") == "small bowel"


def test_reward_scores_the_last_turn() -> None:
    sample = SlakeSample(
        qid=0, image_path="a.jpg", question="q", answer="Liver", content_type="Size"
    )
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_prefix_len=0,
        prompt_delta_token_ids=np.array([1], dtype=np.int32),
        completion_token_ids=np.array([2], dtype=np.int32),
        completion_logprobs=np.array([0.0], dtype=np.float32),
        completion_message={"role": "assistant", "content": "<answer>liver</answer>"},
    )
    rollout = Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=[turn]
    )
    reward = RewardSlakeAnswer(RewardSlakeAnswer.Config())
    assert asyncio.run(reward(rollout, sample)) == 1.0


def test_muse_spark_lora_recipe_applies_lora_to_the_attention_of_every_layer() -> None:
    from torchtitan.models.common.lora import lora_projections
    from torchtitan_recipes.rl.slake import rl_muse_spark_816b_a42b_slake_lora_gb300

    config = rl_muse_spark_816b_a42b_slake_lora_gb300(
        hf_assets_path="/checkpoint/muse", slake_data_dir="/data/slake"
    )

    # wqkv, wo and output_gate in each of the 62 layers; nothing in the vision tower or
    # experts.
    assert len(lora_projections(config.trainer.model)) == 3 * 62
