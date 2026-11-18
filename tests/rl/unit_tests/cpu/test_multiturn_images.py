# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A multi-turn image rollout keeps every image with its placeholder run, end to end.

Real Qwen3.5 image-reference renderer -> TokenEnv -> the rollout loop -> TrainingSamples.
Downloads the Qwen3.5 tokenizer from the Hugging Face Hub; set ``QWEN3_5_TOKENIZER`` to
use a local copy instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from torchtitan.components.data.multimodal.vision_processor import (
    PixelBudgetResize,
    VisionProcessor,
)
from torchtitan.components.renderer import hash_image_file
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.models.qwen3_5.renderer import (
    Qwen35ImageRefRenderer,
    Qwen35ImageRefRendererConfig,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout import RolloutStatus
from torchtitan.rl.rollout.message_env import MessageEnvInitOutput, MessageEnvStepOutput
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

VISION_PROCESSOR = dataclasses.replace(
    VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    resize=PixelBudgetResize.Config(min_pixels=32 * 32, max_pixels=128 * 128),
)
PROCESSOR = VISION_PROCESSOR.build()


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        os.environ.get("QWEN3_5_TOKENIZER", "Qwen/Qwen3.5-0.8B")
    )


@pytest.fixture
def image_paths(tmp_path) -> list[str]:
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (96, 64)), ("b.png", (64, 128))):
        path = tmp_path / name
        Image.fromarray(
            rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        ).save(path)
        paths.append(str(path))
    return paths


class _TwoImageEnv:
    """Turn 1 shows image a; the env's reply shows image b; turn 2 ends the rollout."""

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self._turn = 0

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": self._paths[0]},
                        {"type": "text", "text": "What organ is this?"},
                    ],
                }
            ]
        )

    async def step(self, completion_message) -> MessageEnvStepOutput:
        self._turn += 1
        if self._turn > 1:
            return MessageEnvStepOutput(env_messages=[], done=True)
        return MessageEnvStepOutput(
            env_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare with this scan."},
                        {"type": "image", "path": self._paths[1]},
                    ],
                }
            ],
            done=False,
        )

    async def close(self) -> None:
        pass


class _Config:
    def __init__(self, value) -> None:
        self._value = value

    def build(self, **kwargs):
        return self._value


def _assert_placeholders(token_ids, images, placeholders, image_pad_id) -> None:
    """Each image sits on a run of <|image_pad|> as long as its token count, and
    every <|image_pad|> belongs to exactly one image."""
    covered = []
    for image, placeholder in zip(images, placeholders, strict=True):
        width, height = Image.open(image.source_path).size
        assert placeholder.length == PROCESSOR.num_placeholder_tokens(
            PROCESSOR.grid_thw(height, width)
        )
        covered += range(placeholder.offset, placeholder.offset + placeholder.length)
    assert covered == np.flatnonzero(np.asarray(token_ids) == image_pad_id).tolist()


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_rollout_keeps_each_image_on_its_placeholders(
    image_paths, tokenizer, enable_thinking: bool
):
    """Without thinking, turn 2 bridges and the rollout is one sample [a, b]. With it,
    the renderer drops turn 1's reasoning from the history, so the rollout branches
    into samples [a] and [a, b]."""
    renderer = Qwen35ImageRefRenderer(
        tokenizer,
        Qwen35ImageRefRendererConfig(
            vision_processor=VISION_PROCESSOR,
            enable_thinking=enable_thinking,
        ),
    )
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    reasoning = "Let me look.\n</think>\n\n" if enable_thinking else ""
    completions = [
        tokenizer.encode(f"{reasoning}Liver.<|im_end|>", add_special_tokens=False),
        tokenizer.encode(f"{reasoning}Same organ.<|im_end|>", add_special_tokens=False),
    ]
    prompts = []

    async def generate_fn(prompt, *, request_id, **kwargs):
        token_ids = completions[len(prompts)]
        prompts.append(
            (list(prompt.token_ids), prompt.images, prompt.image_placeholders)
        )
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=token_ids,
            token_logprobs=[-0.1] * len(token_ids),
            finish_reason="stop",
        )

    async def release_session_fn(**kwargs) -> None:
        pass

    worker = MessageEnvRolloutWorker(
        SimpleNamespace(
            rubric=_Config(None),
            message_env=None,
            token_env=None,
            advantage=_Config(None),
        )
    )
    env = TokenEnv.Config().build(
        message_env=_TwoImageEnv(image_paths), renderer=renderer
    )
    rollout = asyncio.run(
        worker._run_single_rollout(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            env=env,
            sampling=SamplingConfig(),
            group_id=0,
            rollout_id=0,
        )
    )
    assert rollout.status == RolloutStatus.COMPLETED

    a, b = (hash_image_file(Path(path).read_bytes()) for path in image_paths)
    (_, first_images, _), (second_ids, second_images, _) = prompts
    # The generator gets a, then a where it was plus b.
    assert [image.hash_value for image in first_images] == [a]
    assert [image.hash_value for image in second_images] == [a, b]
    # The rollout stores each turn's new images only: a is in turn 2's shared prefix (with
    # thinking, the prefix ends at turn 1's dropped reasoning, after a).
    assert [
        [image.hash_value for image in turn.prompt_delta_images]
        for turn in rollout.turns
    ] == [[a], [b]]
    assert [image.source_path for image in second_images] == image_paths
    assert second_images[0] == first_images[0]
    # The second prompt is what a full render of the conversation renders.
    full = renderer.render(
        rollout.turns[-1].prompt_messages, add_generation_prompt=True
    )
    assert list(full.token_ids) == second_ids

    samples = (
        TrainingSampleBuilder.Config()
        .build()
        .rollout_to_training_samples(
            dataclasses.replace(rollout, reward=1.0, advantage=0.5)
        )
    )
    # The builder rebuilds each sample's full images from the deltas.
    assert [[image.hash_value for image in sample.images] for sample in samples] == (
        [[a], [a, b]] if enable_thinking else [[a, b]]
    )
    for token_ids, images, placeholders in [
        *prompts,
        *(
            (sample.token_ids, sample.images, sample.image_placeholders)
            for sample in samples
        ),
    ]:
        _assert_placeholders(token_ids, images, placeholders, image_pad_id)
