# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import io
import os

import datasets
import pytest
from PIL import Image

from torchtitan.components.data.multimodal.utils.image import ImageSpec
from torchtitan.rl.examples.dapo_math import score_math_response
from torchtitan.rl.examples.geometry3k import (
    Geometry3kDataset,
    Geometry3kEnv,
    Geometry3kSample,
)


def _png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_release(path, rows: list[dict]) -> str:
    """A local copy of the release layout: ``data/train-00000-of-00001.parquet``."""
    features = datasets.Features(
        {
            "images": datasets.Sequence(datasets.Image()),
            "problem": datasets.Value("string"),
            "answer": datasets.Value("string"),
        }
    )
    os.makedirs(path / "data")
    datasets.Dataset.from_list(rows, features=features).to_parquet(
        str(path / "data" / "train-00000-of-00001.parquet")
    )
    return str(path)


def test_dataset_writes_each_image_once_and_yields_its_path(tmp_path) -> None:
    release = _write_release(
        tmp_path / "release",
        [
            {
                "images": [{"bytes": _png((40, 30)), "path": None}],
                "problem": "<image>Find x.",
                "answer": "2 \\sqrt { 221 }",
            }
        ],
    )
    config = Geometry3kDataset.Config(path=release, image_dir=str(tmp_path / "images"))

    [sample] = Geometry3kDataset(config)._samples
    image_path = str(tmp_path / "images" / "train" / "00000_0.png")
    assert sample == Geometry3kSample(
        index=0,
        image_paths=(image_path,),
        problem="<image>Find x.",
        ground_truth="2 \\sqrt { 221 }",
    )
    with open(image_path, "rb") as f:
        assert f.read() == _png((40, 30))
    # A second build reuses the written file.
    modified = os.path.getmtime(image_path)
    Geometry3kDataset(config)
    assert os.path.getmtime(image_path) == modified


def test_dataset_rejects_a_marker_count_mismatch(tmp_path) -> None:
    release = _write_release(
        tmp_path / "release",
        [
            {
                "images": [{"bytes": _png((8, 8)), "path": None}],
                "problem": "Find x.",
                "answer": "3",
            }
        ],
    )

    with pytest.raises(ValueError, match="0 <image> markers but 1 images"):
        Geometry3kDataset(
            Geometry3kDataset.Config(path=release, image_dir=str(tmp_path / "images"))
        )


def test_env_puts_each_image_in_place_of_its_marker() -> None:
    sample = Geometry3kSample(
        index=0,
        image_paths=("a.png", "b.png"),
        problem="Compare <image> with <image>. Find x.",
        ground_truth="3",
    )

    init = asyncio.run(Geometry3kEnv(Geometry3kEnv.Config(), env_input=sample).init())

    [message] = init.init_prompt_messages
    assert message["content"] == [
        {"type": "text", "text": "Compare"},
        {"type": "image", "path": "a.png"},
        {"type": "text", "text": "with"},
        {"type": "image", "path": "b.png"},
        {
            "type": "text",
            "text": ". Find x.\nThink step by step, then put the final answer in \\boxed{}.",
        },
    ]


@pytest.mark.parametrize(
    ("response", "ground_truth", "score"),
    [
        (r"so x = \boxed{2\sqrt{221}}", r"2 \sqrt { 221 }", 1.0),
        (r"\boxed{12\pi}", r"12 \pi", 1.0),
        (r"\boxed{108.5}", "108.5", 1.0),
        (r"\boxed{109}", "108.5", 0.0),
        ("x = 3", "3", 0.0),
    ],
)
def test_math_verify_grades_geometry3k_answers(response, ground_truth, score) -> None:
    assert score_math_response(response, ground_truth) == score


def test_recipe_builds() -> None:
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k

    config = rl_grpo_qwen3_5_0_8b_geometry3k()

    assert config.generator.engine.image_spec == config.async_loop.batcher.image_spec


@pytest.mark.parametrize("batcher_image_spec", [None, ImageSpec(patch_size=14)])
def test_recipe_needs_one_image_spec_on_generator_and_batcher(
    batcher_image_spec,
) -> None:
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k

    config = rl_grpo_qwen3_5_0_8b_geometry3k()
    async_loop = config.async_loop
    batcher = dataclasses.replace(async_loop.batcher, image_spec=batcher_image_spec)

    with pytest.raises(ValueError, match="same ImageSpec, or both None"):
        dataclasses.replace(
            config, async_loop=dataclasses.replace(async_loop, batcher=batcher)
        )


def test_lora_recipe_applies_one_lora_to_the_language_model_of_both_roles() -> None:
    from torchtitan.models.common.lora import lora_projections
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k_lora

    config = rl_grpo_qwen3_5_0_8b_geometry3k_lora()

    projections = lora_projections(config.trainer.model)
    # 6 full-attention layers x 4, 18 GatedDeltaNet layers x 2, 24 MLPs x 2.
    assert len(projections) == 108
    assert not any(fqn.startswith("vision_encoder") for fqn, _, _ in projections)
    assert lora_projections(config.generator.model) == projections
    # The generators load the frozen base themselves; syncs carry only the adapters.
    assert config.generator.checkpointer.initial_load_in_hf


def test_recipe_rejects_a_generator_lora_that_differs_from_the_trainer() -> None:
    from torchtitan.config.transform import apply_transforms, LoRATransform
    from torchtitan_recipes.rl.geometry3k import (
        rl_grpo_qwen3_5_0_8b_geometry3k,
        rl_grpo_qwen3_5_0_8b_geometry3k_lora,
    )

    config = rl_grpo_qwen3_5_0_8b_geometry3k_lora()
    base_generator = rl_grpo_qwen3_5_0_8b_geometry3k().generator
    generator = apply_transforms(
        base_generator,
        [LoRATransform(rank=16, alpha=32.0, target_modules=["feed_forward.w2"])],
    )

    with pytest.raises(ValueError, match="same LoRATransform"):
        dataclasses.replace(config, generator=generator)


def test_recipe_rejects_a_generator_checkpoint_that_differs_from_the_trainer() -> None:
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k_lora

    config = rl_grpo_qwen3_5_0_8b_geometry3k_lora()
    generator = dataclasses.replace(
        config.generator,
        checkpointer=dataclasses.replace(
            config.generator.checkpointer, initial_load_path="/other/checkpoint"
        ),
    )

    with pytest.raises(ValueError, match="must load the trainer's initial checkpoint"):
        dataclasses.replace(config, generator=generator)
