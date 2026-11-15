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


def test_recipe_vision_processor_matches_vision_encoder() -> None:
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k

    config = rl_grpo_qwen3_5_0_8b_geometry3k()
    processor = config.vision_processor

    assert processor is not None and processor.patch_order == "block"
    for model in (config.generator.model, config.trainer.model):
        encoder = model.vision_encoder
        assert processor.patch_size == encoder.patch_size
        assert processor.temporal_patch_size == encoder.temporal_patch_size
        assert processor.spatial_merge_size == encoder.spatial_merge_size


def test_recipe_gives_image_tokens_the_positions_its_model_takes() -> None:
    """The batcher gives the trainer M-RoPE positions exactly when the model takes them, as
    the generator does; Controller.Config rejects a mismatch."""
    from torchtitan_recipes.rl.geometry3k import rl_grpo_qwen3_5_0_8b_geometry3k

    config = rl_grpo_qwen3_5_0_8b_geometry3k()
    batcher = config.async_loop.batcher

    assert batcher.build_mrope_positions
    with pytest.raises(ValueError, match="build_mrope_positions=False.*M-RoPE"):
        dataclasses.replace(
            config,
            async_loop=dataclasses.replace(
                config.async_loop,
                batcher=dataclasses.replace(batcher, build_mrope_positions=False),
            ),
        )
