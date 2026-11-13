# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism with image inputs on the RL trainer path.

Packs image microbatches with the batcher (each distinct image once per microbatch, every
placeholder token pointing at its row of the image embeddings), then trains them through
`Trainer.forward_backward` with Qwen3.5-0.8B: once without CP (FSDP=2, each rank the same
microbatch) and once per CP layout. CP shards the tokens, their embedding rows and their
M-RoPE positions with one permutation, while every CP rank encodes all of the
microbatch's images. So the decoder's input embeddings and positions must equal the
reference's bit for bit, and the vision encoder's gradient, summed over the CP ranks, must
match the reference up to bf16 noise (a missing sum would halve it at CP=2).

A process builds one trainer (its device meshes are process-wide), so each layout trains in
its own two-GPU torchrun launch of this file, which saves what the tests compare.

    TORCHTITAN_QWEN3_5_0_8B_HF_PATH=<checkpoint> pytest tests/rl/unit_tests/gpu/test_cp_images.py -v
"""

import argparse
import asyncio
import io
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image
from renderers import PlaceholderRange
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import hash_image_file
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
from torchtitan.models.common.attention.gdn import InnerGDN
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import (
    build_model_config as build_qwen3_5_model_config,
    QWEN3_5_VISION_PROCESSOR_KWARGS,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import TrainingSample
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

pytestmark = pytest.mark.multi_gpu

HF_PATH = os.environ.get("TORCHTITAN_QWEN3_5_0_8B_HF_PATH")
SEQ_LEN = 1024
IMAGE_PAD, VISION_START, VISION_END, EOS = 248056, 248053, 248054, 248046
VISION_PROCESSOR = VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)

# name -> (FSDP degree, CP degree, CP inner attention, head-tail load balancing)
LAYOUTS = {
    "no_cp": (2, 1, None, False),
    "allgather_headtail": (1, 2, AllGatherKVVarlenInnerAttention, True),
    "ulysses": (1, 2, UlyssesVarlenInnerAttention, False),
}


def _image(height: int, width: int, seed: int) -> PromptImage:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height // 32, width // 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).resize((width, height), Image.BILINEAR).save(buffer, "PNG")
    data = buffer.getvalue()
    return PromptImage(
        hash_value=hash_image_file(data),
        source_path=f"synthetic_{seed}.png",
        grid_thw=VISION_PROCESSOR.build().grid_thw(height, width),
        data=data,
    )


def _sample(index: int, prompt: list, completion: list[int] | int) -> TrainingSample:
    """``prompt``: ints (that many text tokens) and images, in order."""
    rng = np.random.default_rng(index)
    processor = VISION_PROCESSOR.build()
    token_ids, images, placeholders = [], [], []
    for part in prompt:
        if isinstance(part, int):
            token_ids += rng.integers(1000, 50000, size=part).tolist()
            continue
        num_tokens = processor.num_placeholder_tokens(part.grid_thw)
        token_ids.append(VISION_START)
        placeholders.append(PlaceholderRange(offset=len(token_ids), length=num_tokens))
        token_ids += [IMAGE_PAD] * num_tokens + [VISION_END]
        images.append(part)
    prompt_len = len(token_ids)
    if isinstance(completion, int):
        completion = rng.integers(1000, 50000, size=completion).tolist()
    token_ids += completion
    loss_mask = np.arange(len(token_ids)) >= prompt_len
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
        token_ids=np.asarray(token_ids, dtype=np.int64),
        loss_mask=loss_mask,
        logprobs=np.where(loss_mask, -1.5, 0.0).astype(np.float32),
        # Positive: with ratio << 1 on random tokens, PPO clipping would zero the
        # gradient of negative-advantage tokens.
        advantage=np.where(loss_mask, 1.0 + 0.25 * index, 0.0).astype(np.float32),
        images=tuple(images),
        image_placeholders=tuple(placeholders),
    )


def _cases() -> dict[str, list[TrainingSample]]:
    """Token ranges are in the 1024-token microbatch. CP=2 shards at 512 contiguously
    (Ulysses), or into rank 0: [0, 256) + [768, 1024) and rank 1: [256, 768) (head-tail)."""
    small = _image(256, 256, seed=1)  # 64 placeholder tokens
    large = _image(768, 512, seed=2)  # 384, longer than a head-tail chunk
    rng = np.random.default_rng(3)
    # The policy sampled the placeholder ids: text, not an image.
    sampled_placeholders = [VISION_START] + [IMAGE_PAD] * 8 + [VISION_END]
    return {
        # One image, encoded once, in two samples: [11, 75) on rank 0 and [727, 791) on
        # rank 1 (Ulysses) or split between ranks 1 and 0 at 768 (head-tail).
        "shared_image_on_both_ranks": [
            _sample(0, [10, small, 20], completion=30),
            _sample(1, [600, small, 20], completion=40),
        ],
        # [301, 685) splits at 512 (Ulysses), a sampled placeholder run, a text sample.
        "long_image_and_sampled_placeholder": [
            _sample(
                2,
                [300, large, 20],
                completion=rng.integers(1000, 50000, size=10).tolist()
                + sampled_placeholders
                + rng.integers(1000, 50000, size=20).tolist(),
            ),
            _sample(3, [40], completion=100),
        ],
    }


def _build_trainer(layout: str) -> Trainer:
    fsdp, cp, cp_attention, headtail = LAYOUTS[layout]
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=fsdp,
        context_parallel_degree=cp,
        context_parallel_load_balancer=(
            HeadTailCPLoadBalancer.Config() if headtail else None
        ),
    )
    model, _ = build_rl_model_configs(
        build_qwen3_5_model_config,
        "0.8B_mm",
        trainer_parallelism=parallelism,
        generator_parallelism=InferenceParallelismConfig(tensor_parallel_degree=1),
        seq_len=SEQ_LEN,
        attn_backend="varlen",
    )
    # The trainer of `rl_grpo_qwen3_5_0_8b_geometry3k` at this layout, built here: importing
    # the recipe imports its math grader, which needs math_verify, and the GPU CI jobs do
    # not install it.
    config = Trainer.Config(
        model=model,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, min_lr_factor=1.0
            ),
        ),
        training=TrainingConfig(
            disable_accelerator_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=SEQ_LEN,
        ),
        parallelism=parallelism,
        checkpointer=CheckpointManager.Config(initial_load_in_hf=True),
        loss=ChunkedLossWrapper.Config(
            num_chunks=4,
            loss_fn=GRPOLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        ),
    )
    config.debug.seed = 42
    config.debug.deterministic = True
    if cp > 1:
        config = apply_transforms(
            config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        VarlenInnerAttention: cp_attention,
                        InnerGDN: ContextParallelInnerGDN,
                    }
                )
            ],
        )
    return Trainer(
        config,
        precision=PrecisionConfig(),
        max_num_documents=None,
        hf_assets_path=HF_PATH,
        output_dir=tempfile.mkdtemp(prefix=f"test_cp_images_{layout}_"),
    )


def _global_token_index(trainer: Trainer, layout: str) -> torch.Tensor:
    """The microbatch position of each of this rank's local tokens."""
    _, cp, _, headtail = LAYOUTS[layout]
    if cp == 1:
        return torch.arange(SEQ_LEN)
    context = trainer.engine.parallelism_context
    order = torch.arange(SEQ_LEN)
    if headtail:
        with context.activate_spmd():
            balancer = HeadTailCPLoadBalancer.Config().build(
                seq_len=SEQ_LEN, attention_metadata=None
            )
            order = balancer.generate_permutation()[0].cpu().long()
    cp_rank = context.get_optional_mesh("cp").get_local_rank()
    local = SEQ_LEN // cp
    return order[cp_rank * local : (cp_rank + 1) * local]


def _batcher() -> Batcher:
    return Batcher.Config(build_mrope_positions=True).build(
        num_tokens_per_microbatch_per_dp_rank=SEQ_LEN,
        max_context_length=SEQ_LEN,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=EOS,
        temperature=1.0,
        vision_processor=VISION_PROCESSOR,
    )


def _run(layout: str) -> dict[str, dict]:
    """Per case: the full-microbatch decoder input embeddings and positions (on rank 0),
    and the vision encoder's gradients."""
    trainer = _build_trainer(layout)
    batcher = _batcher()
    fsdp = LAYOUTS[layout][0]
    model = trainer.engine.model_parts[0]
    captured = {}

    def capture_decoder_input(module, args, kwargs):
        # A decoder layer takes (x, attention_metadata, positions, ...).
        captured["embeds"], captured["positions"] = args[0].detach(), args[2].detach()

    hook = next(iter(model.layers.values())).register_forward_pre_hook(
        capture_decoder_input, with_kwargs=True
    )
    index = _global_token_index(trainer, layout)
    results = {}
    for case, samples in _cases().items():
        microbatch = batcher._pack_training_samples(samples).preprocess_images()
        # Without CP, both FSDP ranks train the microbatch: the global counts double.
        asyncio.run(
            trainer.forward_backward(
                [[microbatch]],
                microbatch.loss_token_counts * fsdp,
                microbatch.routing_token_counts * fsdp,
            )
        )
        parts = [None] * dist.get_world_size()
        local = {key: tensor.float().cpu() for key, tensor in captured.items()}
        dist.all_gather_object(parts, (index, local))
        full = {}
        for key in ("embeds", "positions"):
            shape = parts[0][1][key].shape[1:]
            full[key] = torch.empty((SEQ_LEN, *shape))
            # Without CP every rank holds the whole microbatch; with CP, its shard.
            for part_index, part in parts[: dist.get_world_size() // fsdp]:
                full[key][part_index] = part[key]
        grads = {}
        for name, param in model.named_parameters():
            if "vision_encoder" in name and param.grad is not None:
                grad = param.grad
                grads[name] = (
                    (grad.full_tensor() if isinstance(grad, DTensor) else grad)
                    .float()
                    .cpu()
                )
        results[case] = {**full, "vision_grads": grads}
    hook.remove()
    return results


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, dict[str, dict]]:
    if HF_PATH is None:
        pytest.skip("set TORCHTITAN_QWEN3_5_0_8B_HF_PATH to a Qwen3.5-0.8B checkpoint")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    output_dir = tmp_path_factory.mktemp("cp_images")
    results = {}
    for layout in LAYOUTS:
        output = output_dir / f"{layout}.pt"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                f"--master-port={_free_port()}",
                __file__,
                f"--layout={layout}",
                f"--output={output}",
            ],
            check=True,
        )
        results[layout] = torch.load(output, weights_only=False)
    return results


@pytest.mark.parametrize("layout", ["allgather_headtail", "ulysses"])
def test_cp_gives_each_token_its_image_embedding_and_position(runs, layout) -> None:
    for case, reference in runs["no_cp"].items():
        result = runs[layout][case]
        assert torch.equal(result["embeds"], reference["embeds"]), case
        assert torch.equal(result["positions"], reference["positions"]), case


@pytest.mark.parametrize("layout", ["allgather_headtail", "ulysses"])
def test_cp_sums_the_vision_encoder_gradient_over_ranks(runs, layout) -> None:
    for case, reference in runs["no_cp"].items():
        grads = runs[layout][case]["vision_grads"]
        expected = torch.cat([g.flatten() for g in reference["vision_grads"].values()])
        actual = torch.cat(
            [grads[name].flatten() for name in reference["vision_grads"]]
        )
        # bf16 CP numerics move these by a few percent (cosine >= 0.99 measured); a
        # missing or averaged sum over the CP ranks halves the norm.
        norm_ratio = (actual.norm() / expected.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
        assert 0.85 < norm_ratio < 1.15, (case, norm_ratio)
        assert cosine > 0.98, (case, cosine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=list(LAYOUTS), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = _run(args.layout)
    if dist.get_rank() == 0:
        torch.save(results, args.output)
    dist.barrier()
