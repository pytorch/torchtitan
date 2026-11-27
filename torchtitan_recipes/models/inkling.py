# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fine-tune Inkling 2 Mini's internal attention projections on one DGX Spark."""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import DATASETS, TextProcessor
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import (
    CallableFunction,
    DebugConfig,
    PrecisionConfig,
    TrainingConfig,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.experiments.transformers_modeling_backend.tokenizer import (
    HFBackendTokenizer,
)
from torchtitan.experiments.inkling import build_model_config
from torchtitan.experiments.inkling.state_dict_adapter import export_merged_checkpoint
from torchtitan.observability.logging import init_logger
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)


def inkling_2_mini_lora(
    *,
    checkpoint: Path = Path("assets/hf/Inkling-2-Mini"),
    text: Path | None = None,
    seq_len: int = 16,
    steps: int = 10,
) -> Trainer.Config:
    """Build a standard Trainer config. Use the CLI for source-checked resume and merge."""
    if seq_len < 1 or steps < 1:
        raise ValueError("seq_len and steps must be positive")
    model = build_model_config("2-mini")
    if seq_len > model.max_context_length:
        raise ValueError(f"seq_len exceeds the model limit {model.max_context_length}")
    model.max_context_length = seq_len
    dataset = (
        DATASETS["c4_test"]
        if text is None
        else SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="text",
                split="train",
                load_dataset_kwargs={"data_files": str(text)},
            ),
            processor=TextProcessor.Config(),
        )
    )
    lora = LoRATransform(rank=8, alpha=8.0, target_modules=["attn.wq_du"])
    hf_assets_path = str(checkpoint.resolve())
    config = Trainer.Config(
        model=model,
        hf_assets_path=hf_assets_path,
        # The published template uses Hugging Face's generation tags.
        tokenizer=HFBackendTokenizer.Config(),
        loss=CrossEntropyLoss.Config(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1, enable_sequence_parallel=False
        ),
        precision=PrecisionConfig(dtype="bfloat16", param_dtype="bfloat16"),
        training=TrainingConfig(
            steps=steps,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            disable_accelerator_graphs=True,
        ),
        activation_remat=None,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            # Short conv and loop messages share context within one row.
            # Never put a second document after the first one in that context.
            max_num_documents=1,
            shuffle=False,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r"\.lora_[ab]\.weight$", lr=1e-3)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, min_lr_factor=1.0
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpointer=CheckpointManager.Config(
            initial_load_path=hf_assets_path,
            initial_load_in_hf=True,
            model_param_checkpoint_type_map=CallableFunction.Config(
                fn=lora.param_checkpoint_type
            ),
            last_save_model_only=False,
        ),
        debug=DebugConfig(seed=42),
    )
    return apply_transforms(config, [lora])


def verify_checkpoint_source(source: Path, run: Path) -> None:
    """Bind a run to the contents used for HF initialization, tokenization and export."""
    if run.resolve().is_relative_to(source.resolve()):
        raise ValueError("The training output must be outside the source checkpoint")
    manifest_path = run / "source-manifest.json"
    if not manifest_path.exists() and (run / "checkpoint").exists():
        raise ValueError(f"Cannot resume: missing source manifest in {run}")
    contents = {}
    for file in source.rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                contents[str(file.relative_to(source))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    if not contents:
        raise ValueError(f"Empty or missing source checkpoint: {source}")
    if manifest_path.exists():
        if json.loads(manifest_path.read_bytes()) != contents:
            raise ValueError(f"Run {run} is bound to different checkpoint contents")
    else:
        run.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x") as stream:
            json.dump(contents, stream, indent=2, sort_keys=True)
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text", type=Path, required=True, help="UTF-8, one document per line"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    destination = args.output_dir / f"merged-step-{args.steps}"
    if destination.exists():
        parser.error(f"merged checkpoint already exists: {destination}")
    init_logger()
    config = inkling_2_mini_lora(
        checkpoint=args.checkpoint,
        text=args.text,
        seq_len=args.seq_len,
        steps=args.steps,
    )
    config.dump_folder = str(args.output_dir)
    verify_checkpoint_source(args.checkpoint, args.output_dir)
    trainer = config.build()
    try:
        trainer.train()
        if trainer.engine.num_completed_steps != args.steps:
            raise ValueError(
                f"Requested step {args.steps}, but the trainer finished at "
                f"step {trainer.engine.num_completed_steps}"
            )
        (model,) = trainer.engine.model_parts
        # LoRA B starts at zero. Report the learned projections, not just a loss.
        b_norms = {
            name: param.detach().float().norm().item()
            for name, param in model.named_parameters()
            if param.requires_grad and name.endswith(".lora_b.weight")
        }
        if not all(map(math.isfinite, b_norms.values())):
            raise RuntimeError("Non-finite internal attention LoRA weights")
        if not any(b_norms.values()):
            raise RuntimeError("No internal attention adapter was updated")
        logger.info("Trained internal LoRA B norms: %s", b_norms)
        verify_checkpoint_source(args.checkpoint, args.output_dir)
        export_merged_checkpoint(model, args.checkpoint, destination)
        logger.info(
            "Merged pretrained checkpoint with trained adapters: %s", destination
        )
    finally:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
