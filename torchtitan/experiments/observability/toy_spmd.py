# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is not a production training recipe. It is just dummy incomplete code
to demonstrate observability APIs.

Toy SPMD training with TP + FSDP2 + per-layer compile on 4 GPUs.
Each rank gets different valid token counts (via loss_mask) to exercise
weighted metric reduction.

Run:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.observability import (
    add_step_tag,
    EventType,
    init_observability,
    MaxMetric,
    NoOpMetric,
    record_event,
    record_metric,
    record_span,
)
from torchtitan.observability.analysis import generate_gantt_trace
from torchtitan.observability.metrics_processor import MetricsProcessor
from torchtitan.tools.logging import init_logger

# ---- Config ----
NUM_STEPS = 20
EVAL_FREQ = 10
D_MODEL = 64
HIDDEN_DIM = 128
N_HEADS = 3
VOCAB_SIZE = 32
SEQ_LEN = 16
BATCH_SIZE = 8
DP_SIZE = 2
LR = 1e-3
IGNORE_INDEX = -100
ENABLE_WANDB = True
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_spmd")


# ---- Data ----
class RepeatDataset:
    """Yields the same batch every step so we can overfit and see loss decrease."""

    def __init__(
        self, tokens: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
    ):
        self.tokens = tokens
        self.labels = labels
        self.loss_mask = loss_mask

    def __iter__(self):
        while True:
            yield self.tokens, self.labels, self.loss_mask


def setup_data(
    device: torch.device = torch.device("cpu"),
    *,
    dp_rank: int = 0,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> RepeatDataset:
    """Create fixed training data for overfitting.

    Each DP rank gets a different seed so loss_mask (and thus valid token
    count) differs across ranks, exercising weighted metric reduction.
    """
    torch.manual_seed(42 + dp_rank)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss_mask = torch.randint(0, 2, (batch_size, seq_len), device=device).float()
    return RepeatDataset(tokens, labels, loss_mask)


# ---- Model ----
class MLPBlock(nn.Module):
    """MLP block with multiple projections (heads) for testing per-head
    observability within compiled regions."""

    def __init__(self, d_model: int, hidden_dim: int, n_heads: int = N_HEADS):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {str(i): nn.Linear(d_model, hidden_dim, bias=False) for i in range(n_heads)}
        )
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        parts = []
        for head in self.heads.values():
            parts.append(F.silu(head(h)))
        return x + self.out_proj(sum(parts))


class TinyModel(nn.Module):
    """3-layer model with embedding, MLP blocks, and output head."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = 3,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleDict(
            {str(i): MLPBlock(d_model, hidden_dim) for i in range(n_layers)}
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h)
        h = self.norm(h)
        return self.output(h)


# ---- Trainer ----
class ToyTrainer:
    """Minimal trainer with TP + compile + FSDP.

    Mirrors the structure of torchtitan/trainer.py:
    - __init__: build model, apply parallelism, create optimizer
    - batch_generator: wraps dataloader, yields batches
    - train_step: one forward/backward/optimizer step
    - validate: one forward pass (no backward)
    - train: owns the training loop
    """

    def __init__(
        self, device, parallel_dims: ParallelDims, output_dir, *, mp_config=None
    ):
        self.device = device
        self.parallel_dims = parallel_dims
        dp_mesh = parallel_dims.get_mesh("fsdp")
        tp_mesh = parallel_dims.get_mesh("tp")
        self.dp_mesh = dp_mesh
        self.rank = dist.get_rank()
        self.output_dir = output_dir
        self.step = 0

        if mp_config is None:
            mp_config = MetricsProcessor.Config()
        self.metrics_processor = mp_config.build(
            parallel_dims=parallel_dims, dump_folder=output_dir
        )

        torch.manual_seed(0)
        with record_span("setup/model_build", EventType.BUILD_MODEL):
            model = TinyModel().to(device)
            self._apply_tp(model, tp_mesh)
            self._apply_compile(model)
            self._apply_fsdp(model, dp_mesh)
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    @staticmethod
    def _replicate_params(module: nn.Module, tp_mesh) -> None:
        """Wrap non-TP-parallelized params as Replicate DTensors on the TP
        mesh so they work with FSDP."""
        for p_name, param in module.named_parameters():
            replicated = nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated)

    def _apply_tp(self, model, tp_mesh):
        """Apply tensor parallelism. Embeddings and output use TP plans;
        remaining params are wrapped as Replicate DTensors."""
        parallelize_module(
            model,
            tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), use_local_output=False
                ),
                "output": ColwiseParallel(
                    output_layouts=Shard(-1), use_local_output=False
                ),
            },
        )
        self._replicate_params(model.norm, tp_mesh)
        for layer in model.layers.values():
            parallelize_module(
                layer,
                tp_mesh,
                {"out_proj": RowwiseParallel(use_local_output=False)},
            )
            for head_name in layer.heads:
                parallelize_module(
                    layer,
                    tp_mesh,
                    {f"heads.{head_name}": ColwiseParallel(use_local_output=False)},
                )
            self._replicate_params(layer.norm, tp_mesh)

    def _apply_compile(self, model):
        """Per-layer torch.compile."""
        for layer_id, block in model.layers.named_children():
            model.layers.register_module(layer_id, torch.compile(block, fullgraph=True))

    def _apply_fsdp(self, model, dp_mesh):
        """FSDP2 wrapping. Applied last (after TP and compile)."""
        fully_shard(model.tok_embeddings, mesh=dp_mesh)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)
        fully_shard([model.norm, model.output], mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

    def batch_generator(self, data_iterable):
        """Wraps a dataloader into an iterator."""
        data_iterator = iter(data_iterable)
        while True:
            with record_span("trainer_time/data_loading_s", EventType.FETCHING_BATCH):
                batch = next(data_iterator)
            yield batch

    def compute_loss(self, logits, labels, loss_mask):
        """Compute cross-entropy loss with masking under loss_parallel.

        Returns (loss_sum, valid_tokens). The caller divides for backward.
        """
        masked_labels = labels.clone().flatten()
        masked_labels[loss_mask.flatten() == 0] = IGNORE_INDEX
        loss_sum = F.cross_entropy(
            logits.flatten(0, 1).float(),
            masked_labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        valid_tokens = (masked_labels != IGNORE_INDEX).sum()
        return loss_sum, valid_tokens

    def train_step(self, tokens, labels, loss_mask):
        """One training step."""
        with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
            with loss_parallel():
                logits = self.model(tokens)
                loss_sum, valid_tokens = self.compute_loss(logits, labels, loss_mask)
                # Globally-normalized loss: each token contributes equally to
                # gradients regardless of which DP rank it's on (matches titan).
                global_valid_tokens = valid_tokens.detach().clone().float()
                dist.all_reduce(global_valid_tokens, group=self.dp_mesh.get_group())
                loss = loss_sum / global_valid_tokens
                self.optimizer.zero_grad()
                loss.backward()

        with record_span("trainer_time/optimizer_s", EventType.OPTIM):
            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Report globally-reduced loss. Each DP rank has
        # local_loss_sum / global_valid_tokens; SUM gives the global loss.
        loss_scalar = loss.detach().full_tensor().clone()
        dist.all_reduce(
            loss_scalar, op=dist.ReduceOp.SUM, group=self.dp_mesh.get_group()
        )
        record_metric("training/loss_mean", NoOpMetric(value=loss_scalar.item()))
        record_metric("training/grad_norm_max", MaxMetric(value=grad_norm.item()))
        record_metric("training/lr", NoOpMetric(value=LR))
        record_event(
            {"train.loss": loss_scalar.item(), "train.grad_norm": grad_norm.item()}
        )

    def validate(self, tokens, labels, loss_mask):
        """Run one forward pass for validation (no backward)."""
        with torch.no_grad(), loss_parallel():
            logits = self.model(tokens)
            loss_sum, valid_tokens = self.compute_loss(logits, labels, loss_mask)
            global_valid_tokens = valid_tokens.detach().clone().float()
            dist.all_reduce(global_valid_tokens, group=self.dp_mesh.get_group())
            val_loss = loss_sum / global_valid_tokens
        val_loss_scalar = val_loss.detach().full_tensor().clone()
        dist.all_reduce(
            val_loss_scalar, op=dist.ReduceOp.SUM, group=self.dp_mesh.get_group()
        )
        record_metric("validation/loss_mean", NoOpMetric(value=val_loss_scalar.item()))

    def train(self, num_steps):
        """Full training loop. Mirrors Trainer.train structure."""
        dp_rank = self.dp_mesh.get_local_rank()
        dataloader = setup_data(self.device, dp_rank=dp_rank)
        data_iterator = self.batch_generator(dataloader)

        for step in range(1, num_steps + 1):
            self.step = step
            self.metrics_processor.set_step(step)

            # Simulate GC on every 5th step (mirrors gc_handler.run)
            if step % 5 == 0:
                add_step_tag("gc")

            with record_span("trainer_time/step_s", EventType.STEP):
                tokens, labels, loss_mask = next(data_iterator)
                self.train_step(tokens, labels, loss_mask)

            if step % EVAL_FREQ == 0:
                add_step_tag("eval")
                with record_span("trainer_time/validation_s", EventType.EVAL):
                    self.validate(tokens, labels, loss_mask)

            self.metrics_processor.log(step)

    def close(self):
        """Cleanup."""
        self.metrics_processor.close()


def main():
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()
    assert world_size == 4, f"Requires 4 GPUs, got {world_size}"

    if rank == 0 and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    dist.barrier()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_logger()
    init_observability(source="trainer", output_dir=OUTPUT_DIR, rank=rank)

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=DP_SIZE,
        cp=1,
        tp=world_size // DP_SIZE,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DPx2TP, {NUM_STEPS} steps")

    mp_config = MetricsProcessor.Config(
        enable_wandb=ENABLE_WANDB,
        console_log_metric_keys=[
            "training/loss_mean",
            "training/grad_norm_max",
            "training/lr",
        ],
    )
    trainer = ToyTrainer(device, parallel_dims, OUTPUT_DIR, mp_config=mp_config)
    trainer.train(NUM_STEPS)
    trainer.close()

    if rank == 0:
        sys_logs = os.path.join(OUTPUT_DIR, "system_logs")
        trace_path = os.path.join(OUTPUT_DIR, "analysis", "system_metrics_gantt.json")
        generate_gantt_trace(sys_logs, trace_path)
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
