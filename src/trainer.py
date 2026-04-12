"""Core training loop for MoE model."""

import math
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful

from src.components.checkpoint import CheckpointManager
from src.components.lr_scheduler import build_lr_schedulers
from src.components.optimizer import build_optimizers_with_moe_load_balancing
from src.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer, resolve_tokenizer_path
from src.config import TORCH_DTYPE_MAP
from src.config.config import Config, build_job_config
from src.data import DataloaderExhaustedError, build_text_dataloader
from src.distributed import ParallelDims
from src.distributed import utils as dist_utils
from src.logging import init_logger, logger
from src.models.moe.model import MoETransformer
from src.models.parallelize import apply_fsdp, apply_moe_ep


class TrainState(Stateful):
    def __init__(self):
        self.step = 0
        self.ntokens_seen = 0

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    cfg: Config,
    job_config: Any,
    dp_world_size: int,
    dp_rank: int,
    batch_mesh: Any,
) -> dict[str, float]:
    """Run one full pass over the eval dataset and return loss/ppl/top1 metrics.

    All ranks must execute the same number of forward passes because FSDP issues
    collectives during forward. We use an all-reduce MIN of a "has batch" flag
    so every rank stops together once any rank runs out of data.
    """
    was_training = model.training
    model.eval()

    eval_loader = build_text_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        job_config=job_config,
        infinite=False,
        dataset_path=cfg.eval.dataset_path,
    )

    local_loss_sum = 0.0  # sum over per-token cross entropy
    local_correct = 0
    local_tokens = 0

    data_iter = iter(eval_loader)
    multi_rank = batch_mesh is not None and batch_mesh.size() > 1
    pg = batch_mesh.get_group() if multi_rank else None

    while True:
        input_dict: dict[str, torch.Tensor] | None = None
        labels: torch.Tensor | None = None
        try:
            input_dict, labels = next(data_iter)
            has_batch = 1
        except (StopIteration, DataloaderExhaustedError):
            has_batch = 0

        # All ranks must stop together
        if multi_rank:
            flag = torch.tensor([has_batch], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=pg)
            if flag.item() == 0:
                break
        elif has_batch == 0:
            break

        assert input_dict is not None and labels is not None
        tokens = input_dict["input"].to(device)
        labels_dev = labels.to(device)
        pred = model(tokens)

        pred_flat = pred.flatten(0, 1).float()
        labels_flat = labels_dev.flatten(0, 1)

        per_token_loss = F.cross_entropy(pred_flat, labels_flat, reduction="none")
        local_loss_sum += per_token_loss.sum().item()

        top1 = pred_flat.argmax(-1)
        local_correct += (top1 == labels_flat).sum().item()
        local_tokens += labels_flat.numel()

    # Reduce across DP ranks
    stats = torch.tensor(
        [local_loss_sum, local_correct, local_tokens],
        device=device,
        dtype=torch.float64,
    )
    if multi_rank:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=pg)

    total_loss_sum, total_correct, total_tokens = stats.tolist()
    if was_training:
        model.train()

    if total_tokens == 0:
        logger.warning("Eval set was empty across all ranks")
        return {"loss": float("nan"), "ppl": float("nan"), "top1_acc": float("nan")}

    loss_mean = total_loss_sum / total_tokens
    return {
        "loss": loss_mean,
        "ppl": math.exp(min(loss_mean, 20.0)),  # cap to avoid float overflow
        "top1_acc": total_correct / total_tokens,
    }


def train(cfg: Config):
    """Run MoE model training from a resolved Config."""
    init_logger(log_dir=cfg.logging.log_dump)

    # quack and compile are mutually exclusive: CuTe-DSL kernels are opaque to Dynamo
    if cfg.quack.enable and cfg.compile.enable:
        raise ValueError(
            "quack.enable and compile.enable are mutually exclusive: "
            "QuACK kernels cannot be traced by torch.compile"
        )

    job_config = build_job_config(cfg)

    # Init distributed
    world_size = dist_utils.init_distributed(job_config.comm)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=cfg.parallelism.dp_shard,
        cp=1, tp=1, pp=1,
        ep=cfg.parallelism.ep,
        etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    # Device and seeds
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    dist_utils.set_determinism(
        parallel_dims, device, job_config.debug, distinct_seed_mesh_dims=[]
    )

    # Tokenizer
    tokenizer_path = resolve_tokenizer_path(cfg.data.tokenizer)
    tokenizer = HuggingFaceTokenizer(tokenizer_path)

    # Model config: seq_len for RoPE, vocab_size from tokenizer (unless pinned), quack flag
    vocab_size = cfg.model.vocab_size or tokenizer.get_vocab_size()
    if cfg.model.vocab_size is None:
        logger.info(f"Auto-detected vocab_size={vocab_size} from tokenizer")
    model_cfg = cfg.model.model_copy(
        update={
            "max_seq_len": cfg.training.seq_len,
            "vocab_size": vocab_size,
            "use_quack": cfg.quack.enable,
        }
    )

    # Dataloader
    batch_mesh = parallel_dims.get_optional_mesh("batch")
    if batch_mesh is not None:
        dp_world_size = batch_mesh.size()
        dp_rank = batch_mesh.get_local_rank()
    else:
        dp_world_size = 1
        dp_rank = 0
    dataloader = build_text_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        job_config=job_config,
    )

    # Gradient accumulation
    grad_accum_steps = cfg.training.grad_accum_steps
    global_batch_size = (
        cfg.training.local_batch_size * dp_world_size * grad_accum_steps
    )

    # Build model on meta device
    logger.info(
        f"Building MoE model: {cfg.model.n_layers}L, dim={cfg.model.dim}, "
        f"{cfg.model.num_experts} experts, top_k={cfg.model.top_k}"
    )
    with torch.device("meta"):
        model = MoETransformer(model_cfg)

    # Verify param groups
    expert_params = sum(
        p.numel() for n, p in model.named_parameters() if "moe.experts" in n
    )
    non_expert_params = sum(
        p.numel() for n, p in model.named_parameters() if "moe.experts" not in n
    )
    assert expert_params > 0, "No expert params found"
    assert non_expert_params > 0, "No non-expert params found"
    logger.info(
        f"Params: {expert_params + non_expert_params:,} total "
        f"({expert_params:,} expert, {non_expert_params:,} non-expert)"
    )

    # Parallelism: EP → AC → compile → FSDP (order matters)
    if parallel_dims.ep_enabled:
        apply_moe_ep(model, parallel_dims.get_mesh("ep"))

    compile_enabled = cfg.compile.enable
    if job_config.activation_checkpoint.mode != "none":
        from src.distributed.activation_checkpoint import apply_ac

        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=compile_enabled,
        )
        logger.info(
            f"Applied activation checkpointing: mode={job_config.activation_checkpoint.mode}"
        )

    if compile_enabled:
        from src.models.parallelize import apply_compile

        apply_compile(model, backend=cfg.compile.backend, ep_enabled=parallel_dims.ep_enabled)

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        apply_fsdp(
            model,
            parallel_dims.get_mesh("fsdp"),
            param_dtype=TORCH_DTYPE_MAP[cfg.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[cfg.training.mixed_precision_reduce],
            ep_degree=cfg.parallelism.ep,
            edp_mesh=parallel_dims.get_optional_mesh("efsdp"),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

    model.to_empty(device=device)
    model.init_weights(buffer_device=device)
    model.train()

    # Optimizer, LR scheduler, checkpoint
    optimizers = build_optimizers_with_moe_load_balancing(
        [model], job_config.optimizer, parallel_dims
    )
    lr_schedulers = build_lr_schedulers(
        optimizers, job_config.lr_scheduler, cfg.training.max_steps
    )
    train_state = TrainState()
    checkpointer = CheckpointManager(
        dataloader=dataloader,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        checkpoint_config=job_config.checkpoint,
        sd_adapter=None,
        base_folder="",  # checkpoint_dump is passed as absolute folder via build_job_config
    )
    checkpointer.load(step=job_config.checkpoint.load_step)

    total_steps = cfg.training.max_steps
    seq_len = cfg.training.seq_len
    max_norm = cfg.training.max_norm
    log_step = cfg.logging.log_step
    eval_step = cfg.eval.eval_step if cfg.eval.enable else 0
    tokens_per_step = global_batch_size * seq_len

    if cfg.eval.enable and not cfg.eval.dataset_path:
        raise ValueError("eval.enable is True but eval.dataset_path is empty")

    logger.info(
        f"Training: steps={total_steps}, "
        f"local_batch={cfg.training.local_batch_size}, "
        f"global_batch={global_batch_size}, "
        f"grad_accum={grad_accum_steps}, "
        f"seq_len={seq_len}"
    )

    # Training loop
    if cfg.quack.enable:
        from quack import cross_entropy as quack_cross_entropy

        # quack's cross_entropy backward kernel requires its grad input to have
        # stride 1, but .sum()/count inside reduction="mean" produces a stride-0
        # broadcast grad. We work around it by calling reduction="none" and
        # materializing the grad via an identity autograd.Function that
        # contiguates grad_output before it reaches the custom kernel.
        class _ContiguousGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.contiguous()

        def cross_entropy_fn(pred, labels):
            pred_flat = pred.flatten(0, 1).float()
            labels_flat = labels.flatten(0, 1)
            per_token = quack_cross_entropy(pred_flat, labels_flat, reduction="none")
            per_token = _ContiguousGrad.apply(per_token)
            return per_token.mean()
    else:
        def cross_entropy_fn(pred, labels):
            return F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    data_iter = iter(dataloader)
    start_time = time.perf_counter()

    while train_state.step < total_steps:
        train_state.step += 1
        step = train_state.step
        step_start = time.perf_counter()

        optimizers.zero_grad()
        accumulated_loss = 0.0

        for _ in range(grad_accum_steps):
            input_dict, labels = next(data_iter)
            tokens = input_dict["input"].to(device)
            labels = labels.to(device)
            pred = model(tokens)
            loss = cross_entropy_fn(pred, labels)
            (loss / grad_accum_steps).backward()
            accumulated_loss += loss.detach().item() / grad_accum_steps

        grad_norm = dist_utils.clip_grad_norm_(
            list(model.parameters()),
            max_norm,
            foreach=True,
            ep_enabled=parallel_dims.ep_enabled,
        )

        checkpointer.maybe_wait_for_staging()
        optimizers.step()
        lr_schedulers.step()
        train_state.ntokens_seen += tokens_per_step

        checkpointer.save(step, last_step=(step == total_steps))

        if step % log_step == 0:
            step_time = time.perf_counter() - step_start
            lr = lr_schedulers.schedulers[0].get_last_lr()[0]
            logger.info(
                f"step: {step}/{total_steps}  "
                f"loss: {accumulated_loss:.4f}  "
                f"grad_norm: {grad_norm:.4f}  "
                f"lr: {lr:.2e}  "
                f"tok/s: {tokens_per_step / step_time:,.0f}  "
                f"step_time: {step_time:.2f}s"
            )

        if eval_step > 0 and step % eval_step == 0:
            eval_start = time.perf_counter()
            metrics = run_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                cfg=cfg,
                job_config=job_config,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                batch_mesh=batch_mesh,
            )
            eval_time = time.perf_counter() - eval_start
            logger.info(
                f"eval @ step {step}/{total_steps}  "
                f"loss: {metrics['loss']:.4f}  "
                f"ppl: {metrics['ppl']:.2f}  "
                f"top1: {metrics['top1_acc']:.4f}  "
                f"eval_time: {eval_time:.2f}s"
            )

    logger.info(
        f"Training completed in {time.perf_counter() - start_time:.1f}s  "
        f"tokens_seen: {train_state.ntokens_seen:,}"
    )
    checkpointer.close()
    dist.destroy_process_group()
