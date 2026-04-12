"""Core training loop for MoE model."""

import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.components.checkpoint import CheckpointManager
from src.components.lr_scheduler import build_lr_schedulers
from src.components.optimizer import build_optimizers_with_moe_load_balancing
from src.components.tokenizer import HuggingFaceTokenizer, resolve_tokenizer_path
from src.components.train_state import TrainState
from src.config import TORCH_DTYPE_MAP
from src.config.config import Config, build_job_config
from src.distributed import ParallelDims
from src.distributed import utils as dist_utils
from src.hf_datasets.text_datasets import build_text_dataloader
from src.models.moe.model import MoETransformer
from src.models.parallelize import apply_fsdp, apply_moe_ep
from src.tools.logging import init_logger, logger


def train(cfg: Config):
    """Run MoE model training from a resolved Config."""
    init_logger()
    job_config = build_job_config(cfg)

    # Init distributed
    world_size = dist_utils.init_distributed(job_config.comm)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=cfg.parallelism.dp_shard,
        cp=1,
        tp=1,
        pp=1,
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

    # Model config with seq_len for RoPE
    model_cfg = cfg.model.model_copy(update={"max_seq_len": cfg.training.seq_len})

    # Tokenizer and dataloader
    tokenizer_path = resolve_tokenizer_path(cfg.data.tokenizer)
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
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

    # Parallelism: EP then FSDP
    if parallel_dims.ep_enabled:
        apply_moe_ep(model, parallel_dims.get_mesh("ep"))

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
        base_folder=cfg.dump_folder,
    )
    checkpointer.load(step=job_config.checkpoint.load_step)

    total_steps = cfg.training.max_steps
    seq_len = cfg.training.seq_len
    max_norm = cfg.training.max_norm
    log_freq = cfg.training.log_freq
    tokens_per_step = global_batch_size * seq_len

    logger.info(
        f"Training: steps={total_steps}, "
        f"local_batch={cfg.training.local_batch_size}, "
        f"global_batch={global_batch_size}, "
        f"grad_accum={grad_accum_steps}, "
        f"seq_len={seq_len}"
    )

    # Training loop
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
            loss = F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))
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

        if step % log_freq == 0:
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

    logger.info(
        f"Training completed in {time.perf_counter() - start_time:.1f}s  "
        f"tokens_seen: {train_state.ntokens_seen:,}"
    )
    checkpointer.close()
    dist.destroy_process_group()
