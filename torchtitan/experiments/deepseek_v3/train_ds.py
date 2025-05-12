# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 8 train.py
# bash run_training.sh

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.datasets.tokenizer.hf_tokenizer import get_hf_tokenizer
from torchtitan.experiments.deepseek_v3.infra.parallelize_deepseek import (
    parallelize_deepseek,
)

# from checkpoint import load_weights_from_hf
from torchtitan.experiments.deepseek_v3.models.model import DeepseekForCausalLM
from torchtitan.experiments.deepseek_v3.models.model_config import (
    deepseek_config_registry,
)
from torchtitan.tools.logging import init_logger, logger


# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"

# dataloader

"""
tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
        )
"""


# Run full model
def run_full_model(
    config: JobConfig,
):

    # setup mesh
    pp_dim = config.parallelism.pipeline_parallel_degree
    ep_dim = config.parallelism.expert_parallel_degree
    fsdp_dim = config.parallelism.data_parallel_shard_degree
    logger.info(f"{pp_dim=}, {ep_dim=}, {fsdp_dim=}")

    world_mesh = dist.init_device_mesh(
        "cuda", (pp_dim, ep_dim, fsdp_dim), mesh_dim_names=("pp", "ep", "fsdp")
    )

    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    model_args = deepseek_config_registry.get(model_id, None)
    if model_args is None:
        raise ValueError(f"Model {model_id} not found in registry.")

    # TODO - remove this for full model
    # model_args.num_hidden_layers = 16

    (
        model,
        pp_size,
        pp_rank,
        pp_mesh,
        ep_size,
        ep_rank,
    ) = parallelize_deepseek(world_mesh, device, model_args, rank)

    # build tokenizer
    tokenizer = get_hf_tokenizer(model_id)

    from torchtitan.datasets.hf_datasets import build_hf_dataloader

    # TODO - ep is not the same as dp really...just a temp shim atm.
    dataloader = build_hf_dataloader(
        dp_world_size=ep_size, dp_rank=ep_rank, tokenizer=tokenizer, job_config=config
    )
    logger.info(f"Success! {dataloader=}")

    # Synthetic setting
    microbatches = pp_size * 2

    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    # Example inputs

    torch.manual_seed(ep_rank)
    bs = config.training.batch_size  # 4
    seqlen = config.training.seq_len  # 128

    x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
    label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy

    # Run forward and backward
    steps = 2
    loss = float("inf")
    for _ in range(steps):
        if pp_size > 1:
            # Create pipeline stage
            stage = PipelineStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )

            # Create pipeline schedule
            losses = []
            pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)

            if pp_rank == 0:
                y = pp_schedule.step(x)
            elif pp_rank == pp_size - 1:
                y = pp_schedule.step(target=label, losses=losses)
                loss = torch.mean(torch.stack(losses))
            else:
                pp_schedule.step()
        else:
            y = model(x)
            loss = loss_fn(y, label)
            loss.backward()

        if pp_rank == pp_size - 1:
            logger.info(f"logits: {y.shape}")
            logger.info(f"{loss=}")

        if pp_rank == 0:
            param = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
            logger.info(f"{torch.linalg.norm(param.grad)=}")

        model.zero_grad()

    logger.info("Backward done")


if __name__ == "__main__":

    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    """trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
    """

    run_full_model(config)

    dist.destroy_process_group()
