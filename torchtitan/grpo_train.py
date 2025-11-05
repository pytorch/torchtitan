# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import time
from datetime import timedelta
from typing import Any, Generator, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import (  # noqa: F401
    DeviceMesh,
    distribute_tensor,
    DTensor,
)

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.grpo.data_handling import OnlineDataHandler
from torchtitan.grpo.grpo_step import (
    compute_grpo_loss_from_predictions,
    compute_logp_from_model,
    scale_rewards,
)
from torchtitan.grpo.sglang_handling import (
    get_hostname_url,
    get_sglang_urls,
    param_to_sglang_data,
    send_param,
    setup_group,
    wait_for_sglang,
)
from torchtitan.grpo.utils import VocabParallelEntropyFunction
from torchtitan.models.attention import init_attention_mask
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    # core configs
    job_config: JobConfig
    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec

    # swappable training components in TrainSpec
    tokenizer: train_spec_module.BaseTokenizer | None
    dataloader: train_spec_module.BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor
    model_args: train_spec_module.BaseModelArgs

    # non-swappable training components
    checkpointer: CheckpointManager
    ft_manager: FTManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Generator[None, None, None]
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]  # type: DeviceMesh
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
            dp_mesh = None

        # TODO: Figure out how to support fault tolerance with separate SGLang groups.
        assert (
            not job_config.fault_tolerance.enable
        ), "Fault tolerance is not supported yet."

        self.ft_manager = FTManager(job_config.fault_tolerance)
        self.dp_degree, self.dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        if parallel_dims.cp_enabled:
            self.cp_degree = world_mesh["cp"].size()
        else:
            self.cp_degree = 1

        if parallel_dims.dp_shard_enabled:
            self.dp_shard_degree = world_mesh["dp_shard"].size()
            self.dp_shard_rank = world_mesh["dp_shard"].get_local_rank()
        else:
            self.dp_shard_degree = 1
            self.dp_shard_rank = 0
        if parallel_dims.dp_replicate_enabled:
            self.dp_replicate_degree = world_mesh["dp_replicate"].size()
            self.dp_replicate_rank = world_mesh["dp_replicate"].get_local_rank()
        else:
            self.dp_replicate_degree = 1
            self.dp_replicate_rank = 0

        if parallel_dims.pp_enabled:
            pp_mesh = world_mesh["pp"]  # type: DeviceMesh

        if parallel_dims.tp_enabled:
            self.tp_mesh = world_mesh["tp"]  # type: DeviceMesh
            self.tp_degree = self.tp_mesh.size()
            self.tp_rank = self.tp_mesh.get_local_rank()
            self.tp_src_rank = torch.distributed.get_rank() - self.tp_rank
            self.tp_group = self.tp_mesh.get_group()
        else:
            self.tp_degree = 1
            self.tp_rank = 0
            self.tp_src_rank = 0
            self.tp_group = None

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )
        self.use_ref_model = (job_config.grpo.kl_beta > 0) and (
            job_config.grpo.kl_estimator_type != "none"
        )
        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # build tokenizer and dataloader
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        self.data_handler = OnlineDataHandler()

        # build model (using meta init)
        model_args = self.train_spec.model_args[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config)
        self.model_args = model_args

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = self.train_spec.model_cls(model_args)
            if self.use_ref_model:
                ref_model = self.train_spec.model_cls(model_args)
            else:
                ref_model = None

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)
        if self.use_ref_model:
            model_converters.convert(ref_model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(job_config, reduction="none")

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.entropy_loss_fn = VocabParallelEntropyFunction.apply

        if job_config.training.epochs is not None:
            raise RuntimeError(
                "Epochs are not supported for RL training, please set steps instead."
            )
        if job_config.checkpoint.interval == "epoch":
            raise RuntimeError(
                "Epochs are not supported for RL training, please use steps instead for checkpointing."
            )

        # apply parallelisms and initialization
        self.ref_model_parts = None
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model
            if self.use_ref_model:
                (_, self.ref_model_parts, _, _,) = self.train_spec.pipelining_fn(
                    model,
                    parallel_dims,
                    job_config,
                    self.device,
                    model_args,
                    self.train_spec.parallelize_fn,
                    self.loss_fn,
                )
                del ref_model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()
            if self.use_ref_model:
                for m in self.ref_model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        m.init_weights(buffer_device=buffer_device)

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]
            if self.use_ref_model:
                ref_model = self.train_spec.parallelize_fn(
                    ref_model, parallel_dims, job_config
                )
                ref_model.to_empty(device=init_device)
                with torch.no_grad():
                    ref_model.init_weights(buffer_device=buffer_device)
                self.ref_model_parts = [ref_model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = CheckpointManager(
            dataloader=None,
            model_parts=self.model_parts,
            ref_model_parts=self.ref_model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(
                    model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enable:
            assert self.train_spec.build_validator_fn is not None

            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.train_spec.build_loss_fn(job_config),
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.debug(
            f"TP RANK: {self.tp_rank}, "
            f"TP GROUP: {torch.distributed.get_process_group_ranks(self.tp_group) if self.tp_group else None}, "
            f"DP RANK: {dp_rank}, "
            f"DP GROUP: {torch.distributed.get_process_group_ranks(dp_mesh.get_group()) if dp_mesh else None}, "
            f"DP Replicate Rank: {self.dp_replicate_rank}, "
            f"DP Shard Rank: {self.dp_shard_rank}, "
        )
        # Wait for SGlang servers to be ready
        job_config.grpo.sglang_urls = get_sglang_urls(job_config)
        wait_for_sglang(job_config.grpo.sglang_urls)

        slurm_logdir = os.environ.get("LOGDIR", None)
        if slurm_logdir is None:
            raise EnvironmentError(
                "LOGDIR environment variable is not set. "
                "Please set the environment variable to point to the "
                "LOGDIR directory where the log files are stored."
            )
        if torch.distributed.get_rank() == 0:
            # only do this one on rank
            sglang_json = {
                "dp_shard_degree": self.dp_shard_degree,
                "tp_degree": self.tp_degree,
                "param_mappings": {},
            }
            for param_name, param in sorted(
                self.model_parts[0].named_parameters(), key=lambda item: item[0]
            ):  # type: str, DTensor
                new_name, needs_permute = param_to_sglang_data(param_name)
                local_shape = list(param.to_local().shape)
                local_shape[0] = (
                    local_shape[0] // self.dp_shard_degree
                )  # need to apply FSDP sharding to dim 0
                sglang_json["param_mappings"][param_name] = {
                    "sglang_name": new_name,
                    "needs_permute": needs_permute,
                    "tp_shard_dim": param.placements[-1].dim
                    if param.placements[-1].is_shard()
                    else 0,
                    "local_shape": local_shape,
                }
            with open(os.path.join(slurm_logdir, "sglang_json.json"), "w") as f:
                json.dump(sglang_json, f, indent=2)
            self.data_handler.register_atropos(job_config, self.step, global_batch_size)

        # setup sglang process
        self.total_group_size = self.dp_degree * self.tp_degree
        self.total_group_size += (
            len(job_config.grpo.sglang_urls) * job_config.grpo.sglang_tp
        )
        local_rank = self.dp_shard_rank * self.tp_degree + self.tp_rank
        self.sglang_nccl_group, self.sglang_gloo_group = None, None
        self.weight_dtypes = {}
        if self.dp_replicate_rank == 0:
            logger.debug("Grabbing sglang dtypes...")
            while not os.path.exists(f"{os.environ['LOGDIR']}/sglang_dtypes.json"):
                time.sleep(1)
            with open(f"{os.environ['LOGDIR']}/sglang_dtypes.json", "r") as f:
                self.weight_dtypes = json.load(f)
            logger.debug(
                f"Setting up SGlang process groups, dp_shard_degree: {self.dp_shard_degree}, tp_degree: {self.tp_degree}"
            )
            hostname = "localhost" if local_rank < 8 else get_hostname_url()
            logger.debug(
                f"total: {self.total_group_size}, rank: {self.dp_shard_rank * self.tp_degree + self.tp_rank}, pg_server: {hostname}"
            )
            self.sglang_nccl_group, self.sglang_gloo_group = setup_group(
                hostname, job_config.grpo.sglang_port, self.total_group_size, local_rank
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def forward_backward_step(
        self,
        batch_dict: dict,
        job_config: JobConfig,
    ) -> Tuple[torch.Tensor, dict, int]:
        # TODO: Reintroduce off policy data
        total_offp_accum = 0

        model_parts = self.model_parts
        parallel_dims = self.parallel_dims
        sequence_lengths = batch_dict.get("sequence_lengths")
        batch = batch_dict["batch"]
        dynamic_scale = batch_dict["dynamic_scale"]
        dynamic_grad_accum_size = batch_dict["dynamic_grad_accum_size"]
        total_masked_tokens = batch_dict["total_masked_tokens"]
        curr_len = -1
        input_ids, labels, masks, inf_logps, rewards = batch
        # if tp_rank == 0:
        device_type = utils.device_type
        input_ids = torch.from_numpy(input_ids).to(device_type)
        labels = torch.from_numpy(labels).to(device_type)
        mask = torch.from_numpy(masks).to(device_type)
        reward = torch.from_numpy(rewards).to(device_type).reshape(-1, 1)
        inf_logps = torch.from_numpy(inf_logps).to(device_type)
        # Multiply by scaling coefs
        reward = scale_rewards(
            reward, job_config.grpo.pos_scaler, job_config.grpo.neg_scaler
        )
        ntokens_since_last_log = labels.numel()
        # convert mask and reward to DTensor
        if self.tp_group is not None:
            mask = DTensor.from_local(mask, device_mesh=self.tp_mesh)
            reward = DTensor.from_local(reward, device_mesh=self.tp_mesh)
            if inf_logps is not None:
                inf_logps = DTensor.from_local(inf_logps, device_mesh=self.tp_mesh)
            else:
                inf_logps = None
        # Create the FlexAttention mask according to the input
        if getattr(self.model_args, "use_flex_attn", False):
            cp_mesh = (
                parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
            )

            init_attention_mask(
                input_ids,
                self.tokenizer.eos_id,
                cp_mesh,
                sequence_lengths=sequence_lengths,
            )
        elif sequence_lengths is not None:
            raise RuntimeError("`sequence_lengths` only supported with FlexAttention")

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        # TODO: actually figure this out if we ever use CP, right now these may be wrong
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[input_ids] + [labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={input_ids, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )
        logger.debug(
            f"nanobatch Input size: {input_ids.size()}, labels size: {labels.size()}"
        )

        if parallel_dims.pp_enabled:
            # TODO: get this working sometime
            raise RuntimeError("PipelineParallelism is not yet supported for GRPO.")
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                # Get pre-computed logps from batch_dict (only for off-policy)
                old_logp = batch_dict["logp"]
                old_ref_logp = batch_dict["ref_logp"]
                with self.maybe_enable_amp:
                    # Compute fresh predictions and logps
                    pred, logp = compute_logp_from_model(
                        self.model_parts[0],
                        input_ids,
                        labels,
                        self.loss_fn,
                        job_config.grpo.temperature,
                    )

                    # Use the comprehensive helper with old_logp for microbatching
                    mb_loss, metrics, _ = compute_grpo_loss_from_predictions(
                        pred=pred,
                        labels=labels,
                        reward=reward,
                        mask=mask,
                        loss_fn=self.loss_fn,
                        entropy_loss_fn=self.entropy_loss_fn,
                        job_config=job_config,
                        device=self.device,
                        old_logp=old_logp,  # Pre-computed logp
                        old_ref_logp=old_ref_logp,
                        ref_pred=None,
                        ref_model=self.ref_model_parts[0]
                        if self.use_ref_model
                        else None,
                        input_ids=input_ids,
                        use_ref_model=self.use_ref_model,
                        inf_logps=inf_logps,
                    )
                    if not job_config.grpo.grpo_by_token:
                        mb_loss = (mb_loss * mask).sum(-1) / mask.sum(-1)
                        mb_loss = mb_loss.mean()
                        mb_loss = (
                            dynamic_scale
                            * mb_loss
                            / (dynamic_grad_accum_size + total_offp_accum)
                        )
                    else:
                        mb_loss = (mb_loss * mask).sum() / (
                            total_masked_tokens + total_offp_accum
                        )
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                del pred
                mb_loss.backward()
                loss = mb_loss.detach()
                # Save batch for off-policy if needed
                # TODO: Reintroduce off-policy
                # if job_config.training.grpo_num_epochs > 1:
                #     with torch.no_grad():
                #         new_batch.append(
                #             (
                #                 input_ids.detach(),
                #                 labels.detach(),
                #                 logp.detach(),
                #                 ref_logp.detach()
                #                 if use_ref_model
                #                 else None,
                #                 mask.detach(),
                #                 reward.detach(),
                #                 dynamic_scale,
                #                 dynamic_grad_accum_size,
                #                 total_masked_tokens,
                #             )
                #         )

        return loss, metrics, ntokens_since_last_log

    def grab_batch(self):
        data_load_start = time.perf_counter()
        (
            batches,
            max_token_len,
            dynamic_batch_size,
            dynamic_grad_acc_size,
            data_lens,
        ) = self.data_handler.data_handling(
            self.sglang_gloo_group,
            self.cp_degree,
            self.dp_degree,
            self.dp_replicate_rank,
            self.device,
            self.job_config,
            self.step,
        )
        data_loading_time = time.perf_counter() - data_load_start
        # Slice to just this dp index
        start = self.dp_rank * dynamic_grad_acc_size
        end = (self.dp_rank + 1) * dynamic_grad_acc_size
        batches = batches[start:end]
        return (
            batches,
            max_token_len,
            dynamic_batch_size,
            dynamic_grad_acc_size,
            data_lens,
            data_loading_time,
        )

    def create_microbatches(
        self,
        job_config: JobConfig,
        batches,
        data_lens,
        max_token_len,
        dynamic_batch_size,
        dynamic_grad_accum_size,
    ):
        batch_prep_time_start = time.perf_counter()
        actual_grad_accum = 0
        microbatches = list()
        for microbatch_idx in range(job_config.grpo.num_microbatches):
            microbatch = []
            mb_start = (
                microbatch_idx
                * dynamic_grad_accum_size
                // job_config.grpo.num_microbatches
            )
            mb_end = (
                (microbatch_idx + 1)
                * dynamic_grad_accum_size
                // job_config.grpo.num_microbatches
            )
            mb_batches = batches[mb_start:mb_end]
            dynamic_batch = list()
            start_len = mb_batches[0][0].shape[1]
            len_step = dynamic_batch_size
            curr_len = -1

            # TODO: Add back in reuse of old batches
            # if len(old_batch) > 0:
            #     if job_config.training.grpo_by_token:
            #         total_offp_accum = sum([old[0][-1] for old in old_batch])
            #     else:
            #         total_offp_accum = sum([old[0][-2] for old in old_batch])
            # else:
            #     total_offp_accum = 0

            # preprocess actual grad accum
            logger.debug(
                f"Beginning microbatch {microbatch_idx + 1}/{job_config.grpo.num_microbatches} setup..."
            )
            total_masked_tokens = 0

            # Process batches in this microbatch
            for accum_idx in range(len(mb_batches)):
                # get batch
                dynamic_batch.append(0)
                if job_config.grpo.grpo_by_token:
                    try:
                        total_masked_tokens += mb_batches[accum_idx][2].sum()
                    except IndexError as err:
                        # If it's index 0, save the json file somewhere...
                        if torch.distributed.get_rank() == 0:
                            filename = f"batch_step_{self.train_state.step + 1}_rank_{self.world_rank}_error.json"
                            logger.error("Data mismatch!")
                            raise AssertionError(
                                "Data mismatch in microbatch creation"
                            ) from err
                if curr_len == -1:
                    curr_len = data_lens[len_step * accum_idx]
                if start_len // curr_len > 1:
                    if accum_idx == len(mb_batches) - 1:
                        # ignore it
                        pass
                    else:
                        mult = start_len // curr_len
                        if len(dynamic_batch) == mult:
                            # ready
                            pass
                        else:
                            # wait
                            continue
                curr_len = -1
                actual_grad_accum += 1
                dynamic_batch = list()
            logger.debug(f"dynamic: {len(mb_batches)}, actual: {actual_grad_accum}")
            dynamic_batch = list()
            new_batch = list()
            total_bs = 0
            for accum_idx in tqdm.tqdm(
                range(len(mb_batches)), disable=os.environ.get("DISABLE_TQDM", False)
            ):
                # get batch
                dynamic_batch.append(mb_batches[accum_idx])
                if curr_len == -1:
                    curr_len = data_lens[len_step * accum_idx]
                if start_len // curr_len > 1:
                    if accum_idx == len(mb_batches) - 1:
                        # ignore it
                        pass
                    else:
                        mult = start_len // curr_len
                        if len(dynamic_batch) == mult:
                            # ready
                            pass
                        else:
                            # wait
                            continue
                total_bs += len(dynamic_batch)
                logger.debug(
                    f"Length dynamic batch: {len(dynamic_batch)}, "
                    f"total_parsed_batches: {total_bs}, "
                    f"total_gas: {dynamic_grad_accum_size}"
                )
                if len(dynamic_batch) > 1:
                    # cat all the stuff
                    batch = [
                        np.concatenate([item[p] for item in dynamic_batch], axis=0)[
                            :, :curr_len
                        ]
                        for p in range(4)
                    ]
                    batch.append(
                        np.concatenate([item[4] for item in dynamic_batch], axis=0)
                    )
                else:
                    batch = [dynamic_batch[0][p][:, :curr_len] for p in range(4)] + [
                        dynamic_batch[0][-1]
                    ]
                dynamic_scale = len(dynamic_batch)
                curr_len = -1
                dynamic_batch = list()

                # Convert to tensors for pre-computation
                input_ids = torch.from_numpy(batch[0]).to(self.device)
                labels = torch.from_numpy(batch[1]).to(self.device)

                # do logp and ref_logp
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=self.world_mesh["cp"],
                        cp_buffers=[input_ids, labels]
                        + [m.freqs_cis for m in self.model_parts],
                        cp_seq_dims=[1, 1] + [0 for _ in self.model_parts],
                        cp_no_restore_buffers={input_ids, labels},
                        cp_rotate_method=job_config.parallelism.context_parallel_rotate_method,
                    )
                    if self.parallel_dims.cp_enabled
                    else None
                )
                with torch.no_grad():
                    if (job_config.grpo.num_microbatches > 1) and (microbatch_idx > 0):
                        with self.train_context(optional_context_parallel_ctx):
                            with self.maybe_enable_amp:
                                pred, logp = compute_logp_from_model(
                                    self.model_parts[0],
                                    input_ids,
                                    labels,
                                    self.loss_fn,
                                    job_config.grpo.temperature,
                                )
                                del pred
                                logp = logp.detach()
                    else:
                        logp = None
                    if self.use_ref_model:
                        with self.train_context(optional_context_parallel_ctx):
                            with self.maybe_enable_amp:
                                ref_pred, ref_logp = compute_logp_from_model(
                                    self.ref_model_parts[0],
                                    input_ids,
                                    labels,
                                    self.loss_fn,
                                    job_config.grpo.temperature,
                                )
                                del ref_pred
                                ref_logp = ref_logp.detach()
                    else:
                        ref_logp = None
                microbatch.append(
                    {
                        "batch": batch,
                        "dynamic_scale": dynamic_scale,
                        "dynamic_grad_accum_size": dynamic_grad_accum_size
                        // job_config.grpo.num_microbatches,
                        "total_masked_tokens": total_masked_tokens,
                        "logp": logp,
                        "ref_logp": ref_logp,
                    }
                )
            microbatches.append(microbatch)
        return (
            microbatches,
            actual_grad_accum,
            time.perf_counter() - batch_prep_time_start,
        )

    def train_step(self):
        logger.debug("prepping training step...")
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        try:
            data_load_start = time.perf_counter()
            logger.debug("grabbing batch...")
            (
                batches,
                max_token_len,
                dynamic_batch_size,
                dynamic_grad_acc_size,
                data_lens,
                data_loading_time,
            ) = self.grab_batch()
            logger.debug("creating microbatches...")
            microbatches, actual_grad_accum, batch_prep_time = self.create_microbatches(
                job_config=self.job_config,
                batches=batches,
                data_lens=data_lens,
                max_token_len=max_token_len,
                dynamic_batch_size=dynamic_batch_size,
                dynamic_grad_accum_size=dynamic_grad_acc_size,
            )
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
        except AssertionError:
            logger.error("Skipping batch due to data handling error")
            return
        logger.debug("performing training step...")
        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        loss = 0.0
        all_metrics = list()
        all_weights = list()
        grad_norms = list()
        logger.debug(f"Microbatches in this step: {len(microbatches)}")
        for mb_idx, microbatch in enumerate(microbatches):
            # For each PPO Microbatch (different from the microbatch in grad acc/dp)
            self.optimizers.zero_grad()
            logger.debug(f"Nanobatches in this step: {len(microbatch)}")
            for nanobatch in microbatch:
                # What a "microbatch" is usually called in a pretraining context
                nb_loss, nb_metrics, n_tokens_seen = self.forward_backward_step(
                    batch_dict=nanobatch,
                    job_config=self.job_config,
                )
                all_metrics.append(nb_metrics)
                all_weights.append(
                    # len(nanobatch["batch"][0])
                    nanobatch["dynamic_scale"]
                    / nanobatch["dynamic_grad_accum_size"]
                )
                nb_loss = nb_loss
                loss += nb_loss.mean().item() / len(microbatches)
                self.ntokens_seen += n_tokens_seen
                # Accumulate tokens for MFU/throughput computation
                self.metrics_processor.ntokens_since_last_log += n_tokens_seen

            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.job_config.training.max_norm,
                foreach=True,
                pp_mesh=(
                    parallel_dims.world_mesh["pp"] if parallel_dims.pp_enabled else None
                ),
                ep_enabled=parallel_dims.ep_enabled,
            )
            grad_norms.append(grad_norm.mean().item())
            self.optimizers.step()
        self.checkpointer.maybe_wait_for_staging()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return
        loss = torch.tensor(loss).to(self.device)
        if parallel_dims.dp_cp_enabled:
            ft_pg = self.ft_manager.loss_sync_pg
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_mean(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    parallel_dims.world_mesh["dp_cp"],
                    ft_pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen
        # Yeah, maybe we should include a max here, but spikes will still be seen in the graph, it's hard to
        # not show a 1000 spike when it's usually below 1.0 after all.
        grad_norm = np.mean(grad_norms)
        extra_metrics = self.collate_metrics(all_metrics, all_weights)
        extra_metrics.update(
            {
                "n_tokens_seen": global_ntokens_seen,
                "loss_metrics/lr": lr,
                "time_metrics/data_prep_s": data_loading_time,
                "time_metrics/batch_prep_s": batch_prep_time,
            }
        )
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics=extra_metrics,
        )

    def collate_metrics(self, metrics_per_nb, weights_per_nb):
        # For now we'll just assume we'll average them all together
        out_metrics = {}
        keys = list()
        for metrics_dict in metrics_per_nb:
            keys.extend(metrics_dict.keys())
        # remove duplicates
        keys = sorted(set(keys))
        ft_pg = self.ft_manager.loss_sync_pg
        for key in keys:
            total_weight = 0.0
            value = None
            for metrics, weight in zip(metrics_per_nb, weights_per_nb):
                if key in metrics:
                    if "min" in key:
                        value = (
                            min(value, metrics[key])
                            if value is not None
                            else metrics[key]
                        )
                        total_weight = 1
                    elif "max" in key:
                        value = (
                            max(value, metrics[key])
                            if value is not None
                            else metrics[key]
                        )
                        total_weight = 1
                    else:
                        if value is not None:
                            value += metrics[key] * weight
                        else:
                            value = metrics[key] * weight
                        total_weight += weight
            value /= total_weight
            logging.debug(f"Metric: {key}, Weight: {total_weight}, value: {value}")
            if "min" in key:
                out_metrics[key] = -dist_utils.dist_max(
                    torch.tensor(-value, dtype=torch.float32, device=self.device),
                    self.parallel_dims.world_mesh["dp_cp"],
                    ft_pg,
                )
            elif "max" in key:
                out_metrics[key] = dist_utils.dist_max(
                    torch.tensor(value, dtype=torch.float32, device=self.device),
                    self.parallel_dims.world_mesh["dp_cp"],
                    ft_pg,
                )
            else:
                out_metrics[key] = dist_utils.dist_mean(
                    torch.tensor(value, device=self.device, dtype=torch.float32),
                    self.parallel_dims.world_mesh["dp_cp"],
                    ft_pg,
                )
        return out_metrics
        for key in list(metrics_per_nb[-1].keys()):
            # check to see which indexes the key is in
            out_metrics[key] = sum(
                [m[key] * w for m, w in zip(metrics_per_nb, weights_per_nb)]
            )
        return out_metrics

    def ema_ref_weights(self):
        if self.use_ref_model and (0.0 < self.job_config.grpo.ref_model_ema < 1.0):
            with torch.no_grad():
                for mdl_param, ref_param in zip(
                    self.model_parts[0].parameters(),
                    self.ref_model_parts[0].parameters(),
                ):  # type: DTensor, (DTensor || Tensor)
                    # Due to weirdness, we need to get the full tensor, then place it back into whatever the ref
                    # model decides to be. Since the mdl param is dp_sharded, we need to do this operation on the
                    # full_tensor as ref_param will not be sharded in the FSDP context.
                    if isinstance(ref_param, DTensor):
                        full_ref = (
                            ref_param.full_tensor().detach()
                        )  # type: torch.Tensor
                    else:
                        full_ref = ref_param.detach()
                    full_pol = mdl_param.full_tensor().detach()  # type: torch.Tensor
                    full_ref.data.lerp_(
                        full_pol.data.to(full_ref.dtype),
                        weight=1.0 - self.job_config.grpo.ref_model_ema,
                    )
                    if isinstance(ref_param, DTensor):
                        # now to distribute it back...
                        updated_ref = distribute_tensor(
                            full_ref, ref_param.device_mesh, ref_param.placements
                        )
                    else:
                        updated_ref = full_ref
                    # then update...
                    ref_param.copy_(updated_ref)
                    del full_ref
                    del full_pol
                    del updated_ref
        torch.distributed.barrier()

    def send_weights(self):
        named_params = {k: v for (k, v) in self.model_parts[0].named_parameters()}
        for name in named_params:
            param = named_params[name]
            logger.debug(
                f"name: {name}, requires_grad: {param.requires_grad}, "
                f"shape: {param.shape}, local_shape: {param.to_local().shape}, placements: {param.placements}"
            )
            if param.requires_grad:
                if self.dp_replicate_rank == 0:
                    logger.debug(
                        f"rank {torch.distributed.get_rank()} sending sglang params for {name}"
                    )
                    local_param = param.to_local()
                    send_param(
                        local_param,
                        name,
                        param.shape,
                        self.weight_dtypes,
                        self.tp_degree,
                        self.dp_shard_degree,
                        self.total_group_size,
                        self.sglang_gloo_group,
                        self.sglang_nccl_group,
                    )
        # To account for fun with updating sglang...
        torch.distributed.barrier()

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as memory_profiler,
            maybe_semi_sync_training(
                job_config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    self.model_args.n_layers
                    if hasattr(self.model_args, "n_layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    self.train_spec.fragment_fn
                    if hasattr(self.train_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step()
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Change timeouts here since we're going away from this comfy highly connected world, potentially, with the
                # sglang updates...
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(minutes=10),
                    world_mesh=self.parallel_dims.world_mesh,
                )

                # Update ref and inference weights...
                self.ema_ref_weights()
                self.send_weights()

                # Back to normal timeout

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # Run validation if validator is available
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout (assuming lazy init and compilation are finished)
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=self.parallel_dims.world_mesh,
                )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metrics_processor:
            self.metrics_processor.close()


if __name__ == "__main__":
    import logging

    init_logger(logging.DEBUG)
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")
