from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

import einops
import torch
import torch.nn as nn

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.components.validate import BaseValidator
from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer
from xx.common.helpers import parse_info
from xx.training.diffusion.schedulers import RFScheduler

from .dataset import WorldModelDataLoader
from .loss import WorldModelLoss
from .model import WorldModel
from .tokenizer import WorldModelTokenizer


ValidationContext = Callable[[], AbstractContextManager[None]]


def _batch_size(inputs: dict[str, torch.Tensor]) -> int:
    return next(iter(inputs.values())).shape[0]


def _tensor_dict_to_device(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}


def _segment_names_from_info(info: torch.Tensor) -> list[str]:
    names: list[str] = []
    for item in info.cpu().numpy():
        try:
            name = parse_info(item).get("name")
        except ValueError:
            continue
        if name is not None:
            names.append(str(name))
    return names


def _prepare_worldmodel_batch(
    *,
    model: WorldModel,
    tokenizer: WorldModelTokenizer,
    input_dict: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    scheduler: RFScheduler,
    discrete_timesteps: torch.Tensor,
    pose_dropout: float,
    inference_conditioning_frames: int,
    future_size_frames: int,
    no_noise_conditioning_frames_prob: float,
    fake_timesteps_prob: float,
    train: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    latents = tokenizer.encode(input_dict, device=device, dtype=dtype)
    augments = input_dict["augments_pos_ref_augment"].to(device=device, dtype=dtype).clone()
    eulers = input_dict["ref_augment_from_augments_euler"].to(device=device, dtype=dtype).clone()
    fidxs = input_dict["fidxs"].to(device=device, dtype=torch.int64)
    targets = _tensor_dict_to_device(targets, device)
    batch_size, num_frames = latents.shape[:2]

    with torch.no_grad():
        latents = model.scale_latents(latents)
        noise = torch.randn_like(latents)
        if train:
            timesteps = scheduler.sample_timestep((batch_size,))
        else:
            indexes = torch.randint(0, discrete_timesteps.numel(), (batch_size,), device=device)
            timesteps = discrete_timesteps[indexes]
        timesteps = einops.repeat(timesteps, "b -> b t", t=num_frames).clone()

        pose_mask = torch.ones((batch_size, num_frames), device=device, dtype=torch.bool)
        if inference_conditioning_frames < num_frames:
            drop = torch.rand((batch_size, 1), device=device) < pose_dropout
            pose_mask[:, inference_conditioning_frames:] = drop
        augments[pose_mask] = 0
        eulers[pose_mask] = 0

        mask = torch.ones_like(latents, device=device, dtype=torch.bool)
        fake_timesteps = timesteps.clone()
        if torch.rand((), device=device) < no_noise_conditioning_frames_prob:
            end = min(inference_conditioning_frames, num_frames)
            mask[:, :end] = False
            timesteps[:, :end] = scheduler.no_noise_timestep
            fake_timesteps[:, :end] = scheduler.no_noise_timestep
            start = min(future_size_frames, end)
            if start < end and torch.rand((), device=device) < fake_timesteps_prob:
                fake_timesteps[:, start:end] = scheduler.sample_timestep((batch_size, end - start))
                mask[:, start:] = True

        noisy_latents = scheduler.add_noise(latents, noise, fake_timesteps)
        targets = {**targets, "v": latents - noise, "mask": mask}

    return {
        "x": noisy_latents,
        "t": timesteps,
        "augments_pos_ref_augment": augments,
        "ref_augment_from_augments_euler": eulers,
        "pose_mask": pose_mask.to(dtype=torch.int64),
        "fidx": fidxs,
    }, targets


class WorldModelValidator(BaseValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool
        steps: int
        dataloader: WorldModelDataLoader.Config
        pose_dropout: float
        noise_scheduler_steps: int
        no_noise_conditioning_frames_prob: float
        fake_timesteps_prob: float

        def __post_init__(self) -> None:
            if self.steps < 0:
                raise ValueError("worldmodel validation steps must be >= 0")

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: WorldModelLoss,
        validation_context: ValidationContext,
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
        **kwargs: Any,
    ) -> None:
        del parallelism, kwargs
        super().__init__(config=config)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.tokenizer = cast(WorldModelTokenizer, tokenizer)
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{training_id}:worldmodel:validation")

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        if self.config.steps == 0:
            return
        model = cast(WorldModel, model_parts[0])
        model.eval()
        device = next(model.parameters()).device
        dtype = _floating_model_dtype(model)
        scheduler = RFScheduler(steps=self.config.noise_scheduler_steps).to(device=device)
        discrete_timesteps = scheduler.timesteps[:-1]
        dataloader = self.config.dataloader.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
            validation_steps=self.config.steps,
        )
        samples = 0
        term_sums: dict[str, torch.Tensor] = {}
        try:
            for num_steps, (input_dict, targets) in enumerate(dataloader):
                if num_steps >= self.config.steps:
                    break
                batch_size = _batch_size(input_dict)
                samples += batch_size
                self.metrics_processor.ntokens_since_last_log += batch_size * self.seq_len
                if "info" in input_dict:
                    self.unique_segment_counter.update(_segment_names_from_info(input_dict["info"]))
                model_inputs, targets = _prepare_worldmodel_batch(
                    model=model,
                    tokenizer=self.tokenizer,
                    input_dict=input_dict,
                    targets=targets,
                    device=device,
                    dtype=dtype,
                    scheduler=scheduler,
                    discrete_timesteps=discrete_timesteps,
                    pose_dropout=self.config.pose_dropout,
                    inference_conditioning_frames=self.config.dataloader.inference_conditioning_frames,
                    future_size_frames=self.config.dataloader.future_size_frames,
                    no_noise_conditioning_frames_prob=self.config.no_noise_conditioning_frames_prob,
                    fake_timesteps_prob=self.config.fake_timesteps_prob,
                    train=False,
                )
                with self.validation_context():
                    outputs = model(**model_inputs)
                    _loss_vec, terms = self.loss_fn(outputs, targets)
                for name, term in terms.items():
                    term_sums[name] = term_sums.get(name, torch.zeros((), device=device)) + term.float().sum()
        finally:
            dataloader.close()
            model.train()

        if samples == 0:
            return
        sample_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        global_samples = dist_utils.dist_sum(sample_tensor, batch_mesh) if batch_mesh is not None else float(sample_tensor.item())
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")

        def global_mean(value: torch.Tensor) -> float:
            total = dist_utils.dist_sum(value, loss_mesh) if self.parallel_dims.dp_cp_enabled else float(value.item())
            return total / global_samples

        loss = global_mean(term_sums["loss"])
        extra_metrics = {f"validation_metrics/worldmodel/{name}": global_mean(value) for name, value in term_sums.items() if name != "loss"}
        extra_metrics["validation_metrics/dataset/unique_segments_seen"] = (
            self.unique_segment_counter.global_count(batch_mesh.get_group())
            if batch_mesh is not None
            else self.unique_segment_counter.local_count()
        )
        self.metrics_processor.log_validation(loss=loss, step=step, extra_metrics=extra_metrics)


class WorldModelTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        loss: WorldModelLoss.Config
        dataloader: WorldModelDataLoader.Config
        tokenizer: WorldModelTokenizer.Config
        validator: WorldModelValidator.Config
        pose_dropout: float
        noise_scheduler_steps: int
        no_noise_conditioning_frames_prob: float
        fake_timesteps_prob: float

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            _validate_worldmodel_config(self)

    def __init__(self, config: Config):
        super().__init__(config)
        self.dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]
        self.train_noise_scheduler = RFScheduler(steps=config.noise_scheduler_steps).to(device=self.device)
        self.discrete_timesteps = self.train_noise_scheduler.timesteps[:-1]
        self.tokenizer = cast(WorldModelTokenizer, self.tokenizer)
        self.loss_fn = cast(WorldModelLoss, self.loss_fn)
        training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{training_id}:worldmodel:train")

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                input_dict, targets = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.metrics_processor.ntokens_since_last_log += _batch_size(input_dict) * self.config.training.seq_len
            self.metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)
            yield input_dict, targets

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        local_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = cast(WorldModel, self.model_parts[0])
        with sl.log_trace_span("worldmodel_prepare_batch"):
            model_inputs, targets = _prepare_worldmodel_batch(
                model=model,
                tokenizer=self.tokenizer,
                input_dict=input_dict,
                targets=targets,
                device=self.device,
                dtype=self.dtype,
                scheduler=self.train_noise_scheduler,
                discrete_timesteps=self.discrete_timesteps,
                pose_dropout=self.config.pose_dropout,
                inference_conditioning_frames=self.config.dataloader.inference_conditioning_frames,
                future_size_frames=self.config.dataloader.future_size_frames,
                no_noise_conditioning_frames_prob=self.config.no_noise_conditioning_frames_prob,
                fake_timesteps_prob=self.config.fake_timesteps_prob,
                train=True,
            )
        self.ntokens_seen += model_inputs["x"].shape[0] * self.config.training.seq_len
        with self.train_context():
            outputs = model(**model_inputs)
            loss_vec, terms = self.loss_fn(outputs, targets)
            loss = loss_vec.sum() / local_samples
            del outputs, loss_vec
            loss.backward()
        return loss, terms

    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> None:
        self.optimizers.zero_grad()
        lr_metrics = self.lr_schedulers.get_metrics()
        parallel_dims = self.parallel_dims
        batch_mesh = parallel_dims.get_optional_mesh("batch")

        microbatches = []
        step_segment_names: set[str] = set()
        local_samples = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, targets = next(data_iterator)
            local_samples += _batch_size(input_dict)
            if "info" in input_dict:
                step_segment_names.update(_segment_names_from_info(input_dict["info"]))
            microbatches.append((input_dict, targets))
        sl.log_trace_scalar({"local_samples": int(local_samples)})

        local_samples = local_samples.to(self.device)
        global_samples = dist_utils.dist_sum(local_samples, batch_mesh) if batch_mesh is not None else float(local_samples.item())
        local_samples_float = local_samples.to(dtype=torch.float32)

        accumulated_losses = []
        metric_sums: dict[str, torch.Tensor] = {}
        for input_dict, targets in microbatches:
            loss, metrics = self.forward_backward_step(
                input_dict=input_dict,
                targets=targets,
                local_samples=local_samples_float,
            )
            accumulated_losses.append(loss.detach())
            for name, value in metrics.items():
                if name == "loss":
                    continue
                metric_sums[name] = metric_sums.get(name, torch.zeros((), device=self.device)) + value.float().sum()

        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        self.unique_segment_counter.update(step_segment_names)

        if not self.metrics_processor.should_log(self.step):
            return

        loss = torch.sum(torch.stack(accumulated_losses))
        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_sum(loss * local_samples_float, loss_mesh) / global_samples
            global_max_loss = dist_utils.dist_max(loss.detach(), loss_mesh)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
            )
            metric_values = {name: dist_utils.dist_sum(value, loss_mesh) / global_samples for name, value in metric_sums.items()}
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen
            metric_values = {name: float((value / global_samples).item()) for name, value in metric_sums.items()}

        extra_metrics: dict[str, Any] = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **{f"worldmodel/{name}": value for name, value in metric_values.items()},
            "dataset/unique_segments_seen": (
                self.unique_segment_counter.global_count(batch_mesh.get_group())
                if batch_mesh is not None
                else self.unique_segment_counter.local_count()
            ),
        }
        stats = self.dataloader.stats() if hasattr(self.dataloader, "stats") else None
        if stats is not None:
            extra_metrics.update(
                {
                    "dataloader/shuffle_full": stats.full,
                    "dataloader/shuffle_empty": stats.empty,
                    "dataloader/shuffle_in_flight": stats.in_flight,
                }
            )
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def close(self) -> None:
        self.dataloader.close()
        super().close()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["unique_segment_counter"] = self.unique_segment_counter.state_dict()
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None:
            state["validation_unique_segment_counter"] = validator_unique_segment_counter.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "unique_segment_counter" in state_dict:
            self.unique_segment_counter.load_state_dict(state_dict["unique_segment_counter"])
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None and "validation_unique_segment_counter" in state_dict:
            validator_unique_segment_counter.load_state_dict(state_dict["validation_unique_segment_counter"])


def _floating_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _validate_worldmodel_config(config: WorldModelTrainer.Config) -> None:
    if config.model_spec is None:
        raise ValueError("worldmodel requires a model_spec")
    if not isinstance(config.model_spec.model, WorldModel.Config):
        raise TypeError("worldmodel model_spec must contain WorldModel.Config")
    model_config = config.model_spec.model
    total_frames = config.dataloader.context_size_frames + config.dataloader.future_size_frames
    if model_config.input_size[0] != total_frames:
        raise ValueError(f"model input frames {model_config.input_size[0]} != dataloader frames {total_frames}")
    if model_config.input_size[1:] != config.dataloader.latent_size:
        raise ValueError("model input spatial size must match dataloader latent_size")
    if model_config.in_channels != config.dataloader.in_channels:
        raise ValueError("model in_channels must match dataloader in_channels")
    if config.parallelism.tensor_parallel_degree > 1 or config.parallelism.pipeline_parallel_degree > 1 or config.parallelism.context_parallel_degree > 1 or config.parallelism.expert_parallel_degree > 1:
        raise ValueError("worldmodel supports FSDP/HSDP only")
    model_config._sync_derived_fields()
    config.training.seq_len = model_config.num_patches


def _set_single_process_env() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")


def _stock_trainer_mock_config() -> WorldModelTrainer.Config:
    from torchtitan.components.optimizer import default_adamw

    from .config_registry import _dataloader_config, model_registry, worldmodel

    config = worldmodel()
    model_spec = model_registry("debugmodel")

    config.dump_folder = "./outputs/worldmodel_trainer_mock"
    config.model_spec = model_spec
    config.dataloader = _dataloader_config(
        split="train",
        shuffle_size=8,
        min_mixing=0.25,
        num_writers=1,
        num_readers=1,
        latent_size=model_spec.model.input_size[1:],
        mock_data=True,
        mock_segment_batch_size=2,
    )
    config.optimizer = default_adamw(lr=1e-3)
    config.lr_scheduler.warmup_steps = 0
    config.lr_scheduler.total_steps = 1
    config.training.local_batch_size = 2
    config.training.global_batch_size = 2
    config.training.steps = 1
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "float32"
    config.checkpoint.enable = False
    config.activation_checkpoint.mode = "none"
    config.compile.enable = False
    config.metrics.log_freq = 1
    config.metrics.enable_reporterv2 = False
    config.validator.enable = False
    config.pose_dropout = 0.0
    config.noise_scheduler_steps = 2
    config.no_noise_conditioning_frames_prob = 0.0
    config.fake_timesteps_prob = 0.0
    config.debug.seed = 0
    return config


def main() -> None:
    from torchtitan.observability import structured_logger as sl
    from torchtitan.tools.logging import init_logger

    init_logger()
    _set_single_process_env()
    config = _stock_trainer_mock_config()
    sl.init_structured_logger(
        source="worldmodel_trainer_mock",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    trainer: WorldModelTrainer | None = None
    try:
        trainer = config.build()
        trainer.train()
        print({"step": trainer.step, "ntokens_seen": trainer.ntokens_seen})
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
