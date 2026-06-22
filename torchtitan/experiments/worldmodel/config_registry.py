from __future__ import annotations

import os
from collections.abc import Callable

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, register_float8_precompute_scale_hook
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from xx.common.compressor_helpers import COMPRESSOR_STATS
from xx.datasets.helpers import BASE_DIR_GT_10M, DEFAULT_10M_TRAIN_LIST

from .dataset import WorldModelDataLoader
from .loss import WorldModelLoss
from .model import TransformerConfig, WorldModel, parallelize_worldmodel
from .tokenizer import WorldModelTokenizer
from .trainer import WorldModelTrainer, WorldModelValidator


COMPRESSOR_MODEL = "c04337f8-b83f-4e34-b07a-5f7396978d67"
FEATURE_DIR = "http://data-ssd.comma.life/vae_model_features"
LATENT_CHANNELS = 32
LATENT_SIZE = (16, 32)
IMAGE_SIZE = (128, 256)

WORLD_MODEL_FLOAT8_FILTER_FQNS = [
    "x_embedder",
    "augments_pos_ref_augment_embedder",
    "ref_augment_from_augments_euler_embedder",
    "pose_mask_embedder",
    "t_embedder",
    "fidx_embedder",
    "final_layer",
    "plan_head",
]


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = _worldmodel_configs()[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="worldmodel",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_worldmodel,
        pipelining_fn=None,
        post_optimizer_build_fn=register_float8_precompute_scale_hook,
        state_dict_adapter=None,
    )


def worldmodel() -> WorldModelTrainer.Config:
    local_batch_size = 16
    validation_freq = 512
    steps = validation_freq * 30
    validation_steps = 8
    compile_config = CompileConfig(enable=True, components=["model", "loss"])
    optimizer = default_adamw(lr=2e-4, weight_decay=1e-2)
    optimizer.implementation = "fused_opt_states_bf16"
    local_world_size, world_size, num_nodes = _world_sizes()
    checkpoint_base_folder = _reporterv2_checkpoint_base_folder()

    return WorldModelTrainer.Config(
        hf_assets_path=".",
        dump_folder=checkpoint_base_folder or "./outputs",
        loss=WorldModelLoss.Config(plan_loss_weight=0.1),
        tokenizer=WorldModelTokenizer.Config(
            compressor_model=COMPRESSOR_MODEL,
            compressor_in_channels="auto",
        ),
        model_spec=model_registry(
            "base",
            converters=[
                _blocks_only_float8(
                    model_compile_enabled=compile_config.enable and "model" in compile_config.components,
                )
            ],
        ),
        dataloader=_dataloader_config(split="train"),
        optimizer=optimizer,
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2 * validation_freq,
            total_steps=steps,
            decay_ratio=0.1,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            global_batch_size=local_batch_size * world_size * 2,
            seq_len=1,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=compile_config,
        metrics=MetricsProcessor.Config(
            log_freq=16,
            enable_reporterv2=True,
            save_freq=validation_freq,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder=os.getenv("REPORTERV2_TRAINING_ID") or "checkpoint",
            interval=validation_freq * 5,
            async_mode="async",
            keep_latest_k=0,
            enable_first_step_checkpoint=True,
            last_save_model_only=False,
            checkpoint_id_format="",
            exclude_from_loading=[
                "optimizer",
                "lr_scheduler",
                "dataloader",
                "train_state",
            ],
        ),
        validator=WorldModelValidator.Config(
            enable=True,
            freq=validation_freq,
            steps=validation_steps,
            dataloader=_dataloader_config(split="val", fill_once=True),
            pose_dropout=0.0,
            noise_scheduler_steps=10,
            no_noise_conditioning_frames_prob=0.0,
            fake_timesteps_prob=0.0,
        ),
        pose_dropout=0.1,
        noise_scheduler_steps=10,
        no_noise_conditioning_frames_prob=0.5,
        fake_timesteps_prob=0.5,
        debug=DebugConfig(seed=0),
    )


def _worldmodel_configs() -> dict[str, Callable[[], WorldModel.Config]]:
    return {
        "base": _model_config,
        "debugmodel": _debug_model_config,
    }


def _debug_model_config() -> WorldModel.Config:
    return _model_config(
        input_size=(15, 4, 4),
        patch_size=(1, 2, 2),
        hidden=64,
        heads=4,
        layers=1,
        plan_layers=1,
        mlp_multiple_of=16,
        attention_impl="FLEX",
    )


def _model_config(
    *,
    input_size: tuple[int, int, int] = (15, *LATENT_SIZE),
    patch_size: tuple[int, int, int] = (1, 2, 2),
    hidden: int = 2304,
    heads: int = 36,
    layers: int = 56,
    plan_layers: int = 4,
    mlp_multiple_of: int = 256,
    attention_impl: str = "FLEX",
    attention_mask: str = "NONE",
    norm: str = "RMSNorm",
) -> WorldModel.Config:
    stats = COMPRESSOR_STATS[COMPRESSOR_MODEL]
    return WorldModel.Config(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        pose_size=6,
        time_factor=1.0,
        compressor_mean=stats["mean"],
        compressor_std=stats["std"],
        transformer=TransformerConfig(
            n_layer=layers,
            n_embd=hidden,
            n_head=heads,
            act="GELU",
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            biased_linears=True,
            prenorm=False,
            qk_norm=True,
            mlp_mult=4,
            mlp_multiple_of=mlp_multiple_of,
            attention_mask=attention_mask,
            norm=norm,
            attention_impl=attention_impl,
        ),
        plan_head=TransformerConfig(
            n_layer=plan_layers,
            n_embd=hidden,
            n_head=heads,
            act="GELU",
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            biased_linears=True,
            prenorm=True,
            qk_norm=False,
            mlp_mult=2,
            mlp_multiple_of=1,
            attention_mask="LAST_FRAME_CAUSAL",
            norm="LayerNorm",
            attention_impl=attention_impl,
        ),
    )


def _dataloader_config(
    *,
    split: str,
    dataset: str = DEFAULT_10M_TRAIN_LIST,
    dataset_path: str | None = None,
    shuffle_size: int = 50_000,
    min_mixing: float = 0.5,
    num_writers: int = 2,
    num_readers: int = 4,
    fill_once: bool = False,
    base_dir: str = BASE_DIR_GT_10M,
    feature_dir: str | None = None,
    compressor_model: str = COMPRESSOR_MODEL,
    in_channels: int = LATENT_CHANNELS,
    latent_size: tuple[int, int] = LATENT_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    context_size_frames: int = 10,
    future_size_frames: int = 5,
    max_future_frames: int = 50,
    inference_conditioning_frames: int = 14,
    fps: int = 5,
    train_skip: int = 40,
    val_skip: int = 800,
    nan_engaged_plans: bool = False,
    limit: int | None = None,
    mock_data: bool = False,
    mock_segment_batch_size: int = 8,
) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        shuffle_size=shuffle_size,
        min_mixing=min_mixing,
        num_writers=num_writers,
        num_readers=num_readers,
        fill_once=fill_once,
        base_dir=base_dir,
        feature_dir=feature_dir,
        compressor_model=compressor_model,
        in_channels=in_channels,
        latent_size=latent_size,
        image_size=image_size,
        context_size_frames=context_size_frames,
        future_size_frames=future_size_frames,
        max_future_frames=max_future_frames,
        inference_conditioning_frames=inference_conditioning_frames,
        fps=fps,
        train_skip=train_skip,
        val_skip=val_skip,
        nan_engaged_plans=nan_engaged_plans,
        limit=limit,
        mock_data=mock_data,
        mock_segment_batch_size=mock_segment_batch_size,
    )


def _blocks_only_float8(*, model_compile_enabled: bool, emulate: bool = False) -> Float8LinearConverter.Config:
    return Float8LinearConverter.Config(
        recipe_name="tensorwise",
        filter_fqns=WORLD_MODEL_FLOAT8_FILTER_FQNS,
        emulate=emulate,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        model_compile_enabled=model_compile_enabled,
    )


def _world_sizes() -> tuple[int, int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(max(1, world_size // max(1, local_world_size)))))
    return local_world_size, world_size, num_nodes


def _reporterv2_checkpoint_base_folder() -> str:
    host = os.getenv("REPORTERV2_HOST")
    return f"{host.rstrip('/')}/checkpoint" if host else ""


def main() -> None:
    spec = model_registry("debugmodel")
    model = spec.model.build()
    config = model.config
    nparams = sum(param.numel() for param in model.parameters())
    head_dim = config.transformer.n_embd // config.transformer.n_head
    print(
        {
            "flavor": spec.flavor,
            "input_size": config.input_size,
            "num_patches": config.num_patches,
            "hidden": config.transformer.n_embd,
            "heads": config.transformer.n_head,
            "head_dim": head_dim,
            "layers": config.transformer.n_layer,
            "parameters": nparams,
        }
    )


if __name__ == "__main__":
    main()
