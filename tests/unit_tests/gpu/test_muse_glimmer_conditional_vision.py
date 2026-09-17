# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.models.muse_glimmer import model_registry, parallelize_muse_glimmer
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel
from torchtitan.trainer import Trainer


pytestmark = pytest.mark.multi_gpu


def _build_input(
    *,
    image_active: bool,
    patch_dim: int,
    device_type: str,
) -> dict[str, object]:
    tokens_T = torch.arange(1, 9, device=device_type)
    labels_T = tokens_T.clone()
    batch: dict[str, object] = {
        "input": tokens_T,
        "labels": labels_T,
        "num_valid_tokens": int(labels_T.numel()),
        "positions": torch.arange(8, device=device_type),
    }
    if image_active:
        tokens_T[0] = 2004
        labels_T[0] = -100
        batch["num_valid_tokens"] = int(labels_T.numel() - 1)
        batch.update(
            pixel_values=torch.randn(
                4,
                patch_dim,
                device=device_type,
            ),
            grid_thw=torch.tensor(
                [[1, 2, 2]],
                device=device_type,
                dtype=torch.int64,
            ),
            special_tokens={"image_id": 2004},
        )
    return batch


def _build_trainer(
    *,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
    model_parts: list[nn.Module],
    optimizers: OptimizersContainer,
    loss_fn: CrossEntropyLoss,
    gradient_accumulation_steps: int,
    num_pp_microbatches: int,
) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        training=training,
        parallelism=parallelism,
    )
    trainer.parallel_dims = parallel_dims
    trainer.model_parts = model_parts
    trainer.optimizers = optimizers
    trainer.loss_fn = loss_fn
    trainer.gradient_accumulation_steps = gradient_accumulation_steps
    trainer.num_pp_microbatches = num_pp_microbatches
    trainer.device = torch.device(  # pyrefly: ignore[read-only]
        torch.cuda.current_device()
    )
    trainer.train_context = get_spmd_context(parallel_dims=parallel_dims)
    trainer.lr_schedulers = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        get_metrics=lambda: {},
        step=lambda: None,
    )
    trainer.metrics_processor = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        should_log=lambda _step: False,
    )
    trainer.checkpointer = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        maybe_wait_for_staging=lambda: None,
    )
    trainer.dataloader = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        max_num_documents=1,
    )
    trainer.sdc_replayer = None
    trainer.step = 0
    trainer.ntokens_seen = 0
    return trainer


def _vision_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    modules = (
        model.vision_encoder,  # pyrefly: ignore[missing-attribute]
        model.vision_adapter,  # pyrefly: ignore[missing-attribute]
        model.vision_projection,  # pyrefly: ignore[missing-attribute]
        model.perception_emb_norm,  # pyrefly: ignore[missing-attribute]
    )
    assert all(module is not None for module in modules)
    return tuple(
        parameter
        for module in modules
        for parameter in module.parameters()  # pyrefly: ignore[missing-attribute]
    )


class TestMuseGlimmerDPGradientAccumulation(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_uneven_image_presence_across_accumulation_groups(self) -> None:
        parallelism = ParallelismConfig(data_parallel_shard_degree=2)
        parallel_dims = ParallelDims.from_config(parallelism, self.world_size)
        parallel_dims.build_mesh()
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model_config = cast(MuseGlimmerModel.Config, model_spec.model)
        assert model_config.vision_encoder is not None
        model_config.vision_encoder.num_layers = 0
        model_config.layers = model_config.layers[:2]
        model_config.update_from_config(config=SimpleNamespace(parallelism=parallelism))

        with torch.device("meta"):
            model = model_config.build()
        training = TrainingConfig(disable_cuda_graphs=True, max_context_length=8)
        compile_config = CompileConfig()
        model = parallelize_muse_glimmer(
            model,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=None,
            dump_folder="",
        )
        model.to_empty(device=self.device_type)
        with torch.no_grad():
            torch.manual_seed(0)
            model.init_weights()
        model.train()

        optimizer_config = default_adamw(lr=0.1)
        optimizers = optimizer_config.build(model_parts=[model])
        assert model_spec.post_optimizer_build_fn is not None
        model_spec.post_optimizer_build_fn(optimizers, [model], parallel_dims)
        loss_fn = CrossEntropyLoss.Config(
            global_vocab_size=model_config.vocab_size
        ).build(compile_config=compile_config)
        trainer = _build_trainer(
            training=training,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            model_parts=[model],
            optimizers=optimizers,
            loss_fn=loss_fn,
            gradient_accumulation_steps=2,
            num_pp_microbatches=1,
        )
        trainer.fwd_bwd_fn = trainer._forward_backward_body
        assert model.vision_encoder is not None
        tracked_parameter = model.vision_encoder.conv1_linear.weight
        vision_parameters = _vision_parameters(model)
        assert model.tok_embeddings is not None
        text_parameter = model.tok_embeddings.embedding.weight
        patch_dim = model.vision_encoder.conv1_linear.in_features
        batch_rank = parallel_dims.get_mesh("batch").get_local_rank()
        self.assertEqual(len(optimizers.optimizers), 1)
        optimizer = optimizers.optimizers[0]

        for active_dp_rank, activity_order, expect_vision_update in (
            (0, (True, False), True),
            (1, (False, True), True),
            (None, (False, False), False),
        ):
            batches = (
                _build_input(
                    image_active=(
                        microbatch_active
                        and active_dp_rank is not None
                        and batch_rank == active_dp_rank
                    ),
                    patch_dim=patch_dim,
                    device_type=self.device_type,
                )
                for microbatch_active in activity_order
            )

            vision_parameter_before = tracked_parameter.to_local().detach().clone()
            text_parameter_before = text_parameter.to_local().detach().clone()
            vision_state_before = {
                parameter: {
                    name: value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in optimizer.state[parameter].items()
                }
                for parameter in vision_parameters
            }
            text_step_before = optimizer.state[text_parameter].get("step")
            if text_step_before is not None:
                text_step_before = text_step_before.item()
            trainer.train_step(iter(batches))
            if expect_vision_update:
                self.assertFalse(
                    torch.equal(tracked_parameter.to_local(), vision_parameter_before)
                )
            else:
                self.assertEqual(tracked_parameter.to_local(), vision_parameter_before)
                for parameter in vision_parameters:
                    self.assertIsNone(parameter.grad)
                    self.assertEqual(
                        optimizer.state[parameter]["step"].item(),
                        vision_state_before[parameter]["step"].item(),
                    )
                    self.assertEqual(
                        optimizer.state[parameter], vision_state_before[parameter]
                    )
            self.assertFalse(
                torch.equal(text_parameter.to_local(), text_parameter_before)
            )
            if text_step_before is not None:
                self.assertEqual(
                    optimizer.state[text_parameter]["step"].item(),
                    text_step_before + 1,
                )
            self.assertFalse(model._consume_vision_activity())


class TestMuseGlimmerDPPipeline(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_uneven_image_presence_across_pipeline_microbatches(self) -> None:
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            num_pp_microbatches=2,
            pipeline_parallel_schedule="1F1B",
        )
        parallel_dims = ParallelDims.from_config(parallelism, self.world_size)
        parallel_dims.build_mesh()
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model_config = cast(MuseGlimmerModel.Config, model_spec.model)
        assert model_config.vision_encoder is not None
        model_config.vision_encoder.num_layers = 0
        model_config.layers = model_config.layers[:2]
        patch_dim = model_config.vision_encoder.conv1.in_features
        model_config.update_from_config(config=SimpleNamespace(parallelism=parallelism))

        with torch.device("meta"):
            model = model_config.build()
        training = TrainingConfig(disable_cuda_graphs=True, max_context_length=8)
        compile_config = CompileConfig()
        loss_fn = CrossEntropyLoss.Config(
            global_vocab_size=model_config.vocab_size
        ).build(compile_config=compile_config)
        assert model_spec.pipelining_fn is not None
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = model_spec.pipelining_fn(
            model,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=None,
            dump_folder="",
            device=torch.device(self.device_type),
            model_config=model_config,
            parallelize_fn=model_spec.parallelize_fn,
            loss_fn=loss_fn,
        )
        self.assertEqual(len(model_parts), 1)
        model_part = model_parts[0]
        model_part.to_empty(device=self.device_type)
        with torch.no_grad():
            torch.manual_seed(0)
            model_part.init_weights()
        model_part.train()

        optimizer_config = default_adamw(lr=0.1)
        optimizer_config.implementation = "for-loop"
        optimizers = optimizer_config.build(model_parts=model_parts)
        assert model_spec.post_optimizer_build_fn is not None
        model_spec.post_optimizer_build_fn(optimizers, model_parts, parallel_dims)
        trainer = _build_trainer(
            training=training,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            model_parts=model_parts,
            optimizers=optimizers,
            loss_fn=loss_fn,
            gradient_accumulation_steps=1,
            num_pp_microbatches=2,
        )
        trainer.pp_schedule = pp_schedule
        trainer.pp_has_first_stage = has_first_stage
        trainer.pp_has_last_stage = has_last_stage
        trainer._pp_loss_sentinel_on_non_last_stage = torch.full(
            (1,), -1.0, device=self.device_type
        )
        trainer.fwd_bwd_fn = trainer._pp_forward_backward_body
        batch_rank = parallel_dims.get_mesh("batch").get_local_rank()
        first_vision_parameter_before: torch.Tensor | None = None
        if has_first_stage:
            assert model_part.vision_encoder is not None
            tracked_parameter = model_part.vision_encoder.conv1_linear.weight
            first_vision_parameter_before = (
                tracked_parameter.to_local().detach().clone()
            )
        trainer.train_step(
            iter(
                _build_input(
                    image_active=batch_rank == 0 and microbatch_active,
                    patch_dim=patch_dim,
                    device_type=self.device_type,
                )
                for microbatch_active in (True, False)
            )
        )
        if has_first_stage:
            assert first_vision_parameter_before is not None
            assert model_part.vision_encoder is not None
            tracked_parameter = model_part.vision_encoder.conv1_linear.weight
            self.assertFalse(
                torch.equal(tracked_parameter.to_local(), first_vision_parameter_before)
            )
            self.assertFalse(model_part._consume_vision_activity())

        empty_vision_parameter_before: torch.Tensor | None = None
        text_parameter_before: torch.Tensor | None = None
        vision_state_before: dict[nn.Parameter, dict[str, object]] | None = None
        text_step_before: float | None = None
        if has_first_stage:
            self.assertEqual(len(optimizers.optimizers), 1)
            optimizer = optimizers.optimizers[0]
            assert model_part.vision_encoder is not None
            tracked_parameter = model_part.vision_encoder.conv1_linear.weight
            vision_parameters = _vision_parameters(model_part)
            assert model_part.tok_embeddings is not None
            text_parameter = model_part.tok_embeddings.embedding.weight
            empty_vision_parameter_before = (
                tracked_parameter.to_local().detach().clone()
            )
            text_parameter_before = text_parameter.to_local().detach().clone()
            vision_state_before = {
                parameter: {
                    name: value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in optimizer.state[parameter].items()
                }
                for parameter in vision_parameters
            }
            text_step_before = optimizer.state[text_parameter]["step"].item()
        trainer.train_step(
            iter(
                _build_input(
                    image_active=False,
                    patch_dim=patch_dim,
                    device_type=self.device_type,
                )
                for _ in range(2)
            )
        )
        if has_first_stage:
            assert empty_vision_parameter_before is not None
            assert text_parameter_before is not None
            assert vision_state_before is not None
            assert text_step_before is not None
            self.assertEqual(len(optimizers.optimizers), 1)
            optimizer = optimizers.optimizers[0]
            assert model_part.vision_encoder is not None
            tracked_parameter = model_part.vision_encoder.conv1_linear.weight
            vision_parameters = _vision_parameters(model_part)
            assert model_part.tok_embeddings is not None
            text_parameter = model_part.tok_embeddings.embedding.weight
            self.assertEqual(
                tracked_parameter.to_local(), empty_vision_parameter_before
            )
            for parameter in vision_parameters:
                self.assertIsNone(parameter.grad)
                vision_step_before = vision_state_before[parameter]["step"]
                assert isinstance(vision_step_before, torch.Tensor)
                self.assertEqual(
                    optimizer.state[parameter]["step"].item(),
                    vision_step_before.item(),
                )
                self.assertEqual(
                    optimizer.state[parameter], vision_state_before[parameter]
                )
            self.assertFalse(
                torch.equal(text_parameter.to_local(), text_parameter_before)
            )
            self.assertEqual(
                optimizer.state[text_parameter]["step"].item(),
                text_step_before + 1,
            )
            self.assertFalse(model_part._consume_vision_activity())


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
