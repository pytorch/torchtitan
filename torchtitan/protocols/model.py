# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self, TYPE_CHECKING

import torch

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims

from .module import Module

if TYPE_CHECKING:
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.distributed.activation_checkpoint import (
        ActivationCheckpointingConfig,
    )

    from .state_dict_adapter import BaseStateDictAdapter


class BaseModel(Module, ABC):
    """Base class for all model classes.

    Models inherit from BaseModel (which is Module = nn.Module + Configurable).
    Each model defines a nested Config(BaseModel.Config) with model hyperparameters.
    The model is constructed via ``config.build()``.

    ``init_states`` (from Module) auto-recurses; override only for custom
    ordering (e.g., weight tying before init).
    """

    def init_weights(self, **kwargs) -> None:
        """Backward-compatible alias for ``init_states``.

        External tools (e.g., AutoParallel) wrap ``init_weights`` with
        DTensor-aware interception. This alias ensures they can find it.
        """
        # TODO: remove this once autoparallel has wrap_init_states
        buffer_device = kwargs.get("buffer_device")
        self.init_states(buffer_device=buffer_device)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        """Prepare the forward inputs from a dataloader batch.

        Models driven by the standard trainer/validator forward path implement
        this to build any attention masks, apply context-parallel sharding and
        SPMD annotation as needed, and split ``input``/``labels`` out of the
        batch. ``input_dict`` is the batch with ``labels`` folded in; return
        ``(inputs, labels, extra_kwargs)``. Models with aligned multi-output
        objectives may return tuples of input and label tensors.

        Additional keyword arguments may provide model-specific preprocessing
        dependencies that are owned outside the trainable model. The trainer
        calls this through the ``BaseModel`` interface. There is no meaningful default;
        every model used by the training engine must implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement preprocess_inputs()."
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Base config for all models.

        Subclasses define model-specific hyperparameters.
        """

        # TODO: This function violates encapsulation;
        # maybe replace it with config passes from outside.
        @abstractmethod
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            pass

        @abstractmethod
        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            pass

    state_dict_adapter_cls: ClassVar[type[BaseStateDictAdapter] | None] = None
    pipeline_first_stage_module_fqns: ClassVar[tuple[str, ...]] = ()
    pipeline_last_stage_module_fqns: ClassVar[tuple[str, ...]] = ()
    supports_pipeline_parallel: ClassVar[bool] = True

    def pipeline(self, **kwargs: Any) -> tuple[Any, list[BaseModel], bool, bool]:
        """Partition the model and build its pipeline schedule."""
        if not self.supports_pipeline_parallel:
            raise RuntimeError(
                f"{type(self).__name__} does not support pipeline parallelism."
            )

        from torchtitan.distributed.pipeline_parallel import (
            pipeline_llm,
            pipeline_with_first_last_stage_modules,
        )

        parallelism = kwargs["parallelism"]
        if parallelism.pipeline_parallel_module_fqns_per_model_part is None and (
            self.pipeline_first_stage_module_fqns
            or self.pipeline_last_stage_module_fqns
        ):
            return pipeline_with_first_last_stage_modules(
                self,
                first_stage_module_fqns=self.pipeline_first_stage_module_fqns,
                last_stage_module_fqns=self.pipeline_last_stage_module_fqns,
                **kwargs,
            )
        return pipeline_llm(self, **kwargs)

    def parallelize(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig | None,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ) -> Self:
        """Apply the ordered model-level parallelization lifecycle."""
        from torchtitan.distributed.utils import get_spmd_context

        with get_spmd_context(parallel_dims=parallel_dims):
            self._parallelize(parallel_dims)
            if ac_config is not None:
                ac_config.build(dump_folder=dump_folder).apply(self)
            if compile_config is not None and "model" in compile_config.components:
                from torchtitan.distributed.compile import apply_compile

                apply_compile(
                    self,
                    compile_config=compile_config,
                    parallel_dims=parallel_dims,
                )
            if not skip_dp:
                self._apply_fsdp(
                    parallel_dims=parallel_dims,
                    training=training,
                    parallelism=parallelism,
                )
        return self

    @abstractmethod
    def _apply_fsdp(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        pass

    @classmethod
    def _register_optimizer_hooks(
        cls,
        optimizers: OptimizersContainer,
        model_parts: list[BaseModel],
        parallel_dims: ParallelDims,
    ) -> None:
        del optimizers, model_parts, parallel_dims
