# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims

from .module import Module


class ModelConfigConverter(Configurable):
    """Base class for converters that transform the model config tree.

    Subclasses implement ``convert()`` to modify configs before model build
    (e.g. quantization, LoRA).  Converters may return a replacement root
    config when the transform needs to wrap the model config itself.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def convert(self, model_config: Module.Config) -> Module.Config:
        raise NotImplementedError


class BaseModel(Module):
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
        device: torch.device,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Minimal default input pipeline: copy, CP-shard, count tokens, return.

        Used by simple models (e.g. ``FluxModel``) that build no attention masks
        and declare no per-input SPMD layout. Models that need masks or an SPMD
        input layout override this method fully (no ``super()`` call).

        ``input_dict`` is the batch with ``labels`` folded in. Returns
        ``(inputs, labels, extra_kwargs, local_ntokens)``.

        TODO(return-type): the 4th return value (local token count) is a transitional
        workaround. The ``full_dtensor`` backend wraps ``labels`` in a DTensor
        whose ``.numel()`` reports the GLOBAL count, so the trainer cannot count
        from the returned labels. When the DTensor path is removed, drop this
        element and revert to a 3-tuple, letting the trainer do
        ``self.ntokens_seen += labels.numel()`` on the returned (plain,
        CP-sharded) labels.
        """
        # Imported function-locally to avoid a circular import
        # (context_parallel.api -> models.common -> decoder -> protocols.model).
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                None,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        local_ntokens = batch["labels"].numel()
        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch, local_ntokens

    def verify_module_protocol(self) -> None:
        """Verify all submodules satisfy the ``Module`` protocol.

        Catches non-``Module`` submodules early with a clear error message,
        preventing obscure failures when the ``Module`` protocol is being
        used later.

        Override in models where some internal ``nn.Module`` submodules
        cannot conform to the ``Module`` protocol.
        """
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
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
