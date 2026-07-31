# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable

__all__ = ["EMAOptimizersContainer"]


class _EMAParamOptimizer(Optimizer):
    """Holds ``state[p]["ema_params"]`` per parameter for one model part.

    Never step()-ed; reuses ``Optimizer``'s per-param state dict plus the
    FQN-flattening DCP machinery in ``checkpoint_utils.py`` instead of a
    bespoke DTensor state-dict format. Always built over the real params
    even when EMA is disabled -- ``enable`` only controls whether the EMA
    tensor is allocated.
    """

    def __init__(
        self, named_params: list[tuple[str, nn.Parameter]], *, enable: bool
    ) -> None:
        params = [p for _, p in named_params]
        param_names = [canonical_fqn(name) for name, _ in named_params]
        super().__init__([{"params": params, "param_names": param_names}], {})
        if enable:
            for p in params:
                self.state[p]["ema_params"] = p.detach().clone()

    def step(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call "
            "EMAOptimizersContainer.step(current_step) instead."
        )


class EMAOptimizersContainer(OptimizersContainer):
    """Pseudo-optimizer maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizers`` or
    ``LRSchedulersContainer`` -- it's a sibling object, always built and
    wired into ``CheckpointManager`` and a ``register_step_post_hook``;
    ``enable`` decides whether that amounts to anything.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether EMA tracking is active. Always built regardless (so
        trainer.py needs no conditional wiring), but a no-op when False."""

        decay: float | None = None
        """Fixed decay per firing: ema_params = decay * ema_params +
        (1 - decay) * param. If None (default), computed dynamically from
        half_life_fraction instead."""

        half_life_fraction: float = 0.05
        """Used when decay is None: decay = 2 ** (-1 / (half_life_fraction *
        num_updates)). Keeps roughly the most recent half_life_fraction
        share of updates dominant. 0.05 matches the common
        decay = 2 ** (-20 / t) rule of thumb."""

        start_step: int = 0
        """First Trainer.step at which EMA tracking begins. Decoupled from
        the LR scheduler's WSD phases."""

        step_bias: int = 0
        """Manual offset added when computing num_updates, for deliberately
        renumbering Trainer.step (e.g. a new training phase) without
        resetting EMA aging. A normal resume needs no bias -- current_step
        already continues correctly on its own."""

        update_every_n_steps: int = 1
        """Only fire the EMA update every N real optimizer steps."""

        offload_to_cpu: bool = False
        """Keep EMA weights in pinned CPU memory, updated via an async
        side-stream H2D/D2H pipeline, for GH200's NVLink-C2C interconnect."""

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        self.enable = config.enable
        self.decay = config.decay
        self.half_life_fraction = config.half_life_fraction
        self.start_step = config.start_step
        self.step_bias = config.step_bias
        self.update_every_n_steps = config.update_every_n_steps
        self.offload_to_cpu = config.offload_to_cpu
        self.model_parts = model_parts

        self.optimizers: list[_EMAParamOptimizer] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            named_params = [
                (name, p) for name, p in model.named_parameters() if p.requires_grad
            ]
            self.optimizers.append(_EMAParamOptimizer(named_params, enable=self.enable))
            all_params.extend(p for _, p in named_params)
        self._validate_params(all_params)
        self._post_init(all_params)

        self._offload_stream: torch.cuda.Stream | None = None
        self._offload_scratch: dict[int, list[torch.Tensor]] = {}
        self._pending_event: torch.cuda.Event | None = None
        if self.enable and self.offload_to_cpu:
            self._init_cpu_offload()

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    def step(self, current_step: int) -> None:
        """Call directly with the trainer's global step -- never merged into
        Trainer.optimizers, so there's no closure/zero-arg step() to honor."""
        if not self.enable or current_step < self.start_step:
            return
        elapsed = current_step - self.start_step
        if elapsed % self.update_every_n_steps != 0:
            return
        # num_updates counts firings, not raw steps (equal only when
        # update_every_n_steps == 1) -- still stateless, a pure function of
        # current_step. Clamped to >= 1 for the first firing.
        num_updates = max((elapsed + self.step_bias) // self.update_every_n_steps, 1)
        self._update(num_updates)

    def _decay_at(self, num_updates: int) -> float:
        if self.decay is not None:
            return self.decay
        return 2.0 ** (-1.0 / (self.half_life_fraction * num_updates))

    def _update(self, num_updates: int) -> None:
        decay = self._decay_at(num_updates)
        for part_idx, (ema_opt, model) in enumerate(
            zip(self.optimizers, self.model_parts)
        ):
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                continue
            ema_params = [ema_opt.state[p]["ema_params"] for p in params]
            if self.offload_to_cpu:
                # ema_params are pinned local-shard CPU tensors; localize
                # params too so the foreach ops never mix DTensor with Tensor.
                local_params = [self._local_view(p) for p in params]
                self._update_offloaded(part_idx, local_params, ema_params, decay)
            elif torch.is_floating_point(ema_params[0]) or torch.is_complex(
                ema_params[0]
            ):
                torch._foreach_lerp_(ema_params, params, 1.0 - decay)
            else:
                for e, p in zip(ema_params, params):
                    e.copy_(e * decay + p * (1.0 - decay))

    # --- CPU offload path (GH200-optimized: async side-stream, pinned memory) ---

    @staticmethod
    def _local_view(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if isinstance(t, DTensor) else t

    def _init_cpu_offload(self) -> None:
        self._offload_stream = torch.cuda.Stream()
        for ema_opt in self.optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"] = self._pin_local(param_state["ema_params"])

    def _pin_local(self, tensor: torch.Tensor) -> torch.Tensor:
        """DTensor has no pin_memory() dispatch support (NYI:
        aten._pin_memory.default), so pin just the local shard."""
        return self._local_view(tensor).cpu().pin_memory()

    def _materialize_dtensor(
        self, p: torch.Tensor, local: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of ``_pin_local``: move the local shard back onto the
        accelerator and rewrap it as a DTensor matching ``p``'s own
        sharding, only for the duration of a checkpoint save/load -- this is
        what DCP needs to (re)shard EMA state correctly across world sizes.
        ``p`` is still the live DTensor param, so its spec is read directly
        rather than cached. No collective communication:
        ``from_local(run_check=False)`` only communicates to reconcile a
        ``Replicate()`` placement, which FSDP2 params never use.
        """
        if not isinstance(p, DTensor):
            return local
        local_gpu = local.to(p.device, non_blocking=True)
        return DTensor.from_local(
            local_gpu,
            device_mesh=p.device_mesh,
            placements=p.placements,
            run_check=False,
        )

    def _get_scratch(self, key: int, params: list[torch.Tensor]) -> list[torch.Tensor]:
        scratch = self._offload_scratch.get(key)
        if scratch is None:
            scratch = [torch.empty_like(p) for p in params]
            self._offload_scratch[key] = scratch
        return scratch

    def _maybe_wait_pending(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
            self._pending_event = None

    def _update_offloaded(
        self,
        scratch_key: int,
        params: list[torch.Tensor],
        ema_params: list[torch.Tensor],
        decay: float,
    ) -> None:
        self._maybe_wait_pending()
        scratch = self._get_scratch(scratch_key, params)
        stream = self._offload_stream
        assert stream is not None
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch._foreach_copy_(scratch, ema_params, non_blocking=True)  # H2D
            torch._foreach_lerp_(scratch, params, 1.0 - decay)
            torch._foreach_copy_(ema_params, scratch, non_blocking=True)  # D2H
            self._pending_event = torch.cuda.Event()
            self._pending_event.record(stream)
        # Waited on lazily -- next call, or at state_dict() (checkpoint save).

    # --- checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        if not self.enable:
            return {}
        if not self.offload_to_cpu:
            return super().state_dict()
        # Materialize real DTensors for DCP, call through, then restore the
        # pinned-CPU steady state so offload savings only lapse briefly.
        self._maybe_wait_pending()
        originals: dict[int, torch.Tensor] = {}
        for ema_opt in self.optimizers:
            for p, param_state in ema_opt.state.items():
                originals[id(p)] = param_state["ema_params"]
                param_state["ema_params"] = self._materialize_dtensor(
                    p, param_state["ema_params"]
                )
        result = super().state_dict()
        for ema_opt in self.optimizers:
            for p, param_state in ema_opt.state.items():
                param_state["ema_params"] = originals[id(p)]
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.enable:
            return
        if not state_dict:
            # Checkpoint had no EMA data (disabled at save time, or predates
            # this feature) -- cold-start from the just-loaded model weights.
            for ema_opt, model in zip(self.optimizers, self.model_parts):
                for p in (p for p in model.parameters() if p.requires_grad):
                    source = p.detach()
                    if self.offload_to_cpu:
                        source = self._local_view(source)
                    ema_opt.state[p]["ema_params"].copy_(source)
            return
        # DCP calls our state_dict() above to build its load template, so it
        # already receives real DTensors here too -- re-pin them afterward.
        super().load_state_dict(state_dict)
        if self.offload_to_cpu:
            for ema_opt in self.optimizers:
                for param_state in ema_opt.state.values():
                    param_state["ema_params"] = self._pin_local(
                        param_state["ema_params"]
                    )
