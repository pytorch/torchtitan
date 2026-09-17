# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable

from .optimizer import OptimizersContainer

__all__ = ["EMA"]

logger = logging.getLogger(__name__)


class _EMAParamOptimizer(Optimizer):
    """Holds ``state[t]["ema_params"]`` per tensor (parameter or buffer) for
    one model part.

    Never step()-ed; reuses ``Optimizer``'s per-tensor state dict plus the
    FQN-flattening DCP machinery in ``checkpointer/utils.py`` instead of a
    bespoke DTensor state-dict format. Also used for buffer EMA (e.g. MoE's
    ``expert_bias_E``), which is why tensors need not be ``nn.Parameter``s.
    """

    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        tensors = [t for _, t in named_tensors]
        names = [canonical_fqn(name) for name, _ in named_tensors]
        super().__init__([{"params": tensors, "param_names": names}], {})
        for t in tensors:
            self.state[t]["ema_params"] = t.detach().clone()

    def step(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call "
            "EMA.step(current_step) instead."
        )


class EMA(OptimizersContainer):
    """Pseudo-optimizer maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizers`` or
    ``LRSchedulersContainer`` -- it's a sibling object, only built (via
    ``Trainer.Config.ema``) when the user opts in, and stepped explicitly
    from ``train_step()``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
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
        """Manual offset added to the firing count when computing num_updates,
        for deliberately renumbering a new training phase (e.g. a restart that
        resets Trainer.step) without resetting EMA aging. A normal resume
        needs no bias -- the firing count already continues correctly on its
        own. Measured in EMA firings, not raw steps: with
        update_every_n_steps=4, step_bias=2 means "pretend 2 firings already
        happened", not "pretend 2 steps already happened"."""

        update_every_n_steps: int = 1
        """Only fire the EMA update every N real optimizer steps."""

        offload_to_cpu: bool = False
        """Keep EMA weights in pinned CPU memory, updated via an async
        side-stream H2D/D2H pipeline, for GH200's NVLink-C2C interconnect."""

        buffer_patterns: list[str] = field(default_factory=list)
        """Regex patterns (re.search, matched against buffer FQNs from
        model.named_buffers()) selecting which buffers also get an EMA
        tracked alongside trainable parameters -- e.g. MoE's expert_bias_E (a
        register_buffer updated by a non-gradient load-balancing heuristic,
        otherwise silently excluded from any weight-averaging story). Folded
        into the same "ema" checkpoint key as parameters, since param and
        buffer FQNs never collide within a model. Empty (default): no
        buffers tracked, identical behavior to before this option existed."""

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        self.decay = config.decay
        self.half_life_fraction = config.half_life_fraction
        self.start_step = config.start_step
        self.step_bias = config.step_bias
        self.update_every_n_steps = config.update_every_n_steps
        self.offload_to_cpu = config.offload_to_cpu
        self.model_parts = model_parts

        self._param_optimizers: list[_EMAParamOptimizer] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            named_params = [
                (name, p) for name, p in model.named_parameters() if p.requires_grad
            ]
            self._param_optimizers.append(_EMAParamOptimizer(named_params))
            all_params.extend(p for _, p in named_params)
        self._validate_params(all_params)

        self._buffer_patterns = [re.compile(p) for p in config.buffer_patterns]
        self._buffer_optimizers: list[_EMAParamOptimizer] = []
        if self._buffer_patterns:
            total_matched = 0
            for model in model_parts:
                named_buffers = [
                    (name, b)
                    for name, b in model.named_buffers()
                    if any(p.search(name) for p in self._buffer_patterns)
                ]
                total_matched += len(named_buffers)
                self._buffer_optimizers.append(_EMAParamOptimizer(named_buffers))
            if total_matched == 0:
                logger.warning(
                    "EMA.Config.buffer_patterns=%s matched no buffers across any "
                    "model part -- buffer EMA is configured but silently tracking "
                    "nothing. Check the patterns for typos.",
                    config.buffer_patterns,
                )

        # OptimizersContainer.state_dict()/load_state_dict() (reused as-is --
        # see state_dict() below) iterate self.optimizers and merge each
        # one's FQN-keyed flat dict, so folding _buffer_optimizers in here is
        # what gives buffer EMA the same "ema" checkpoint key as parameters.
        self.optimizers: list[_EMAParamOptimizer] = (
            self._param_optimizers + self._buffer_optimizers
        )
        self._post_init(all_params)

        self._offload_stream: torch.cuda.Stream | None = None
        self._offload_scratch: dict[Any, list[torch.Tensor]] = {}
        self._pending_event: torch.cuda.Event | None = None
        if self.offload_to_cpu:
            self._init_cpu_offload()

        self._num_firings = 0

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    def step(self, current_step: int) -> None:
        """Call directly with the trainer's global step -- never merged into
        Trainer.optimizers, so there's no closure/zero-arg step() to honor."""
        if current_step < self.start_step:
            return
        elapsed = current_step - self.start_step
        if elapsed % self.update_every_n_steps != 0:
            return
        # num_updates is an explicit firing count (1, 2, 3, ...), not derived
        # from elapsed/update_every_n_steps -- deriving it that way previously
        # under- or over-counted whenever start_step > 0 (elapsed=0 is a real
        # firing there, unlike the default start_step=0) or step_bias wasn't a
        # multiple of update_every_n_steps (floor division silently dropped
        # the remainder).
        self._num_firings += 1
        num_updates = self._num_firings + self.step_bias
        self._update(num_updates)

    def _decay_at(self, num_updates: int) -> float:
        if self.decay is not None:
            return self.decay
        return 2.0 ** (-1.0 / (self.half_life_fraction * num_updates))

    # TODO: params/buffers are re-derived from the model on every firing
    # (model.parameters()/model.named_buffers() + regex matching for
    # buffers), even though the identical lists are already sitting in
    # ema_opt.param_groups[0]["params"] from construction -- EMA state is
    # already looked up by tensor identity, so identity (and thus this list)
    # is already assumed stable across the run. Reusing param_groups[0]
    # instead of recomputing would remove this per-step traversal/regex cost
    # with no behavior change.
    def _update(self, num_updates: int) -> None:
        decay = self._decay_at(num_updates)
        for part_idx, (ema_opt, model) in enumerate(
            zip(self._param_optimizers, self.model_parts)
        ):
            params = [p for p in model.parameters() if p.requires_grad]
            self._update_group(("param", part_idx), ema_opt, params, decay)
        for part_idx, (ema_opt, model) in enumerate(
            zip(self._buffer_optimizers, self.model_parts)
        ):
            buffers = [
                b
                for name, b in model.named_buffers()
                if any(p.search(name) for p in self._buffer_patterns)
            ]
            self._update_group(("buffer", part_idx), ema_opt, buffers, decay)

    def _update_group(
        self,
        scratch_key: Any,
        ema_opt: "_EMAParamOptimizer",
        tensors: list[torch.Tensor],
        decay: float,
    ) -> None:
        """Shared lerp/decay body for one model part's params or buffers."""
        if not tensors:
            return
        ema_params = [ema_opt.state[t]["ema_params"] for t in tensors]
        if self.offload_to_cpu:
            # ema_params are pinned local-shard CPU tensors; localize the
            # live tensors too so the foreach ops never mix DTensor with
            # Tensor.
            local_tensors = [self._local_view(t) for t in tensors]
            self._update_offloaded(scratch_key, local_tensors, ema_params, decay)
            return
        # buffer_patterns can match tensors of different dtypes within one
        # group (e.g. a float bias alongside an int counter); foreach ops
        # require a homogeneous dtype, so split by dtype-class before batching.
        floating = [
            (e, t)
            for e, t in zip(ema_params, tensors)
            if torch.is_floating_point(e) or torch.is_complex(e)
        ]
        other = [
            (e, t)
            for e, t in zip(ema_params, tensors)
            if not (torch.is_floating_point(e) or torch.is_complex(e))
        ]
        if floating:
            torch._foreach_lerp_(
                [e for e, _ in floating], [t for _, t in floating], 1.0 - decay
            )
        for e, t in other:
            e.copy_(e * decay + t * (1.0 - decay))

    # --- CPU offload path (GH200-optimized: async side-stream, pinned memory) ---

    @staticmethod
    def _local_view(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if isinstance(t, DTensor) else t

    def _init_cpu_offload(self) -> None:
        self._offload_stream = torch.cuda.Stream()
        for ema_opt in self.optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"] = self._pin_local(param_state["ema_params"])

    # TODO: DTensor doesn't support pin_memory() (NYI: aten._pin_memory.default),
    # so we pin the local shard only, then rewrap it as a DTensor around the
    # param's live spec at save/load time (_materialize_dtensor). If DTensor
    # gains native pin_memory() support this shard-unwrap/rewrap dance can be
    # dropped. There may be additional complexity around process-group
    # restore (PG membership can change across resumes/world-size changes)
    # that a native implementation would need to account for.
    def _pin_local(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._local_view(tensor).cpu().pin_memory()

    def _materialize_dtensor(
        self, p: torch.Tensor, local: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of ``_pin_local``: move the local shard back onto the
        accelerator and rewrap it as a DTensor matching ``p``'s own
        sharding, only for the duration of a checkpoint save/load -- this is
        what DCP needs to (re)shard EMA state correctly across world sizes.
        ``p`` is still the live DTensor param, so its spec is read directly
        rather than cached. ``run_check=False`` skips the collective that
        would otherwise verify a ``Replicate()`` placement is consistent
        across ranks (FSDP2 params do carry ``Replicate()`` under HSDP, on
        the dp_replicate axis) -- safe here because EMA's update is a pure,
        per-rank-local function of already-consistent local data, so every
        replica computes the same result without needing a cross-rank check.
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

    def _get_scratch(self, key: Any, params: list[torch.Tensor]) -> list[torch.Tensor]:
        scratch = self._offload_scratch.get(key)
        if scratch is None:
            scratch = [torch.empty_like(p) for p in params]
            self._offload_scratch[key] = scratch
        return scratch

    # TODO: this only makes the offload stream wait for the compute stream
    # (via wait_stream below) before reading live params -- nothing makes
    # the compute stream wait for the offload stream's read before the next
    # iteration's optimizer.step() overwrites those same param tensors.
    # _maybe_wait_pending()'s event.synchronize() is a host-side block that
    # runs too late to help: by the time it executes, the next iteration's
    # write kernels may already be enqueued on the compute stream. A correct
    # fix needs the compute stream to wait on this stream's completion event
    # (e.g. current_stream().wait_event(...)) rather than a host-side sync.
    def _maybe_wait_pending(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
            self._pending_event = None

    def _update_offloaded(
        self,
        scratch_key: Any,
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
        try:
            return super().state_dict()
        finally:
            for ema_opt in self.optimizers:
                for p, param_state in ema_opt.state.items():
                    param_state["ema_params"] = originals[id(p)]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            # Checkpoint had no EMA data (excluded via exclude_from_loading,
            # or predates this feature) -- cold-start from the just-loaded
            # model weights (and buffers, if buffer_patterns is set).
            for ema_opt, model in zip(self._param_optimizers, self.model_parts):
                for p in (p for p in model.parameters() if p.requires_grad):
                    source = p.detach()
                    if self.offload_to_cpu:
                        source = self._local_view(source)
                    ema_opt.state[p]["ema_params"].copy_(source)
            for ema_opt, model in zip(self._buffer_optimizers, self.model_parts):
                for name, b in model.named_buffers():
                    if not any(p.search(name) for p in self._buffer_patterns):
                        continue
                    source = b.detach()
                    if self.offload_to_cpu:
                        source = self._local_view(source)
                    ema_opt.state[b]["ema_params"].copy_(source)
            return
        # DCP calls our state_dict() above to build its load template, so it
        # already receives real DTensors here too -- re-pin them afterward.
        try:
            super().load_state_dict(state_dict)
        finally:
            if self.offload_to_cpu:
                for ema_opt in self.optimizers:
                    for param_state in ema_opt.state.values():
                        param_state["ema_params"] = self._pin_local(
                            param_state["ema_params"]
                        )
