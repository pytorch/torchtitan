# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import re
from collections.abc import Sequence
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

    def __init__(self, named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
        tensors = [t for _, t in named_tensors]
        names = [canonical_fqn(name) for name, _ in named_tensors]
        super().__init__([{"params": tensors, "param_names": names}], {})
        for t in tensors:
            self.state[t]["ema_params"] = t.detach().clone()

    def step(self, closure=None) -> None:  # pyrefly: ignore[bad-override]
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

    # Deliberately extends Configurable.Config, not OptimizersContainer.Config:
    # inheriting the optimizer's fields would put lr/param_groups on the EMA's
    # command-line surface.
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
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
        """Last Trainer.step before EMA tracking begins, so the first update
        fires at start_step + update_every_n_steps. Decoupled from the LR
        scheduler's WSD phases."""

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

        def __post_init__(self) -> None:
            if self.update_every_n_steps < 1:
                raise ValueError("ema.update_every_n_steps must be greater than 0.")
            if not math.isfinite(self.half_life_fraction):
                raise ValueError("ema.half_life_fraction must be finite.")
            if self.half_life_fraction <= 0:
                raise ValueError("ema.half_life_fraction must be greater than 0.")
            if self.step_bias < 0:
                raise ValueError(
                    "ema.step_bias must not be negative; it is added to the firing "
                    "count, and a non-positive count has no decay."
                )
            if self.decay is not None and not (
                math.isfinite(self.decay) and 0 <= self.decay < 1
            ):
                raise ValueError(
                    "ema.decay must be finite and in [0, 1); "
                    "decay=1 never updates the EMA."
                )
            # A fixed decay replaces the half-life schedule outright, so a
            # half_life_fraction set alongside it would do nothing.
            default_half_life = (
                type(self).__dataclass_fields__["half_life_fraction"].default
            )
            if self.decay is not None and self.half_life_fraction != default_half_life:
                logger.warning(
                    "ema.half_life_fraction=%s is ignored because ema.decay=%s is "
                    "set; the decay is then fixed and the half-life schedule is "
                    "never used. Leave decay unset to use half_life_fraction.",
                    self.half_life_fraction,
                    self.decay,
                )

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
                for name, b in named_buffers:
                    if not (torch.is_floating_point(b) or torch.is_complex(b)):
                        raise ValueError(
                            f"EMA.Config.buffer_patterns matched buffer {name!r} of "
                            f"dtype {b.dtype}. Only floating-point and complex buffers "
                            "can be averaged: an integer or boolean average has to be "
                            "rounded back into the buffer's dtype, which freezes it "
                            "once the per-step increment falls below one."
                        )
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
        low_precision = sorted(
            {
                str(param_state["ema_params"].dtype)
                for ema_opt in self.optimizers
                for param_state in ema_opt.state.values()
                if param_state["ema_params"].dtype
                not in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            }
        )
        if low_precision:
            logger.warning(
                "EMA is tracking %s tensors. Once the average and the live "
                "value are close, the per-firing increment rounds away at that "
                "precision and the EMA stops tracking altogether. Keep the "
                "tracked tensors in float32 (FSDP2's mixed_precision_param "
                "already does).",
                ", ".join(low_precision),
            )

        self._post_init(all_params)

        self._offload_stream: torch.cuda.Stream | None = None
        self._offload_pool: torch.Tensor | None = None
        self._pending_event: torch.cuda.Event | None = None
        if self.offload_to_cpu:
            self._init_cpu_offload()

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    # Takes the step rather than an optimizer closure: the firing count has to
    # be derived from it, and this is never merged into Trainer.optimizers.
    @torch.no_grad()
    def step(self, current_step: int) -> None:  # pyrefly: ignore[bad-override]
        """Call directly with the trainer's global step -- never merged into
        Trainer.optimizers, so there's no closure/zero-arg step() to honor."""
        elapsed = current_step - self.start_step
        if elapsed <= 0 or elapsed % self.update_every_n_steps != 0:
            return
        # num_updates is the firing count (1, 2, 3, ...), derived from
        # current_step rather than kept as a counter so that it survives a
        # checkpoint resume: only ema_params are checkpointed, so a stored
        # counter would restart at 0 and the decay would collapse to a full
        # overwrite of the restored EMA. step_bias is added after the division
        # so its value is never truncated by update_every_n_steps.
        num_updates = elapsed // self.update_every_n_steps + self.step_bias
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
        for ema_opt, model in zip(self._param_optimizers, self.model_parts):
            params: list[torch.Tensor] = [
                p for p in model.parameters() if p.requires_grad
            ]
            self._update_group(ema_opt, params, decay)
        for ema_opt, model in zip(self._buffer_optimizers, self.model_parts):
            buffers: list[torch.Tensor] = [
                b
                for name, b in model.named_buffers()
                if any(p.search(name) for p in self._buffer_patterns)
            ]
            self._update_group(ema_opt, buffers, decay)

    def _update_group(
        self,
        ema_opt: "_EMAParamOptimizer",
        tensors: list[torch.Tensor],
        decay: float,
    ) -> None:
        """Shared lerp/decay body for one model part's params or buffers."""
        if not tensors:
            return
        # State is keyed by tensor identity and the tracked set is fixed at
        # construction (see the TODO on _update), so a tensor that appeared or
        # was replaced since then has no entry. state is a defaultdict, so
        # indexing it here would insert an empty entry and fail later with a
        # bare KeyError("ema_params").
        untracked = [t for t in tensors if "ema_params" not in ema_opt.state.get(t, {})]
        if untracked:
            raise RuntimeError(
                f"EMA has no state for {len(untracked)} of {len(tensors)} tensors "
                f"(first: shape {tuple(untracked[0].shape)}, dtype "
                f"{untracked[0].dtype}). EMA tracks the tensors that existed when "
                "it was built, by identity, so unfreezing a parameter, replacing a "
                "parameter or buffer, or rebuilding part of the model after the EMA "
                "is constructed is not supported. Build the EMA after the model is "
                "final."
            )
        ema_params = [ema_opt.state[t]["ema_params"] for t in tensors]
        if self.offload_to_cpu:
            # ema_params are pinned local-shard CPU tensors; localize the
            # live tensors too so the foreach ops never mix DTensor with
            # Tensor.
            local_tensors = [self._local_view(t) for t in tensors]
            self._update_offloaded(local_tensors, ema_params, decay)
            return
        torch._foreach_lerp_(ema_params, tensors, 1.0 - decay)

    # --- CPU offload path (GH200-optimized: async side-stream, pinned memory) ---

    # Upper bound on the GPU scratch the offload pipeline holds. Scratch is
    # carved from one flat buffer and reused chunk by chunk, so it stays this
    # size; a per-tensor cache would instead grow to 1x parameter memory, which
    # is exactly what offloading is meant to free.
    _SCRATCH_BYTES = 128 * 1024 * 1024

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
        sharding, for the copy handed to DCP at checkpoint save/load -- this
        is what DCP needs to (re)shard EMA state correctly across world sizes.
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
        local_gpu = local.to(p.device)
        return DTensor.from_local(
            local_gpu,
            device_mesh=p.device_mesh,
            placements=p.placements,
            run_check=False,
        )

    def _scratch_chunks(
        self, params: list[torch.Tensor], ema_params: list[torch.Tensor]
    ):
        """Split one group into chunks backed by a single reusable GPU buffer.

        Yields (params, ema_params, scratch) triples whose scratch tensors are
        views into that buffer, so peak scratch stays at _SCRATCH_BYTES (or the
        largest single tensor, when that is larger) however many tensors are
        tracked. Offsets are 16-byte aligned so the uint8 pool can be viewed as
        any dtype.
        """
        sizes = [p.numel() * p.element_size() for p in params]
        budget = max(self._SCRATCH_BYTES, max(sizes))
        device = params[0].device
        pool = self._offload_pool
        if pool is None or pool.numel() < budget or pool.device != device:
            pool = torch.empty(budget, dtype=torch.uint8, device=device)
            self._offload_pool = pool
        chunk_p: list[torch.Tensor] = []
        chunk_e: list[torch.Tensor] = []
        scratch: list[torch.Tensor] = []
        offset = 0
        for param, ema_param, size in zip(params, ema_params, sizes):
            start = (offset + 15) // 16 * 16
            if chunk_p and start + size > budget:
                yield chunk_p, chunk_e, scratch
                chunk_p, chunk_e, scratch, start = [], [], [], 0
            chunk_p.append(param)
            chunk_e.append(ema_param)
            scratch.append(
                pool[start : start + size].view(param.dtype).view(param.shape)
            )
            offset = start + size
        if chunk_p:
            yield chunk_p, chunk_e, scratch

    def _maybe_wait_pending(self) -> None:
        """Host-side wait for the trailing D2H, before ema_params is read on
        the host at checkpoint save. Firing-to-firing ordering needs no host
        sync: the whole pipeline runs on one stream and is already ordered."""
        if self._pending_event is not None:
            self._pending_event.synchronize()
            self._pending_event = None

    def _update_offloaded(
        self,
        params: list[torch.Tensor],
        ema_params: list[torch.Tensor],
        decay: float,
    ) -> None:
        stream = self._offload_stream
        assert stream is not None
        stream.wait_stream(torch.cuda.current_stream())
        params_read = torch.cuda.Event()
        with torch.cuda.stream(stream):
            chunks = list(self._scratch_chunks(params, ema_params))
            for index, (chunk_p, chunk_e, scratch) in enumerate(chunks):
                torch._foreach_copy_(scratch, chunk_e, non_blocking=True)  # H2D
                torch._foreach_lerp_(scratch, chunk_p, 1.0 - decay)
                if index == len(chunks) - 1:
                    # Every read of the live params is now enqueued.
                    params_read.record(stream)
                torch._foreach_copy_(chunk_e, scratch, non_blocking=True)  # D2H
            self._pending_event = torch.cuda.Event()
            self._pending_event.record(stream)
        # Hold the compute stream until those reads complete, so the next
        # iteration's optimizer.step() cannot overwrite the params mid-read.
        # The trailing D2H still overlaps with compute.
        torch.cuda.current_stream().wait_event(params_read)

    # --- checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        if not self.offload_to_cpu:
            return super().state_dict()
        # The trailing D2H has to land before DCP reads ema_params.
        self._maybe_wait_pending()
        # super()'s values are the pinned tensors themselves, so materialize
        # DTensors by mapping those values rather than swapping them into the
        # container and putting them back afterwards. Nothing is mutated, so no
        # failure can leave the container holding live GPU DTensors.
        owner = {
            id(param_state["ema_params"]): t
            for ema_opt in self.optimizers
            for t, param_state in ema_opt.state.items()
        }
        materialized = {}
        for key, value in super().state_dict().items():
            if not torch.is_tensor(value):
                materialized[key] = value  # e.g. a param_groups scalar
                continue
            tensor = owner.get(id(value))
            if tensor is None:
                # A tensor value must be the pinned ema_params itself, or the
                # local shard would reach DCP unwrapped and silently fail to
                # reshard. Anything else means super() started copying.
                raise RuntimeError(
                    f"EMA state dict entry {key!r} is not one of the pinned "
                    "ema_params tensors, so it cannot be rewrapped as a "
                    "DTensor for checkpointing."
                )
            materialized[key] = self._materialize_dtensor(tensor, value)
        return materialized

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
            logger.warning(
                "EMA state was not restored; cold-starting it from the loaded "
                "model weights, which discards all EMA history. Expected the "
                "first time EMA is enabled against an older checkpoint. If "
                'checkpoint.exclude_from_loading still lists "ema", remove it, '
                "or every later resume will discard the EMA again."
            )
            return
        if not self.offload_to_cpu:
            super().load_state_dict(state_dict)
            return
        # DCP filled the DTensors state_dict() handed it. Copy those back into
        # the pinned tensors in place, instead of letting
        # Optimizer.load_state_dict swap in the GPU tensors and re-pinning
        # after -- the pinned tensor is never replaced, so the offload
        # invariant is never suspended. The key layout is whatever
        # get_flat_optim_state_dict produces; a test pins that it still
        # matches super().state_dict().
        for ema_opt in self.optimizers:
            for group in ema_opt.param_groups:
                for fqn, tensor in zip(group["param_names"], group["params"]):
                    # A key this container does not own is skipped, matching
                    # load_flat_optim_state_dict, so both paths behave alike.
                    incoming = state_dict.get(f"state.{fqn}.ema_params")
                    if incoming is None:
                        continue
                    ema_opt.state[tensor]["ema_params"].copy_(
                        self._local_view(incoming)
                    )
