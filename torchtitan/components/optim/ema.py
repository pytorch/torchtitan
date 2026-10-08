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
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable

from .optimizer import OptimizersContainer

__all__ = [
    "EMA",
    "FIXED_DECAY_KEY_PREFIX",
    "HALF_LIFE_KEY_PREFIX",
    "ema_key",
    "parse_ema_key",
]

logger = logging.getLogger(__name__)

# Per-tensor state entry holding one EMA copy per configured decay, keyed by
# ema_key(). DCP flattens it to "state.<fqn>.ema_params.<ema_key>".
_EMA_STATE_KEY = "ema_params"
FIXED_DECAY_KEY_PREFIX = "decay_"
HALF_LIFE_KEY_PREFIX = "half_life_"


def ema_key(prefix: str, value: float) -> str:
    """Checkpoint key for one EMA copy, derived from its decay/half life fraction setting.
    E.g. ema_key(FIXED_DECAY_KEY_PREFIX, 0.999) == "decay_0p999".
    """
    encoded = repr(float(value)).replace("+", "").replace(".", "p").replace("-", "m")
    return f"{prefix}{encoded}"


def parse_ema_key(key: str) -> tuple[bool, float]:
    """Inverse of ema_key(): ``(is_decay, value)``, where ``is_decay`` is True
    for an ``EMA.Config.decays`` entry and False for a ``half_life_fractions``
    entry. E.g. parse_ema_key("decay_0p999") == (True, 0.999).
    """
    for prefix, is_decay in (
        (FIXED_DECAY_KEY_PREFIX, True),
        (HALF_LIFE_KEY_PREFIX, False),
    ):
        if key.startswith(prefix):
            encoded = key[len(prefix) :]
            return is_decay, float(encoded.replace("p", ".").replace("m", "-"))
    raise ValueError(
        f"EMA key {key!r} is neither {FIXED_DECAY_KEY_PREFIX}<value> nor "
        f"{HALF_LIFE_KEY_PREFIX}<value>."
    )


class _EMAParamOptimizer(Optimizer):
    """Holds ``state[t]["ema_params"][ema_key]`` per tensor (parameter or
    buffer) for one model part -- one EMA copy per configured decay.

    Never step()-ed; reuses ``Optimizer``'s per-tensor state dict plus the
    FQN-flattening DCP machinery in ``checkpointer/utils.py`` instead of a
    bespoke DTensor state-dict format. Also used for buffer EMA (e.g. MoE's
    ``expert_bias_E``), which is why tensors need not be ``nn.Parameter``s.
    """

    def __init__(
        self,
        named_tensors: Sequence[tuple[str, torch.Tensor]],
        ema_keys: Sequence[str],
    ) -> None:
        tensors = [t for _, t in named_tensors]
        names = [canonical_fqn(name) for name, _ in named_tensors]
        super().__init__([{"params": tensors, "param_names": names}], {})
        for t in tensors:
            self.state[t][_EMA_STATE_KEY] = {
                key: t.detach().clone() for key in ema_keys
            }

    def step(self, closure=None) -> None:  # pyrefly: ignore[bad-override]
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call "
            "EMA.step(current_step) instead."
        )


class EMA(OptimizersContainer):
    """Pseudo-optimizer maintaining online EMAs of model weights.

    Keeps one EMA copy per entry of ``Config.decays`` and
    ``Config.half_life_fractions``. All copies track the same tensors on the
    same cadence, so one pass over the model per firing updates every copy.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizers`` or
    ``LRSchedulersContainer``. ``Optim`` builds and steps it when
    ``Optim.Config.ema`` is set.
    """

    # Deliberately extends Configurable.Config, not OptimizersContainer.Config:
    # inheriting the optimizer's fields would put lr/param_groups on the EMA's
    # command-line surface.
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        decays: list[float] = field(default_factory=list)
        """Fixed-decay EMAs, one copy per entry: ema_params = decay *
        ema_params + (1 - decay) * param on every firing. Each entry must be
        finite and in [0, 1). Checkpointed under ema_key("decay_", decay),
        e.g. "decay_0p999". At least one of decays and half_life_fractions
        must be non-empty; only the listed copies are tracked."""

        half_life_fractions: list[float] = field(default_factory=list)
        """Half-life-schedule EMAs, one copy per entry: decay = 2 ** (-1 /
        (half_life_fraction * num_updates)). Keeps roughly the most recent
        half_life_fraction share of updates dominant. Each entry must be finite
        and positive. Checkpointed under ema_key("half_life_", fraction), e.g.
        "half_life_0p05"."""

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
                raise ValueError(
                    "optim.ema.update_every_n_steps must be greater than 0."
                )
            if self.step_bias < 0:
                raise ValueError(
                    "optim.ema.step_bias must not be negative; it is added to the firing "
                    "count, and a non-positive count has no decay."
                )
            if not self.decays and not self.half_life_fractions:
                raise ValueError(
                    "optim.ema.decays and optim.ema.half_life_fractions are both "
                    "empty, so the EMA would track nothing. Set at least one, or "
                    "leave optim.ema unset to disable EMA."
                )
            for decay in self.decays:
                if not (math.isfinite(decay) and 0 <= decay < 1):
                    raise ValueError(
                        f"optim.ema.decays entry {decay} must be finite and in "
                        "[0, 1); decay=1 never updates the EMA."
                    )
            for fraction in self.half_life_fractions:
                if not (math.isfinite(fraction) and fraction > 0):
                    raise ValueError(
                        f"optim.ema.half_life_fractions entry {fraction} must be "
                        "finite and greater than 0."
                    )
            keys = [ema_key(FIXED_DECAY_KEY_PREFIX, d) for d in self.decays] + [
                ema_key(HALF_LIFE_KEY_PREFIX, f) for f in self.half_life_fractions
            ]
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            if duplicates:
                raise ValueError(
                    f"optim.ema has duplicate EMA copies {duplicates}; each entry "
                    "of decays/half_life_fractions must be unique."
                )

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        self.decays = config.decays
        self.half_life_fractions = config.half_life_fractions
        self.ema_keys = [ema_key(FIXED_DECAY_KEY_PREFIX, d) for d in self.decays] + [
            ema_key(HALF_LIFE_KEY_PREFIX, f) for f in self.half_life_fractions
        ]
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
            self._param_optimizers.append(
                _EMAParamOptimizer(named_params, self.ema_keys)
            )
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
                self._buffer_optimizers.append(
                    _EMAParamOptimizer(named_buffers, self.ema_keys)
                )
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
        self.optimizers = cast(
            list[Optimizer],
            self._param_optimizers + self._buffer_optimizers,
        )
        low_precision = sorted(
            {
                str(ema_param.dtype)
                for ema_opt in self.optimizers
                for param_state in ema_opt.state.values()
                for ema_param in param_state[_EMA_STATE_KEY].values()
                if ema_param.dtype
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
    # be derived from it, and this is never merged into the training optimizers.
    @torch.no_grad()
    def step(self, current_step: int) -> None:  # pyrefly: ignore[bad-override]
        """Call directly with the trainer's global step -- never merged into
        the training optimizers, so there is no closure/zero-arg step() to honor."""
        elapsed = current_step - self.start_step
        if elapsed <= 0 or elapsed % self.update_every_n_steps != 0:
            return
        # num_updates is the firing count (1, 2, 3, ...), derived from
        # current_step rather than kept as a counter so that it survives a
        # checkpoint resume: only the EMA copies are checkpointed, so a stored
        # counter would restart at 0 and the decay would collapse to a full
        # overwrite of the restored EMA. step_bias is added after the division
        # so its value is never truncated by update_every_n_steps.
        num_updates = elapsed // self.update_every_n_steps + self.step_bias
        self._update(num_updates)

    def _decays_at(self, num_updates: int) -> list[float]:
        """This firing's decay for each EMA copy, in ``self.ema_keys`` order."""
        return self.decays + [
            2.0 ** (-1.0 / (f * num_updates)) for f in self.half_life_fractions
        ]

    # TODO: params/buffers are re-derived from the model on every firing
    # (model.parameters()/model.named_buffers() + regex matching for
    # buffers), even though the identical lists are already sitting in
    # ema_opt.param_groups[0]["params"] from construction -- EMA state is
    # already looked up by tensor identity, so identity (and thus this list)
    # is already assumed stable across the run. Reusing param_groups[0]
    # instead of recomputing would remove this per-step traversal/regex cost
    # with no behavior change.
    def _update(self, num_updates: int) -> None:
        decays = self._decays_at(num_updates)
        for ema_opt, model in zip(self._param_optimizers, self.model_parts):
            params: list[torch.Tensor] = [
                p for p in model.parameters() if p.requires_grad
            ]
            self._update_group(ema_opt, params, decays)
        for ema_opt, model in zip(self._buffer_optimizers, self.model_parts):
            buffers: list[torch.Tensor] = [
                b
                for name, b in model.named_buffers()
                if any(p.search(name) for p in self._buffer_patterns)
            ]
            self._update_group(ema_opt, buffers, decays)

    def _update_group(
        self,
        ema_opt: "_EMAParamOptimizer",
        tensors: list[torch.Tensor],
        decays: list[float],
    ) -> None:
        """Shared lerp/decay body for one model part's params or buffers.

        ``decays`` holds this firing's decay for each EMA copy, in
        ``self.ema_keys`` order.
        """
        if not tensors:
            return
        # State is keyed by tensor identity and the tracked set is fixed at
        # construction (see the TODO on _update), so a tensor that appeared or
        # was replaced since then has no entry. state is a defaultdict, so
        # indexing it here would insert an empty entry and fail later with a
        # bare KeyError("ema_params").
        untracked = [
            t for t in tensors if _EMA_STATE_KEY not in ema_opt.state.get(t, {})
        ]
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
        if self.offload_to_cpu:
            # ema_params are pinned local-shard CPU tensors; localize the
            # live tensors too so the foreach ops never mix DTensor with
            # Tensor.
            local_tensors = [self._local_view(t) for t in tensors]
        # Each EMA copy is updated independently, one after another.
        # TODO: with offload_to_cpu, each copy runs its own _update_offloaded,
        # so K copies cost K wait_stream/wait_event round trips and hold the
        # compute stream until the K-th copy's param reads finish. Fusing the
        # copies per chunk (H2D all K copies -> K lerps against one read of
        # chunk_p -> D2H) would keep the stall at about one copy's worth.
        for key, decay in zip(self.ema_keys, decays):
            ema_params = [ema_opt.state[t][_EMA_STATE_KEY][key] for t in tensors]
            if self.offload_to_cpu:
                # pyrefly: ignore [unbound-name]
                self._update_offloaded(local_tensors, ema_params, decay)
            else:
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
                copies = param_state[_EMA_STATE_KEY]
                for key in copies:
                    copies[key] = self._pin_local(copies[key])

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
        # The trailing D2H has to land before DCP reads the EMA copies.
        self._maybe_wait_pending()
        # super()'s values are the pinned tensors themselves, so materialize
        # DTensors by mapping those values rather than swapping them into the
        # container and putting them back afterwards. Nothing is mutated, so no
        # failure can leave the container holding live GPU DTensors.
        owner = {
            id(ema_param): t
            for ema_opt in self.optimizers
            for t, param_state in ema_opt.state.items()
            for ema_param in param_state[_EMA_STATE_KEY].values()
        }
        materialized = {}
        for key, value in super().state_dict().items():
            if not torch.is_tensor(value):
                materialized[key] = value  # e.g. a param_groups scalar
                continue
            tensor = owner.get(id(value))
            if tensor is None:
                # A tensor value must be a pinned EMA copy itself, or the
                # local shard would reach DCP unwrapped and silently fail to
                # reshard. Anything else means super() started copying.
                raise RuntimeError(
                    f"EMA state dict entry {key!r} is not one of the pinned "
                    "EMA tensors, so it cannot be rewrapped as a "
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
                    for ema_param in ema_opt.state[p][_EMA_STATE_KEY].values():
                        ema_param.copy_(source)
            for ema_opt, model in zip(self._buffer_optimizers, self.model_parts):
                for name, b in model.named_buffers():
                    if not any(p.search(name) for p in self._buffer_patterns):
                        continue
                    source = b.detach()
                    if self.offload_to_cpu:
                        source = self._local_view(source)
                    for ema_param in ema_opt.state[b][_EMA_STATE_KEY].values():
                        ema_param.copy_(source)
            logger.warning(
                "EMA state was not restored; cold-starting every EMA copy (%s) "
                "from the loaded model weights, which discards all EMA history. "
                "Expected the first time EMA is enabled, or a decay is added, "
                "against an older checkpoint. If "
                'checkpoint.exclude_from_loading still lists "ema", remove it, '
                "or every later resume will discard the EMA again.",
                ", ".join(self.ema_keys),
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
                    copies = ema_opt.state[tensor][_EMA_STATE_KEY]
                    for key, ema_param in copies.items():
                        # A key this container does not own is skipped,
                        # matching load_flat_optim_state_dict, so both paths
                        # behave alike.
                        incoming = state_dict.get(f"state.{fqn}.{_EMA_STATE_KEY}.{key}")
                        if incoming is None:
                            continue
                        ema_param.copy_(self._local_view(incoming))
