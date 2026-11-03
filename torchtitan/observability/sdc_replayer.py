# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic replay for silent data corruption (SDC) detection.

The replayer checks every forward/backward call of each checked optimizer
step. A call covers one or more gradient accumulation steps and their pipeline
microbatches; a step may span several calls when microbatches are streamed in.
For each call of a checked step it:

1. snapshots Python, CPU, and accelerator RNG state, registered module
   buffers, and the gradients the call starts with;
2. runs the forward/backward once and records a reference signature (loss,
   gradients, buffers, and RNG advancement);
3. restores the snapshot and re-executes ``num_replays`` times, comparing
   each signature with the reference;
4. raises ``SDCReplayMismatch`` on every rank on any divergence; otherwise
   only the final execution's effects remain committed.

A call may start with gradients pending from the step's earlier calls. Restore
copies their values back in place, so a re-run adds to the same gradients the
reference run added to, at the same storage addresses.

Replay is one of several SDC detection strategies (others include shadow
computation on redundant hardware and algorithm-level checks such as
checksummed matmuls). It trades extra forward/backward time on checked
steps for an in-training, hardware-agnostic check, and requires fully
deterministic execution (``debug.deterministic``).

Unregistered execution scratch state is intentionally outside the replay
contract when each invocation overwrites it before reading it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import Configurable


@dataclass(frozen=True, slots=True)
class _ReplaySignature:
    schema: tuple[tuple[str, tuple[int, ...], str, str], ...]
    digests: tuple[torch.Tensor, ...]
    state: tuple[tuple[str, Any], ...]

    def clone(self) -> "_ReplaySignature":
        return _ReplaySignature(
            schema=self.schema,
            digests=tuple(digest.clone() for digest in self.digests),
            state=self.state,
        )


_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class _ReplayResult(Generic[_ResultT]):
    result: _ResultT
    signature: _ReplaySignature


class SDCReplayMismatch(RuntimeError):
    """Raised on every rank when deterministic replay finds a mismatch.

    ``local_step`` is the 1-based position of the mismatching optimizer step
    in the current check schedule; the schedule restarts on ``reset_schedule``
    (e.g. after a checkpoint load), so ``local_step`` restarts with it.
    """

    def __init__(
        self,
        *,
        step: int,
        local_step: int,
        replay: int,
        rank: int,
        signature_mismatch: str | None,
    ) -> None:
        self.step = step
        self.local_step = local_step
        self.replay = replay
        self.rank = rank
        self.signature_mismatch = signature_mismatch
        super().__init__(
            "SDC replay mismatch: "
            f"step={step}, local_step={local_step}, replay={replay}, "
            f"rank={rank}, signature={signature_mismatch!r}"
        )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _hash_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a lightweight device-side digest for replay comparison.

    The current hash is order-insensitive and can therefore collide for tensor
    permutations. Complex tensors are viewed as real components and inherit the
    same limitation across their real and imaginary values. This hash is used
    because it is simple to apply across tensors and fast enough for replay.
    """
    local = _local_tensor(tensor).detach()
    if local.numel() == 0:
        return torch.zeros((), dtype=torch.uint64, device=local.device)
    if local.is_complex():
        # for RoPE caches
        local = torch.view_as_real(local)
    return torch.hash_tensor(local)


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return _local_tensor(tensor).detach().clone()


def _accelerator_rng_states(device: torch.device) -> list[torch.Tensor] | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_rng_state_all()
    device_module = getattr(torch, device.type, None)
    get_rng_state_all = getattr(device_module, "get_rng_state_all", None)
    if get_rng_state_all is not None:
        return get_rng_state_all()
    return None


@dataclass(frozen=True, slots=True)
class _BufferSnapshot:
    module: torch.nn.Module
    name: str
    original: torch.Tensor | None
    value: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    parameter: torch.nn.Parameter
    original: torch.Tensor | None
    value: torch.Tensor | None
    """A copy of ``original``'s values, or None when the parameter had no gradient."""


@dataclass(frozen=True, slots=True)
class _ReplayStateSnapshot:
    python_rng_state: tuple[Any, ...]
    cpu_rng_state: torch.Tensor
    accelerator_rng_states: list[torch.Tensor] | None
    buffers: tuple[_BufferSnapshot, ...]
    gradients: tuple[_GradientSnapshot, ...]


class _ReplayStateProvider:
    """Captures and restores the semantic state of one replayed execution."""

    def __init__(
        self,
        modules: Iterable[torch.nn.Module],
        device: torch.device,
    ) -> None:
        self._modules = tuple(dict.fromkeys(modules))
        # pyrefly: ignore [read-only]
        self._device = device

    def capture(self) -> _ReplayStateSnapshot:
        buffers: list[_BufferSnapshot] = []
        parameters: list[torch.nn.Parameter] = []
        seen_parameters: set[int] = set()
        for root in self._modules:
            for module in root.modules():
                for name, buffer in module._buffers.items():
                    buffers.append(
                        _BufferSnapshot(
                            module=module,
                            name=name,
                            original=buffer,
                            value=None if buffer is None else _clone_tensor(buffer),
                        )
                    )
                for parameter in module.parameters(recurse=False):
                    if id(parameter) not in seen_parameters:
                        seen_parameters.add(id(parameter))
                        parameters.append(parameter)

        # A call starts with None gradients (the step's first call) or with the
        # step's earlier calls' gradients. Restore keeps each tensor's identity,
        # so accelerator graphs keep their addresses, and copies its values back.
        gradients = tuple(
            _GradientSnapshot(
                parameter=parameter,
                original=parameter.grad,
                value=None if parameter.grad is None else _clone_tensor(parameter.grad),
            )
            for parameter in parameters
        )
        return _ReplayStateSnapshot(
            python_rng_state=random.getstate(),
            cpu_rng_state=torch.get_rng_state(),
            accelerator_rng_states=_accelerator_rng_states(self._device),
            buffers=tuple(buffers),
            gradients=gradients,
        )

    def restore(self, state: _ReplayStateSnapshot) -> None:
        random.setstate(state.python_rng_state)
        torch.set_rng_state(state.cpu_rng_state)
        if state.accelerator_rng_states is not None:
            if self._device.type == "cuda":
                torch.cuda.set_rng_state_all(state.accelerator_rng_states)
            else:
                device_module = getattr(torch, self._device.type, None)
                set_rng_state_all = getattr(device_module, "set_rng_state_all", None)
                if set_rng_state_all is None:
                    raise RuntimeError(
                        f"Cannot restore RNG state for accelerator {self._device.type}."
                    )
                set_rng_state_all(state.accelerator_rng_states)
        for snapshot in state.buffers:
            current = snapshot.module._buffers[snapshot.name]
            if snapshot.original is None:
                snapshot.module._buffers[snapshot.name] = None
                continue
            # TODO: with accelerator graphs and num_replays > 1, a replay writes the buffer that
            # its graph captured, not the one put back here, so the check later reports an
            # SDCReplayMismatch. Raise a config error here instead.
            if current is not snapshot.original:
                snapshot.module._buffers[snapshot.name] = snapshot.original
            assert snapshot.value is not None
            _local_tensor(snapshot.original).copy_(snapshot.value)

        # Restore each entry gradient identity, then its values.
        for snapshot in state.gradients:
            if snapshot.original is None:
                snapshot.parameter.grad = None
            else:
                if snapshot.parameter.grad is not snapshot.original:
                    snapshot.parameter.grad = snapshot.original
                assert snapshot.value is not None
                _local_tensor(snapshot.original).copy_(snapshot.value)


def _compare_signature(
    reference: _ReplaySignature, candidate: _ReplaySignature
) -> torch.Tensor:
    device = reference.digests[0].device
    if reference.schema != candidate.schema:
        return torch.ones((), dtype=torch.bool, device=device)
    if tuple(name for name, _ in reference.state) != tuple(
        name for name, _ in candidate.state
    ):
        return torch.ones((), dtype=torch.bool, device=device)
    for (name, ref_value), (_, candidate_value) in zip(
        reference.state, candidate.state, strict=True
    ):
        if ref_value != candidate_value:
            return torch.ones((), dtype=torch.bool, device=device)
    reference_digests = torch.stack(reference.digests)
    candidate_digests = torch.stack(candidate.digests)
    return torch.ne(reference_digests, candidate_digests).any()


def _find_signature_mismatch(
    reference: _ReplaySignature, candidate: _ReplaySignature
) -> str | None:
    if reference.schema != candidate.schema:
        return "tensor schema divergence"
    for index, (ref_digest, candidate_digest) in enumerate(
        zip(reference.digests, candidate.digests, strict=True)
    ):
        if not torch.equal(ref_digest, candidate_digest):
            return reference.schema[index][0]
    if tuple(name for name, _ in reference.state) != tuple(
        name for name, _ in candidate.state
    ):
        return "state schema divergence"
    for (name, ref_value), (_, candidate_value) in zip(
        reference.state, candidate.state, strict=True
    ):
        if ref_value != candidate_value:
            return name
    return None


class SDCReplayer(Configurable):
    """Replay-checks the forward/backward calls of scheduled optimizer steps.

    See the module docstring for the snapshot/execute/restore/compare
    lifecycle. The replayer owns its check schedule: ``step_completed`` counts
    the optimizer steps (``steps_since_reset``), and every call of the first
    ``config.num_steps`` of them is checked; ``reset_schedule`` restarts the
    count, e.g. after a checkpoint load, so the steps right after a restore
    are checked again.

    Example:
        replayer.run_fwd_bwd(call_1, step=1, get_loss=...)  # checked
        replayer.run_fwd_bwd(call_2, step=1, get_loss=...)  # checked, same step
        replayer.step_completed()                           # num_steps=1: later steps run unchecked
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_steps: int = 1
        """How many optimizer steps to check, counted from trainer start and
        restarting after every checkpoint load. Each checked step re-executes
        its forward/backward work ``num_replays`` extra times, so the default
        checks only the first step after every (re)start, where corruption
        from a bad restore or initialization is most likely. -1 checks every
        step (``1 + num_replays`` forward/backward calls per step)."""

        num_replays: int = 1
        """Number of times the checked forward/backward is re-executed and
        compared against the initial reference execution. Must be at least 1;
        higher values catch intermittent corruption a single replay can
        miss."""

        def __post_init__(self) -> None:
            if self.num_steps != -1 and self.num_steps < 1:
                raise ValueError("sdc_replayer.num_steps must be -1 or at least 1.")
            if self.num_replays < 1:
                raise ValueError("sdc_replayer.num_replays must be at least 1.")

    def __init__(
        self,
        config: Config,
        *,
        modules: Iterable[torch.nn.Module],
        device: torch.device,
    ) -> None:
        self.config = config
        self._modules = tuple(dict.fromkeys(modules))
        # pyrefly: ignore [read-only]
        self._device = device
        self._state_provider = _ReplayStateProvider(self._modules, device)
        self._steps_since_reset = 0
        self._validate_hash_support(device)

    @staticmethod
    def _validate_hash_support(device: torch.device) -> None:
        if not hasattr(torch, "hash_tensor"):
            raise RuntimeError("SDC replay requires torch.hash_tensor.")
        try:
            value = torch.arange(3, dtype=torch.int64, device=device)
            torch.hash_tensor(value)
        except (RuntimeError, TypeError) as error:
            raise RuntimeError("SDC replay requires torch.hash_tensor.") from error

    @property
    def steps_since_reset(self) -> int:
        """Optimizer steps observed since construction or ``reset_schedule``."""
        return self._steps_since_reset

    @property
    def checks_current_step(self) -> bool:
        """Whether ``run_fwd_bwd`` checks the calls of the current optimizer step."""
        return (
            self.config.num_steps == -1
            or self._steps_since_reset < self.config.num_steps
        )

    def step_completed(self) -> None:
        """Count one optimizer step; call after the step's last ``run_fwd_bwd``."""
        self._steps_since_reset += 1

    def reset_schedule(self) -> None:
        """Restart the check schedule; call after loading a checkpoint."""
        self._steps_since_reset = 0

    def run_fwd_bwd(
        self,
        execute: Callable[[], _ResultT],
        *,
        step: int,
        get_loss: Callable[[_ResultT], torch.Tensor],
    ) -> _ResultT:
        """Run one forward/backward call and replay-check it if its step is checked.

        Unchecked calls run ``execute`` once with no snapshot or signature
        overhead. ``get_loss`` extracts the loss tensor used in the replay
        signature. ``step`` is the global training step, used only for error
        reporting.
        """
        if not self.checks_current_step:
            return execute()
        return self._run_checked(
            execute,
            step=step,
            local_step=self._steps_since_reset + 1,
            get_loss=get_loss,
        )

    def _signature(self, loss: torch.Tensor) -> _ReplaySignature:
        """Capture tensor hashes and exact RNG states after one execution.

        Tensor digests remain on the loss device for batched comparison. Signature
        names identify the first differing loss, buffer, gradient, or RNG state
        only when a mismatch is detected. RNG states are compared exactly
        because the XOR-reduced tensor hash cancels the paired counter updates
        of a CPU draw (one torch.rand(1) collides about half the time).
        """
        schema: list[tuple[str, tuple[int, ...], str, str]] = []
        digests: list[torch.Tensor] = []
        digest_device = _local_tensor(loss).device

        def add_tensor(name: str, tensor: torch.Tensor) -> None:
            local = _local_tensor(tensor.detach())
            schema.append(
                (name, tuple(local.shape), str(local.dtype), str(local.device))
            )
            digests.append(_hash_tensor(local).to(digest_device))

        add_tensor("loss", loss)
        state: list[tuple[str, Any]] = [
            ("rng:cpu", torch.get_rng_state().numpy().tobytes())
        ]
        accelerator_rng_states = _accelerator_rng_states(self._device)
        if accelerator_rng_states is not None:
            for index, rng_state in enumerate(accelerator_rng_states):
                state.append((f"rng:accelerator:{index}", rng_state.numpy().tobytes()))

        seen_buffers: set[tuple[int, str]] = set()
        for module_index, root in enumerate(self._modules):
            for module_name, module in root.named_modules():
                for buffer_name, buffer in module._buffers.items():
                    buffer_key = (id(module), buffer_name)
                    if buffer_key in seen_buffers:
                        continue
                    seen_buffers.add(buffer_key)
                    qualified_name = (
                        f"{module_name}.{buffer_name}" if module_name else buffer_name
                    )
                    name = f"buffer:{module_index}:{qualified_name}"
                    if buffer is None:
                        schema.append((f"{name}:none", (), "none", "none"))
                        digests.append(
                            torch.zeros((), dtype=torch.uint64, device=digest_device)
                        )
                    else:
                        add_tensor(name, buffer)

        seen_parameters: set[int] = set()
        for module_index, module in enumerate(self._modules):
            for name, parameter in module.named_parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                grad = parameter.grad
                if grad is None:
                    schema.append(
                        (f"gradient:{module_index}:{name}:none", (), "none", "none")
                    )
                    digests.append(
                        torch.zeros((), dtype=torch.uint64, device=digest_device)
                    )
                    continue
                add_tensor(f"gradient:{module_index}:{name}", grad)

        state.append(("state:python_rng", random.getstate()))
        return _ReplaySignature(tuple(schema), tuple(digests), tuple(state))

    def _execute(
        self,
        execute: Callable[[], _ResultT],
        get_loss: Callable[[_ResultT], torch.Tensor],
    ) -> _ReplayResult[_ResultT]:
        result = execute()
        return _ReplayResult(result, self._signature(get_loss(result)))

    def _raise_if_mismatch(
        self,
        *,
        step: int,
        local_step: int,
        replay: int,
        local_mismatch: torch.Tensor,
        reference: _ReplaySignature,
        candidate: _ReplaySignature,
    ) -> None:
        mismatch = local_mismatch.to(dtype=torch.int32, device=self._device)
        global_mismatch = mismatch.clone()
        if dist.is_initialized():
            dist.all_reduce(global_mismatch, op=dist.ReduceOp.MAX)
        if not bool(global_mismatch.item()):
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        signature_mismatch = (
            _find_signature_mismatch(reference, candidate)
            if bool(mismatch.item())
            else None
        )
        details = (rank, signature_mismatch) if signature_mismatch is not None else None
        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, details)
            details = next(detail for detail in gathered if detail is not None)
        assert details is not None
        mismatch_rank, signature_mismatch = details
        raise SDCReplayMismatch(
            step=step,
            local_step=local_step,
            replay=replay,
            rank=mismatch_rank,
            signature_mismatch=signature_mismatch,
        )

    def _run_checked(
        self,
        execute: Callable[[], _ResultT],
        *,
        step: int,
        local_step: int,
        get_loss: Callable[[_ResultT], torch.Tensor],
    ) -> _ResultT:
        baseline = self._state_provider.capture()
        reference = self._execute(execute, get_loss)
        reference_signature = reference.signature.clone()
        del reference

        final_result: _ResultT | None = None
        for replay in range(1, self.config.num_replays + 1):
            self._state_provider.restore(baseline)
            candidate = self._execute(execute, get_loss)
            local_mismatch = _compare_signature(
                reference_signature, candidate.signature
            )
            self._raise_if_mismatch(
                step=step,
                local_step=local_step,
                replay=replay,
                local_mismatch=local_mismatch,
                reference=reference_signature,
                candidate=candidate.signature,
            )
            final_result = candidate.result

        assert final_result is not None
        return final_result
