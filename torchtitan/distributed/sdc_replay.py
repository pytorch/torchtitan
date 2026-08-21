# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic replay for SDC detection."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


@dataclass(kw_only=True, slots=True)
class SDCReplayConfig:
    enabled: bool = False
    """Enable deterministic forward/backward replay for SDC detection."""

    num_steps: int = 1
    """Number of attempt-local optimizer steps to check; -1 checks every step."""

    num_replays: int = 1
    """Number of candidate executions compared with the reference execution."""

    def validate(self) -> None:
        if self.num_steps != -1 and self.num_steps < 1:
            raise ValueError("sdc_replay.num_steps must be -1 or at least 1.")
        if self.num_replays < 1:
            raise ValueError("sdc_replay.num_replays must be at least 1.")


@dataclass(frozen=True, slots=True)
class ReplaySignature:
    schema: tuple[tuple[str, tuple[int, ...], str, str], ...]
    digests: tuple[torch.Tensor, ...]
    state: tuple[tuple[str, Any], ...]

    def clone(self) -> "ReplaySignature":
        return ReplaySignature(
            schema=self.schema,
            digests=tuple(digest.clone() for digest in self.digests),
            state=self.state,
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    loss: torch.Tensor
    signature: ReplaySignature


class SDCReplayMismatch(RuntimeError):
    """Raised on every rank when deterministic replay finds a mismatch."""

    def __init__(
        self,
        *,
        step: int,
        attempt: int,
        replay: int,
        rank: int,
        signature_mismatch: str | None,
    ) -> None:
        self.step = step
        self.attempt = attempt
        self.replay = replay
        self.rank = rank
        self.signature_mismatch = signature_mismatch
        super().__init__(
            "SDC replay mismatch: "
            f"step={step}, attempt={attempt}, replay={replay}, rank={rank}, "
            f"signature={signature_mismatch!r}"
        )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _hash_tensor(tensor: torch.Tensor) -> torch.Tensor:
    local = _local_tensor(tensor).detach()
    if local.numel() == 0:
        return torch.zeros((), dtype=torch.uint64, device=local.device)
    if local.is_complex():
        # for RoPE caches
        local = torch.view_as_real(local)
    return torch.hash_tensor(local)


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return _local_tensor(tensor).detach().clone()


def _accelerator_rng_states(trainer: Any) -> list[torch.Tensor] | None:
    device = getattr(trainer, "device", torch.device("cpu"))
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


@dataclass(frozen=True, slots=True)
class _TrainerStateSnapshot:
    python_rng_state: tuple[Any, ...]
    cpu_rng_state: torch.Tensor
    accelerator_rng_states: list[torch.Tensor] | None
    ntokens_seen: int
    buffers: tuple[_BufferSnapshot, ...]
    gradients: tuple[_GradientSnapshot, ...]


class TrainerReplayStateProvider:
    """Default semantic state provider for one trainer execution unit."""

    def __init__(
        self,
        trainer: Any,
        modules: Iterable[torch.nn.Module],
    ) -> None:
        self._trainer = trainer
        self._modules = tuple(dict.fromkeys(modules))

    def capture(self) -> _TrainerStateSnapshot:
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

        gradients = tuple(
            _GradientSnapshot(
                parameter=parameter,
                original=parameter.grad,
                value=(
                    None if parameter.grad is None else _clone_tensor(parameter.grad)
                ),
            )
            for parameter in parameters
        )
        return _TrainerStateSnapshot(
            python_rng_state=random.getstate(),
            cpu_rng_state=torch.get_rng_state(),
            accelerator_rng_states=_accelerator_rng_states(self._trainer),
            ntokens_seen=self._trainer.ntokens_seen,
            buffers=tuple(buffers),
            gradients=gradients,
        )

    def restore(self, state: _TrainerStateSnapshot) -> None:
        random.setstate(state.python_rng_state)
        torch.set_rng_state(state.cpu_rng_state)
        if state.accelerator_rng_states is not None:
            device = getattr(self._trainer, "device", torch.device("cpu"))
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(state.accelerator_rng_states)
            else:
                device_module = getattr(torch, device.type, None)
                set_rng_state_all = getattr(device_module, "set_rng_state_all", None)
                if set_rng_state_all is None:
                    raise RuntimeError(
                        f"Cannot restore RNG state for accelerator {device.type}."
                    )
                set_rng_state_all(state.accelerator_rng_states)
        self._trainer.ntokens_seen = state.ntokens_seen

        for snapshot in state.buffers:
            current = snapshot.module._buffers[snapshot.name]
            if snapshot.original is None:
                snapshot.module._buffers[snapshot.name] = None
                continue
            if current is not snapshot.original:
                snapshot.module._buffers[snapshot.name] = snapshot.original
            assert snapshot.value is not None
            _local_tensor(snapshot.original).copy_(snapshot.value)

        for snapshot in state.gradients:
            if snapshot.original is None:
                snapshot.parameter.grad = None
                continue
            if snapshot.parameter.grad is not snapshot.original:
                snapshot.parameter.grad = snapshot.original
            assert snapshot.value is not None
            _local_tensor(snapshot.original).copy_(snapshot.value)


def _compare_signature(
    reference: ReplaySignature, candidate: ReplaySignature
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
    reference: ReplaySignature, candidate: ReplaySignature
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


class SDCReplay:
    """Coordinates state restoration, repeated execution, and comparison."""

    Config = SDCReplayConfig

    def __init__(
        self,
        *,
        config: SDCReplayConfig,
        trainer: Any,
        modules: Iterable[torch.nn.Module],
    ) -> None:
        config.validate()
        self.config = config
        self._trainer = trainer
        self._modules = tuple(dict.fromkeys(modules))
        self._state_provider = TrainerReplayStateProvider(
            trainer,
            self._modules,
        )
        self._validate_hash_support(trainer.device)

    @staticmethod
    def _validate_hash_support(device: torch.device) -> None:
        if not hasattr(torch, "hash_tensor"):
            raise RuntimeError("SDC replay requires torch.hash_tensor.")
        try:
            value = torch.arange(3, dtype=torch.int64, device=device)
            torch.hash_tensor(value)
        except (RuntimeError, TypeError) as error:
            raise RuntimeError("SDC replay requires torch.hash_tensor.") from error

    def should_run(self, attempt_step: int) -> bool:
        return self.config.num_steps == -1 or attempt_step < self.config.num_steps

    def _signature(self, loss: torch.Tensor) -> ReplaySignature:
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
        add_tensor("rng:cpu", torch.get_rng_state())
        accelerator_rng_states = _accelerator_rng_states(self._trainer)
        if accelerator_rng_states is not None:
            for index, rng_state in enumerate(accelerator_rng_states):
                add_tensor(f"rng:accelerator:{index}", rng_state)

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

        state = (
            ("state:python_rng", random.getstate()),
            ("state:ntokens_seen", self._trainer.ntokens_seen),
        )
        return ReplaySignature(tuple(schema), tuple(digests), state)

    def _execute(self, execute: Callable[[], torch.Tensor]) -> ReplayResult:
        loss = execute()
        return ReplayResult(loss, self._signature(loss))

    def _raise_if_mismatch(
        self,
        *,
        step: int,
        attempt: int,
        replay: int,
        local_mismatch: torch.Tensor,
        reference: ReplaySignature,
        candidate: ReplaySignature,
    ) -> None:
        mismatch = local_mismatch.to(dtype=torch.int32, device=self._trainer.device)
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
            attempt=attempt,
            replay=replay,
            rank=mismatch_rank,
            signature_mismatch=signature_mismatch,
        )

    def run(
        self,
        execute: Callable[[], torch.Tensor],
        *,
        step: int,
        attempt: int,
    ) -> torch.Tensor:
        baseline = self._state_provider.capture()
        reference = self._execute(execute)
        reference_signature = reference.signature.clone()
        del reference

        final_loss: torch.Tensor | None = None
        for replay in range(1, self.config.num_replays + 1):
            self._state_provider.restore(baseline)
            candidate = self._execute(execute)
            local_mismatch = _compare_signature(
                reference_signature, candidate.signature
            )
            self._raise_if_mismatch(
                step=step,
                attempt=attempt,
                replay=replay,
                local_mismatch=local_mismatch,
                reference=reference_signature,
                candidate=candidate.signature,
            )
            final_loss = candidate.loss

        assert final_loss is not None
        return final_loss
