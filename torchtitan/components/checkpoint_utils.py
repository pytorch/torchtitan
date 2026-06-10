# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal state-dict helpers for distributed checkpointing.

These utilities are a focused replacement for the parts of
``torch.distributed.checkpoint.state_dict`` that TorchTitan actually needs.
The upstream APIs (``get_model_state_dict``, ``set_optimizer_state_dict``, ...)
carry a lot of complexity from legacy FSDP1/DDP support and many features
that do not apply to TorchTitan.

The helpers are:

- ``canonical_fqn`` strips the activation-checkpoint wrapper segment
  (``_checkpoint_wrapped_module``) from a single FQN. The model state dict is
  already canonical -- ``torch.compile`` is applied in place (no segment) and the
  activation-checkpoint wrapper strips its own segment via a state_dict hook -- but
  ``nn.Module.named_parameters()`` is not, so optimizer construction uses this to
  produce FQN-keyed optimizer state that matches the model keys.
- ``get_flat_optim_state_dict`` / ``load_flat_optim_state_dict`` convert between
  an optimizer's native (integer-indexed) state dict and a flat, FQN-keyed dict
  that DCP can save and reshard. ``init_optim_state`` materializes optimizer
  state and is a precondition for both.
"""

from typing import Any

import torch

__all__ = [
    "canonical_fqn",
    "init_optim_state",
    "get_flat_optim_state_dict",
    "load_flat_optim_state_dict",
]

# Segment inserted by the activation checkpoint wrapper (checkpoint_wrapper) in
# named_parameters(). It can appear at any level of the FQN and is not part of the
# canonical model contract. torch.compile is applied in place and adds no segment.
_WRAPPER_PREFIXES: tuple[str, ...] = ("_checkpoint_wrapped_module",)


def canonical_fqn(name: str, prefixes: tuple[str, ...] = _WRAPPER_PREFIXES) -> str:
    """Strip wrapper segments from a dotted FQN.

    A segment may appear at any level, e.g.
    ``layers.0._checkpoint_wrapped_module.attention.wq.weight`` ->
    ``layers.0.attention.wq.weight``.
    """
    return ".".join(p for p in name.split(".") if p not in prefixes)


def init_optim_state(optim: torch.optim.Optimizer) -> None:
    """Materialize optimizer state by running one zero-gradient, zero-lr step.

    Optimizers create their state (e.g. Adam's ``exp_avg``) lazily on the first
    ``step()``. DCP needs the state tensors to exist before save (to read them)
    and before load (to load into them). This runs a step with zero gradients
    and ``lr=0`` so parameters are untouched, then restores ``lr``.

    No-op if state already exists or any gradient is set, so it is safe to call
    repeatedly and never disturbs an in-progress training step.

    Adapted from ``torch.distributed.checkpoint.state_dict._init_optim_state`` to
    drop the dependency on that private API.
    """
    if optim.state:
        return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    # Some optimizers update parameters regardless of gradients due to lr, so set
    # lr to zero before stepping to keep parameters unchanged.
    saved_lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            saved_lrs.append(param_group["lr"])
            param_group["lr"] = (
                torch.tensor(0.0)
                if isinstance(param_group["lr"], torch.Tensor)
                else 0.0
            )
    optim.step(closure=None)
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = saved_lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def get_flat_optim_state_dict(optim: torch.optim.Optimizer) -> dict[str, Any]:
    """Return a flat, FQN-keyed optimizer state dict ready for DCP.

    Output keys are ``state.{fqn}.{state_name}`` and ``param_groups.{fqn}.{key}``.
    The flat layout avoids the integer ``param_group`` index collisions that break
    pipeline-parallel checkpoints (multiple chunks reusing index 0).

    The optimizer state must already exist; call ``init_optim_state`` first.
    """
    fqn_sd = _optim_state_dict_to_fqn_keys(optim.state_dict())

    flat: dict[str, Any] = {}
    for fqn, state in fqn_sd["state"].items():
        _flatten_state_nested(state, f"state.{fqn}", flat)
    for param_group in fqn_sd["param_groups"]:
        for fqn in param_group["params"]:
            for key, value in param_group.items():
                if key != "params":
                    flat[f"param_groups.{fqn}.{key}"] = value
    return flat


def load_flat_optim_state_dict(
    optim: torch.optim.Optimizer, flat_sd: dict[str, Any]
) -> None:
    """Load a flat, FQN-keyed optimizer state dict into ``optim``.

    Inverse of ``get_flat_optim_state_dict``. The optimizer state must already
    exist (it tells us which state tensors to expect); call ``init_optim_state``
    first. Keys in ``flat_sd`` that this optimizer does not own are ignored, so a
    single flat dict covering several optimizers can be passed to each of them.
    """
    optim.load_state_dict(_unflatten_optim_state_dict(optim, flat_sd))


def _optim_state_dict_to_fqn_keys(optim_sd: dict[str, Any]) -> dict[str, Any]:
    """Re-key an optimizer state dict from integer param ids to FQNs.

    Relies on ``param_names`` in each param group, which PyTorch populates when
    the optimizer is built with ``(name, param)`` tuples. ``param_names`` is
    dropped from the result.
    """
    id_to_fqn: dict[int, str] = {}
    new_param_groups: list[dict[str, Any]] = []
    for param_group in optim_sd["param_groups"]:
        if "param_names" not in param_group:
            raise ValueError(
                "Optimizer must be built with (name, param) tuples so that "
                "param_names is available for FQN-keyed state dicts."
            )
        fqns = param_group["param_names"]
        for param_id, fqn in zip(param_group["params"], fqns):
            id_to_fqn[param_id] = fqn
        new_group = {k: v for k, v in param_group.items() if k != "param_names"}
        new_group["params"] = list(fqns)
        new_param_groups.append(new_group)

    new_state: dict[str, Any] = {}
    for param_id, state in optim_sd["state"].items():
        if param_id not in id_to_fqn:
            raise KeyError(
                f"Optimizer state has param id {param_id} that is not in any "
                f"param group. Known ids: {sorted(id_to_fqn)}"
            )
        new_state[id_to_fqn[param_id]] = state

    return {"state": new_state, "param_groups": new_param_groups}


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer, flat_sd: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild an integer-keyed optimizer state dict from a flat, FQN-keyed one.

    Walks the live optimizer's param groups to recover the FQN order and the set
    of state tensors to expect, then pulls matching values out of ``flat_sd``.
    """
    state: dict[int, dict[str, Any]] = {}
    param_groups: list[dict[str, Any]] = []
    param_id = 0
    for param_group in optim.param_groups:
        fqns = param_group["param_names"]
        params = param_group["params"]
        ids: list[int] = []
        for fqn, param in zip(fqns, params):
            ids.append(param_id)
            if param in optim.state:
                param_state: dict[str, Any] = {}
                for state_name in optim.state[param]:
                    flat_key = f"state.{fqn}.{state_name}"
                    if flat_key in flat_sd:
                        param_state[state_name] = flat_sd[flat_key]
                    else:
                        # State value is itself a nested dict (e.g. Shampoo).
                        nested = _reconstruct_nested(flat_sd, flat_key)
                        if nested:
                            param_state[state_name] = nested
                if param_state:
                    state[param_id] = param_state
            param_id += 1

        if not fqns:
            param_groups.append({"params": ids})
            continue
        new_group: dict[str, Any] = {"params": ids}
        for key in param_group:
            if key in ("params", "param_names"):
                continue
            flat_key = f"param_groups.{fqns[0]}.{key}"
            if flat_key not in flat_sd:
                raise KeyError(
                    f"Optimizer param group key {key!r} not found in checkpoint "
                    f"(looked up via param {fqns[0]!r})."
                )
            new_group[key] = flat_sd[flat_key]
        param_groups.append(new_group)

    return {"state": state, "param_groups": param_groups}


def _flatten_state_nested(
    state: dict[str, Any], prefix: str, out: dict[str, Any]
) -> None:
    """Flatten a (possibly nested) per-param state dict into dotted keys."""
    for key, value in state.items():
        flat_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            _flatten_state_nested(value, flat_key, out)
        else:
            out[flat_key] = value


def _reconstruct_nested(flat_sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Rebuild the nested dict stored under ``prefix`` in a flat state dict."""
    result: dict[str, Any] = {}
    prefix_dot = prefix + "."
    for key, value in flat_sd.items():
        if not key.startswith(prefix_dot):
            continue
        current = result
        parts = key[len(prefix_dot) :].split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result
