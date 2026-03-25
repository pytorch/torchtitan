# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn


def canonical_fqn(
    name: str,
    prefixes: tuple[str, ...] = ("_orig_mod", "_checkpoint_wrapped_module"),
) -> str:
    """Strip any matching segment from a dotted name path.

    Handles wrapper prefixes that can appear at any level of the module
    hierarchy, e.g. 'layers.0._orig_mod.attention.wq.weight',
    '_orig_mod.layers.0.weight', or
    'layers.0._checkpoint_wrapped_module.attention.wq.weight'.
    """
    parts = name.split(".")
    cleaned = [p for p in parts if p not in prefixes]
    return ".".join(cleaned)


def canonical_model_state_dict(
    model: nn.Module,
    prefixes: tuple[str, ...] = ("_orig_mod", "_checkpoint_wrapped_module"),
    **kwargs: Any,
) -> dict[str, Any]:
    """Get model state dict with wrapper prefixes stripped from keys.

    Calls model.state_dict(**kwargs) and applies canonical_fqn to every key.
    """
    return {
        canonical_fqn(k, prefixes): v for k, v in model.state_dict(**kwargs).items()
    }


def load_canonical_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    prefixes: tuple[str, ...] = ("_orig_mod", "_checkpoint_wrapped_module"),
    **kwargs: Any,
) -> None:
    """Load a canonical-keyed state dict into a model that may have wrapper prefixes.

    Builds {clean_key: orig_key} mapping from the model's current state dict keys,
    remaps the clean checkpoint keys back to the model's expected keys (which may
    include _orig_mod.), and calls model.load_state_dict(remapped, **kwargs).
    """
    strict = kwargs.get("strict", True)

    clean_to_orig: dict[str, str] = {}
    for key in model.state_dict(keep_vars=True):
        clean_to_orig[canonical_fqn(key, prefixes)] = key

    remapped: dict[str, Any] = {}
    unexpected_keys: list[str] = []
    for clean_key, value in state_dict.items():
        if clean_key in clean_to_orig:
            remapped[clean_to_orig[clean_key]] = value
        else:
            unexpected_keys.append(clean_key)

    if strict:
        if unexpected_keys:
            raise KeyError(
                f"State dict contains {len(unexpected_keys)} key(s) that do not match "
                f"any model parameter (after cleaning prefixes {prefixes}): "
                f"{unexpected_keys}"
            )
        missing_keys = sorted(k for k in clean_to_orig if k not in state_dict)
        if missing_keys:
            raise KeyError(
                f"State dict is missing {len(missing_keys)} model parameter(s) "
                f"(after cleaning prefixes {prefixes}): {missing_keys}"
            )

    model.load_state_dict(remapped, **kwargs)


def init_optim_state(optim: torch.optim.Optimizer) -> None:
    """Initialize optim states by calling step() with zero grads.

    Copied from torch.distributed.checkpoint.state_dict._init_optim_state
    to remove the import dependency on PyTorch's private API.
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

    # Some optimizers will update parameters regardless of grads due to lr, so
    # make lr to zero when calling step().
    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = (
                torch.tensor(0.0)
                if isinstance(param_group["lr"], torch.Tensor)
                else 0.0
            )
    optim.step(closure=None)
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def optim_state_dict_to_fqn_keys(optim_sd: dict[str, Any]) -> dict[str, Any]:
    """Convert integer-keyed optimizer state dict to FQN-keyed.

    Uses param_names from param_groups (set when constructing optimizer with
    (name, param) tuples) to map integer param IDs to FQN strings.
    Removes param_names from output param_groups.
    """
    id_to_fqn: dict[int, str] = {}
    new_param_groups: list[dict[str, Any]] = []

    for pg in optim_sd["param_groups"]:
        if "param_names" not in pg:
            raise ValueError(
                "Optimizer must be constructed with (name, param) tuples "
                "to support FQN-keyed state dicts, but param_names is missing "
                "from param_groups."
            )
        param_names = pg["param_names"]
        params = pg["params"]
        fqn_params = []
        for int_id, fqn in zip(params, param_names):
            id_to_fqn[int_id] = fqn
            fqn_params.append(fqn)
        new_group = {k: v for k, v in pg.items() if k != "param_names"}
        new_group["params"] = fqn_params
        new_param_groups.append(new_group)

    new_state: dict[str, Any] = {}
    for int_id, state_val in optim_sd["state"].items():
        if int_id not in id_to_fqn:
            raise KeyError(
                f"Optimizer state contains param ID {int_id} that is not in any "
                f"param_group. Known param IDs: {sorted(id_to_fqn.keys())}"
            )
        new_state[id_to_fqn[int_id]] = state_val

    return {"state": new_state, "param_groups": new_param_groups}


def _flatten_state_nested(d: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
    """Recursively flatten a nested dict with dot-separated keys."""
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_state_nested(v, key, out)
        else:
            out[key] = v


def flatten_optim_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """Flatten an FQN-keyed optimizer state dict into dot-separated keys.

    Input format:
        {"state": {fqn: {state_key: value, ...}, ...},
         "param_groups": [{"params": [fqn, ...], key: value, ...}, ...]}

    Output format:
        {"state.{fqn}.{state_key}": value, ...,
         "param_groups.{fqn}.{pg_key}": value, ...}

    Does NOT mutate the input.
    """
    ret: dict[str, Any] = {}

    for fqn, state in sd["state"].items():
        _flatten_state_nested(state, f"state.{fqn}", ret)

    for pg in sd["param_groups"]:
        for fqn in pg["params"]:
            for k, v in pg.items():
                if k != "params":
                    ret[f"param_groups.{fqn}.{k}"] = v

    return ret


def _reconstruct_nested(flat_sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Reconstruct a nested dict from flattened keys with the given prefix."""
    result: dict[str, Any] = {}
    prefix_dot = prefix + "."
    for key, val in flat_sd.items():
        if key.startswith(prefix_dot):
            rest = key[len(prefix_dot) :]
            parts = rest.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = val
    return result


def unflatten_and_load_optim_state_dict(
    optim: torch.optim.Optimizer, flat_sd: dict[str, Any]
) -> None:
    """Unflatten a flat optimizer state dict and load it into the optimizer.

    Reads param_names from the optimizer's param_groups to identify which keys
    belong to this optimizer. Reconstructs integer-keyed state dict and calls
    optim.load_state_dict().
    """
    state: dict[int, dict[str, Any]] = {}
    param_groups: list[dict[str, Any]] = []
    param_idx = 0

    for pg in optim.param_groups:
        names = pg["param_names"]
        params = pg["params"]
        int_params: list[int] = []

        for name, param in zip(names, params):
            int_params.append(param_idx)

            # Extract state for this param using known state keys from the
            # live optimizer (init_optim_state must have been called first).
            if param in optim.state:
                param_state: dict[str, Any] = {}
                for state_name in optim.state[param]:
                    flat_key = f"state.{name}.{state_name}"
                    if flat_key in flat_sd:
                        param_state[state_name] = flat_sd[flat_key]
                    else:
                        # Handle nested dict case (e.g., Shampoo optimizer)
                        nested = _reconstruct_nested(flat_sd, flat_key)
                        if nested:
                            param_state[state_name] = nested
                if param_state:
                    state[param_idx] = param_state

            param_idx += 1

        # Extract param_group hyperparameters from the first param's keys
        if not names:
            param_groups.append({"params": int_params})
            continue
        first_name = names[0]
        new_pg: dict[str, Any] = {"params": int_params}
        for k in pg:
            if k in ("params", "param_names"):
                continue
            flat_key = f"param_groups.{first_name}.{k}"
            if flat_key in flat_sd:
                new_pg[k] = flat_sd[flat_key]
            else:
                raise KeyError(
                    f"Optimizer param group key {k!r} not found in checkpoint "
                    f"(looked up via param {first_name!r})."
                )

        param_groups.append(new_pg)

    optim.load_state_dict({"state": state, "param_groups": param_groups})


def get_flat_optim_state_dict(optim: torch.optim.Optimizer) -> dict[str, Any]:
    """Get a flat FQN-keyed optimizer state dict ready for DCP.

    Initializes optimizer state if needed, converts integer param IDs to FQN
    strings, and flattens the result into dot-separated keys.
    """
    init_optim_state(optim)
    fqn_sd = optim_state_dict_to_fqn_keys(optim.state_dict())
    return flatten_optim_state_dict(fqn_sd)


def load_flat_optim_state_dict(
    optim: torch.optim.Optimizer, flat_sd: dict[str, Any]
) -> None:
    """Load a flat FQN-keyed optimizer state dict into the optimizer.

    Initializes optimizer state if needed and unflattens the flat dict back
    into the optimizer's native format before loading.
    """
    init_optim_state(optim)
    unflatten_and_load_optim_state_dict(optim, flat_sd)
