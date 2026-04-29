# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only graph pass that lists all activations saved for backward."""

from __future__ import annotations

import operator
import re
from collections import defaultdict

import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.passes import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
)
from torchtitan.tools.logging import logger

_PASSTHROUGH_TARGETS = frozenset(
    {
        operator.getitem,
        torch.ops._c10d_functional.wait_tensor.default,
    }
)


def _resolve_producing_op(node: torch.fx.Node) -> torch.fx.Node:
    """Walk through getitem/wait_tensor to find the real producing op."""
    while node.target in _PASSTHROUGH_TARGETS:
        parent = node.args[0]
        if not isinstance(parent, torch.fx.Node) or parent.op != "call_function":
            break
        node = parent
    return node


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_shape(val: object) -> str:
    if isinstance(val, torch.Tensor):
        return str(list(val.shape))
    if isinstance(val, (tuple, list)):
        parts = [_format_shape(v) for v in val]
        return "(" + ", ".join(parts) + ")"
    return "?"


def _tensor_nbytes(val: object) -> int:
    if isinstance(val, torch.Tensor):
        try:
            # int() can fail on symbolic shapes (e.g. MoE token dispatch)
            return int(val.numel() * val.element_size())
        except Exception:
            return 0
    if isinstance(val, (tuple, list)):
        return sum(_tensor_nbytes(v) for v in val)
    return 0


def _tensor_dtype(val: object) -> str:
    if isinstance(val, torch.Tensor):
        return str(val.dtype).replace("torch.", "")
    if isinstance(val, (tuple, list)):
        dtypes = {_tensor_dtype(v) for v in val} - {"?"}
        if len(dtypes) == 1:
            return dtypes.pop()
        if dtypes:
            return "/".join(sorted(dtypes))
    return "?"


def _format_bytes(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GiB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.2f} MiB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.2f} KiB"
    return f"{nbytes} B"


_REPO_ROOT = "torchtitan/"


def _parse_stack_frames(
    stack_trace: str,
) -> list[tuple[str, str, str]]:
    raw_lines = stack_trace.strip().splitlines()
    frames = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if line.startswith("File "):
            parts = line.split(", ")
            path = parts[0].removeprefix("File ").strip('"')
            lineno = parts[1].removeprefix("line ").strip() if len(parts) >= 2 else "?"
            code = ""
            if i + 1 < len(raw_lines) and not raw_lines[i + 1].strip().startswith(
                "File "
            ):
                code = raw_lines[i + 1].strip()
                i += 1
            frames.append((path, lineno, code))
        i += 1
    return frames


def _format_stack_trace(stack_trace: str | None) -> str:
    """Extract the last torchtitan frame, falling back to the last frame."""
    if not stack_trace:
        return ""
    frames = _parse_stack_frames(stack_trace)
    if not frames:
        return ""

    chosen = None
    for path, lineno, code in reversed(frames):
        if _REPO_ROOT in path:
            chosen = (path, lineno, code)
            break
    if chosen is None:
        chosen = frames[-1]

    path, lineno, code = chosen
    idx = path.find(_REPO_ROOT)
    if idx >= 0:
        path = path[idx + len(_REPO_ROOT) :]
    else:
        path = path.rsplit("/", 1)[-1]

    loc = f"{path}:{lineno}"
    if code:
        return f"{loc}  {code}"
    return loc


_POLICY_SHORT_NAMES = {
    CheckpointPolicy.MUST_SAVE: "SAVE",
    CheckpointPolicy.PREFER_RECOMPUTE: "RECOMPUTE",
    CheckpointPolicy.MUST_RECOMPUTE: "RECOMPUTE!",
    CheckpointPolicy.MUST_CPU_OFFLOAD: "OFFLOAD",
}


def _format_policy(node: torch.fx.Node) -> str:
    policy = node.meta.get("recompute")
    if policy is None:
        # Untagged activations are kept in GPU memory by default.
        return _POLICY_SHORT_NAMES[CheckpointPolicy.MUST_SAVE]
    return _POLICY_SHORT_NAMES.get(policy, str(policy))


def _format_original_aten(node: torch.fx.Node) -> str:
    original = node.meta.get("original_aten")
    if original is None or str(original) == str(node.target):
        return ""
    return str(original)


# ---------------------------------------------------------------------------
# Layer consolidation helpers
# ---------------------------------------------------------------------------


# Matches PyTorch symbolic shape variables (s0, u12, etc.)
_SYM_VAR_RE = re.compile(r"\b[su]\d+\b")


def _normalize_shape(shape: str) -> str:
    return _SYM_VAR_RE.sub("S", shape)


def _layer_fingerprint(activations: list[dict]) -> tuple:
    """Return a hashable fingerprint for a layer's activation pattern."""
    return tuple(
        (
            act["target"],
            _normalize_shape(act["shape"]),
            act["dtype"],
            act["policy"],
            act["original_aten"],
        )
        for act in activations
    )


def _strip_layer_prefix(fqn: str) -> str:
    """Strip ``layers.<N>.`` prefix to get the relative submodule path."""
    parts = fqn.split(".", 2)
    if parts[0] == "layers" and len(parts) >= 3:
        return parts[2]
    return fqn


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------


def log_activation_memory_policy(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """List all activation nodes whose outputs are consumed by backward nodes.

    Layers with identical activation patterns (same ops, shapes, and dtypes)
    are consolidated and printed once. Non-layer nodes are always printed
    individually.
    This is a read-only pass — the graph is not modified.
    """
    activations_by_layer: dict[int, list[dict]] = defaultdict(list)

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_backward_node(node):
            continue

        num_bwd_users = sum(1 for u in node.users if _is_backward_node(u))
        if num_bwd_users == 0:
            continue

        layer_id = _get_layer_id(node)
        val = node.meta.get("val")
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")

        producer = _resolve_producing_op(node)

        activations_by_layer[layer_id].append(
            {
                "name": node.name,
                "target": str(producer.target),
                "shape": _format_shape(val),
                "dtype": _tensor_dtype(val),
                "nbytes": _tensor_nbytes(val),
                "policy": _format_policy(node),
                "fqn": fqn,
                "original_aten": _format_original_aten(producer),
                "source": _format_stack_trace(producer.meta.get("stack_trace")),
            }
        )

    # Group numbered layers by fingerprint
    fingerprint_to_layers: dict[tuple, list[int]] = defaultdict(list)
    for layer_id, acts in sorted(activations_by_layer.items()):
        if layer_id == _NOT_IN_LAYERS:
            continue
        fingerprint_to_layers[_layer_fingerprint(acts)].append(layer_id)

    def _fmt_target(act: dict) -> str:
        aten = act["original_aten"]
        if aten:
            return f"{act['target']}  (← {aten})"
        return act["target"]

    def _fmt_row(idx: int, act: dict, submodule: str) -> str:
        return (
            f"  {idx:<3} {_format_bytes(act['nbytes']):>10}  {act['dtype']:<10} "
            f"{act['policy']:<12} {act['shape']:<25} {submodule:<30} "
            f"{_fmt_target(act):<55} {act['source']}"
        )

    header = (
        f"  {'#':<3} {'Memory':>10}  {'DType':<10} "
        f"{'Policy':<12} {'Shape':<25} "
        f"{'SubModule':<30} {'Target':<55} {'Source'}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    total_activations = 0
    total_bytes = 0

    # Print non-layer activations
    if _NOT_IN_LAYERS in activations_by_layer:
        other_acts = activations_by_layer[_NOT_IN_LAYERS]
        other_bytes = sum(act["nbytes"] for act in other_acts)
        lines.append(
            f"[non-layer] ({len(other_acts)} activations, {_format_bytes(other_bytes)})"
        )
        for i, act in enumerate(other_acts):
            lines.append(_fmt_row(i, act, act["fqn"]))
            total_activations += 1
            total_bytes += act["nbytes"]
        lines.append("")

    # Print consolidated layer groups
    for fingerprint, layer_ids in sorted(
        fingerprint_to_layers.items(), key=lambda kv: kv[1][0]
    ):
        representative = activations_by_layer[layer_ids[0]]
        num_layers = len(layer_ids)
        layer_bytes = sum(act["nbytes"] for act in representative)

        if num_layers == 1:
            label = f"[layer {layer_ids[0]}]"
        else:
            first, last = layer_ids[0], layer_ids[-1]
            label = f"[layers {first}-{last}] (x{num_layers}, same pattern)"

        lines.append(
            f"{label} ({len(representative)} activations, "
            f"{_format_bytes(layer_bytes)} each, "
            f"{_format_bytes(layer_bytes * num_layers)} total)"
        )
        for i, act in enumerate(representative):
            sub = _strip_layer_prefix(act["fqn"])
            lines.append(_fmt_row(i, act, sub))
        total_activations += len(representative) * num_layers
        total_bytes += layer_bytes * num_layers
        lines.append("")

    num_layers = sum(len(ids) for ids in fingerprint_to_layers.values())
    has_non_layer = _NOT_IN_LAYERS in activations_by_layer
    scope = f"{num_layers} layer(s)" + (" + non-layer" if has_non_layer else "")
    lines.append(sep)
    lines.append(
        f"Total: {total_activations} activations, "
        f"{_format_bytes(total_bytes)} across {scope}"
    )
    lines.append(sep)

    logger.info(
        "Activation listing (forward nodes with backward consumers):\n"
        + "\n".join(lines)
    )

    return gm
