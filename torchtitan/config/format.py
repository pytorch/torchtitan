# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compact human-readable formatting for TorchTitan configuration trees.

Large models expand into dozens of near-identical per-layer config entries
(e.g. ``model.layers``) and per-layer optimizer maps (e.g.
``compute_sharding_by_fqn``). This module folds structurally equivalent
sequence elements and indexed dict keys while preserving concrete config class
boundaries, small per-layer overrides, and each caller's native output syntax
(Python dataclass repr for ``--print-config`` and JSON for ``maybe_log``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from torchtitan.config.configurable import Configurable

_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")
_MAX_EXPLICIT_VARIANTS = 3
_INDENT = "  "


@dataclass(frozen=True, slots=True)
class _Record:
    type_name: str
    fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _LeafRepr:
    text: str


@dataclass(frozen=True, slots=True)
class _FoldedSequence:
    groups: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _IndexedValues:
    by_range: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _VaryingValue:
    count: int
    first_idx: int
    first_val: Any
    last_idx: int
    last_val: Any


def _normalize_addrs(text: str) -> str:
    return _HEX_ADDR_RE.sub(">", text)


def _format_run(start: int, end: int) -> str:
    if end == start:
        return str(start)
    if end == start + 1:
        return f"{start},{end}"
    return f"{start}..{end}"


def _format_index_runs(indices: list[int], *, wrap: tuple[str, str]) -> str:
    parts: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        parts.append(_format_run(start, prev))
        start = prev = idx
    parts.append(_format_run(start, prev))
    open_ch, close_ch = wrap
    sep = ", " if open_ch == "[" else ","
    return f"{open_ch}{sep.join(parts)}{close_ch}"


def _is_canonical_index(segment: str) -> bool:
    return (
        segment.isascii()
        and segment.isdigit()
        and (segment == "0" or not segment.startswith("0"))
    )


def _uses_custom_leaf_to_dict(val: Any) -> bool:
    if not hasattr(val, "to_dict"):
        return False
    if isinstance(val, Configurable.Config):
        if type(val).to_dict is Configurable.Config.to_dict:
            return False
        # Container configs such as Trainer.Config override to_dict() only to
        # walk fields; keep walking fields so nested Config types stay intact.
        return not any(
            isinstance(getattr(val, f.name), Configurable.Config) for f in fields(val)
        )
    return True


def _to_node(val: Any, *, json_mode: bool) -> Any:
    if isinstance(val, str):
        return _normalize_addrs(val)
    if isinstance(val, (int, float, bool, type(None))):
        return val
    if isinstance(val, dict):
        return {k: _to_node(v, json_mode=json_mode) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_to_node(v, json_mode=json_mode) for v in val)
    if is_dataclass(val) and not isinstance(val, type):
        if _uses_custom_leaf_to_dict(val):
            raw = val.to_dict()
            if json_mode and isinstance(raw, dict):
                return _Record(
                    type(val).__qualname__,
                    {k: _to_node(v, json_mode=True) for k, v in raw.items()},
                )
            if not json_mode and isinstance(raw, dict) and set(raw.keys()) == {"repr"}:
                return _LeafRepr(_normalize_addrs(str(raw["repr"])))
        skip_private = isinstance(val, Configurable.Config)
        field_values = {
            f.name: _to_node(getattr(val, f.name), json_mode=json_mode)
            for f in fields(val)
            if (not skip_private or not f.name.startswith("_"))
            and (json_mode or f.repr)
        }
        return _Record(type(val).__qualname__, field_values)
    if json_mode and hasattr(val, "to_dict"):
        return _to_node(val.to_dict(), json_mode=True)
    return _LeafRepr(_normalize_addrs(repr(val)))


def _shape_signature(node: Any) -> Any:
    if isinstance(node, _Record):
        return (
            "record",
            node.type_name,
            tuple((k, _shape_signature(v)) for k, v in node.fields.items()),
        )
    if isinstance(node, dict):
        return ("dict", tuple((k, _shape_signature(v)) for k, v in node.items()))
    if isinstance(node, (list, tuple)):
        return (
            type(node).__name__,
            len(node),
            tuple(_shape_signature(v) for v in node),
        )
    if isinstance(node, _LeafRepr):
        return ("leaf_repr",)
    return ("leaf", type(node).__name__)


def _value_key(node: Any) -> Any:
    if isinstance(node, _Record):
        return (
            "record",
            node.type_name,
            tuple((k, _value_key(v)) for k, v in node.fields.items()),
        )
    if isinstance(node, _LeafRepr):
        return ("leaf_repr", node.text)
    if isinstance(node, _FoldedSequence):
        return ("folded_seq", tuple((k, _value_key(v)) for k, v in node.groups.items()))
    if isinstance(node, _IndexedValues):
        return (
            "indexed_vals",
            tuple((k, _value_key(v)) for k, v in node.by_range.items()),
        )
    if isinstance(node, _VaryingValue):
        return (
            "varying",
            node.count,
            node.first_idx,
            _value_key(node.first_val),
            node.last_idx,
            _value_key(node.last_val),
        )
    if isinstance(node, dict):
        return ("dict", tuple((k, _value_key(v)) for k, v in node.items()))
    if isinstance(node, (list, tuple)):
        return (type(node).__name__, tuple(_value_key(v) for v in node))
    return (type(node).__name__, node)


def _merge_group(items: list[Any], indices: list[int]) -> Any:
    first = items[0]
    if isinstance(first, _Record):
        return _Record(
            first.type_name,
            {
                k: _merge_group([it.fields[k] for it in items], indices)
                for k in first.fields
            },
        )
    if isinstance(first, dict):
        return {k: _merge_group([it[k] for it in items], indices) for k in first}
    if isinstance(first, (list, tuple)):
        merged = [
            _merge_group([it[j] for it in items], indices) for j in range(len(first))
        ]
        return type(first)(merged)

    by_val: dict[Any, list[int]] = defaultdict(list)
    unique_vals: list[tuple[Any, Any]] = []
    for idx, val in zip(indices, items):
        key = _value_key(val)
        if key not in by_val:
            unique_vals.append((key, val))
        by_val[key].append(idx)

    if len(unique_vals) == 1:
        return unique_vals[0][1]
    if len(unique_vals) <= _MAX_EXPLICIT_VARIANTS:
        return _IndexedValues(
            {
                _format_index_runs(by_val[key], wrap=("[", "]")): val
                for key, val in unique_vals
            }
        )
    return _VaryingValue(
        count=len(items),
        first_idx=indices[0],
        first_val=items[0],
        last_idx=indices[-1],
        last_val=items[-1],
    )


def _fold_indexed_dict(mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Fold dotted string keys with identical values (e.g. layers.{0..42}.wq)."""
    if len(mapping) < 2 or not all(isinstance(k, str) for k in mapping):
        return mapping

    items: list[tuple[list[str], Any, Any]] = [
        (key.split("."), val, _value_key(val)) for key, val in mapping.items()
    ]
    while len(items) > 1:
        max_numeric = max(
            (
                sum(1 for seg in segs if _is_canonical_index(seg))
                for segs, _, _ in items
            ),
            default=0,
        )
        if max_numeric == 0:
            break
        merged = False
        for from_right in range(1, max_numeric + 1):
            buckets: dict[
                tuple[tuple[str, ...], tuple[str, ...], Any],
                list[tuple[int, int, list[str], Any, Any]],
            ] = defaultdict(list)
            emitted: list[tuple[int, list[str], Any, Any]] = []
            for pos, (segs, val, vkey) in enumerate(items):
                num_positions = [
                    i for i, seg in enumerate(segs) if _is_canonical_index(seg)
                ]
                if len(num_positions) < from_right:
                    emitted.append((pos, segs, val, vkey))
                    continue
                idx_pos = num_positions[-from_right]
                template = (tuple(segs[:idx_pos]), tuple(segs[idx_pos + 1 :]), vkey)
                buckets[template].append((pos, int(segs[idx_pos]), segs, val, vkey))

            changed = False
            for (prefix, suffix, _), entries in buckets.items():
                if len(entries) >= 2 and len({e[1] for e in entries}) == len(entries):
                    changed = True
                    folded_seg = _format_index_runs(
                        sorted(e[1] for e in entries), wrap=("{", "}")
                    )
                    emitted.append(
                        (
                            entries[0][0],
                            [*prefix, folded_seg, *suffix],
                            entries[0][3],
                            entries[0][4],
                        )
                    )
                else:
                    for pos, _, segs, val, vkey in entries:
                        emitted.append((pos, segs, val, vkey))
            if changed:
                emitted.sort(key=lambda entry: entry[0])
                items = [(segs, val, vkey) for _, segs, val, vkey in emitted]
                merged = True
                break
        if not merged:
            break
    return {".".join(segs): val for segs, val, _ in items}


def _compact(node: Any) -> Any:
    if isinstance(node, _Record):
        return _Record(node.type_name, {k: _compact(v) for k, v in node.fields.items()})
    if isinstance(node, dict):
        compacted = {k: _compact(v) for k, v in node.items()}
        return _fold_indexed_dict(compacted)
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(x, (_Record, dict)) for x in node):
            groups: dict[Any, list[tuple[int, Any]]] = defaultdict(list)
            order: list[Any] = []
            for idx, item in enumerate(node):
                sig = _shape_signature(item)
                if sig not in groups:
                    order.append(sig)
                groups[sig].append((idx, item))
            if any(len(groups[sig]) >= 2 for sig in order):
                folded_groups: dict[str, Any] = {}
                for sig in order:
                    entries = groups[sig]
                    indices = [i for i, _ in entries]
                    values = [v for _, v in entries]
                    range_label = _format_index_runs(indices, wrap=("[", "]"))
                    key = (
                        f"{range_label} ({len(indices)} items)"
                        if len(indices) > 1
                        else range_label
                    )
                    merged = (
                        _merge_group(values, indices) if len(indices) > 1 else values[0]
                    )
                    folded_groups[key] = _compact(merged)
                return _FoldedSequence(folded_groups)
        return type(node)(_compact(v) for v in node)
    return node


def _format_varying(node: _VaryingValue, *, python_repr: bool) -> str:
    def _val_str(val: Any) -> str:
        if isinstance(val, _LeafRepr):
            return val.text if python_repr else repr(val.text)
        return repr(val)

    return (
        f"<varies across {node.count} items: "
        f"[{node.first_idx}]={_val_str(node.first_val)}, ..., "
        f"[{node.last_idx}]={_val_str(node.last_val)}>"
    )


def _to_json_obj(node: Any) -> Any:
    if isinstance(node, _Record):
        return {k: _to_json_obj(v) for k, v in node.fields.items()}
    if isinstance(node, _LeafRepr):
        return node.text
    if isinstance(node, _FoldedSequence):
        return {k: _to_json_obj(v) for k, v in node.groups.items()}
    if isinstance(node, _IndexedValues):
        return {k: _to_json_obj(v) for k, v in node.by_range.items()}
    if isinstance(node, _VaryingValue):
        return _format_varying(node, python_repr=False)
    if isinstance(node, dict):
        return {
            (
                k if isinstance(k, (str, int, float, bool, type(None))) else str(k)
            ): _to_json_obj(v)
            for k, v in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [_to_json_obj(v) for v in node]
    return node


def _render_python(node: Any, level: int = 0) -> str:
    pad = _INDENT * level
    child_pad = _INDENT * (level + 1)

    if isinstance(node, _LeafRepr):
        return node.text
    if isinstance(node, _VaryingValue):
        return _format_varying(node, python_repr=True)
    if isinstance(node, _Record):
        if not node.fields:
            return f"{node.type_name}()"
        lines = [
            f"{child_pad}{k}={_render_python(v, level + 1)},"
            for k, v in node.fields.items()
        ]
        return f"{node.type_name}(\n" + "\n".join(lines) + f"\n{pad})"
    if isinstance(node, _FoldedSequence):
        lines = [
            f"{child_pad}{k!r}: {_render_python(v, level + 1)},"
            for k, v in node.groups.items()
        ]
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(node, _IndexedValues):
        lines = [
            f"{child_pad}{k!r}: {_render_python(v, level + 1)},"
            for k, v in node.by_range.items()
        ]
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(node, dict):
        if not node:
            return "{}"
        lines = [
            f"{child_pad}{k!r}: {_render_python(v, level + 1)},"
            for k, v in node.items()
        ]
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(node, (list, tuple)):
        open_ch, close_ch = ("[", "]") if isinstance(node, list) else ("(", ")")
        if not node:
            return f"{open_ch}{close_ch}"
        rendered = [_render_python(v, level + 1) for v in node]
        if all("\n" not in item for item in rendered):
            inline = ", ".join(rendered)
            if isinstance(node, tuple) and len(rendered) == 1:
                inline += ","
            if len(pad) + len(inline) + 2 <= 88:
                return f"{open_ch}{inline}{close_ch}"
        lines = [f"{child_pad}{item}," for item in rendered]
        return f"{open_ch}\n" + "\n".join(lines) + f"\n{pad}{close_ch}"
    return repr(node)


def format_config(config: Any) -> str:
    """Format a config tree in compact Python dataclass-repr style."""
    compacted = _compact(_to_node(config, json_mode=False))
    return _render_python(compacted)


def format_config_json(config: Any) -> str:
    """Format a config tree as compact indented JSON."""
    compacted = _compact(_to_node(config, json_mode=True))
    return json.dumps(_to_json_obj(compacted), indent=2, ensure_ascii=False)
