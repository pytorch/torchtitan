# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compact, lossless display formatting for TorchTitan configuration trees.

Large models expand into dozens of near-identical per-layer entries
(e.g. ``model.layers``) and per-layer optimizer maps (e.g.
``compute_sharding_by_fqn``). This module flattens a live config tree into
dotted paths (recording concrete dataclass types under ``.__class__`` so
sibling ``Configurable.Config`` subclasses stay distinct) and folds numeric
index segments whenever the remaining path and value match identically.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from typing import Any

from torchtitan.config.configurable import Configurable

_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _normalize_addrs(text: str) -> str:
    return _HEX_ADDR_RE.sub(">", text)


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _is_canonical_index(segment: str) -> bool:
    return (
        segment.isascii()
        and segment.isdigit()
        and (segment == "0" or not segment.startswith("0"))
    )


def _format_run(start: int, end: int) -> str:
    if end == start:
        return str(start)
    if end == start + 1:
        return f"{start},{end}"
    return f"{start}..{end}"


def _format_index_runs(indices: list[int]) -> str:
    parts: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        parts.append(_format_run(start, prev))
        start = prev = idx
    parts.append(_format_run(start, prev))
    return "{" + ",".join(parts) + "}"


def _is_primitive(val: Any) -> bool:
    return isinstance(val, (str, int, float, bool, type(None)))


def _flatten(val: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(val, str):
        out[prefix] = _normalize_addrs(val)
        return
    if isinstance(val, (int, float, bool, type(None))):
        out[prefix] = val
        return
    if isinstance(val, dict):
        if not val:
            out[prefix] = {}
            return
        for k, v in val.items():
            _flatten(v, _join(prefix, str(k)), out)
        return
    if isinstance(val, (list, tuple)):
        if not val:
            out[prefix] = []
            return
        if all(_is_primitive(item) for item in val):
            out[prefix] = [
                _normalize_addrs(item) if isinstance(item, str) else item
                for item in val
            ]
            return
        for i, item in enumerate(val):
            _flatten(item, _join(prefix, str(i)), out)
        return
    if (
        not isinstance(val, type)
        and hasattr(val, "to_dict")
        and type(val).to_dict is not Configurable.Config.to_dict
    ):
        raw = val.to_dict()
        if isinstance(raw, dict) and set(raw.keys()) == {"repr"}:
            out[prefix] = _normalize_addrs(str(raw["repr"]))
            return
        if is_dataclass(val):
            out[_join(prefix, "__class__")] = type(val).__qualname__
        _flatten(raw, prefix, out)
        return
    if is_dataclass(val) and not isinstance(val, type):
        out[_join(prefix, "__class__")] = type(val).__qualname__
        skip_private = isinstance(val, Configurable.Config)
        for f in fields(val):
            if skip_private and f.name.startswith("_"):
                continue
            _flatten(getattr(val, f.name), _join(prefix, f.name), out)
        return
    out[prefix] = _normalize_addrs(repr(val))


def _value_key(val: Any) -> Any:
    if isinstance(val, list):
        return ("list", tuple(_value_key(v) for v in val))
    if isinstance(val, dict):
        return ("dict", tuple((k, _value_key(v)) for k, v in val.items()))
    return (type(val).__name__, val)


def _fold_indexed_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    """Fold dotted paths whose values match identically (e.g. layers.{0..79}.dim)."""
    if len(mapping) < 2:
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
                    folded_seg = _format_index_runs(sorted(e[1] for e in entries))
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


def fold_indexed_paths(paths: Iterable[str]) -> list[str]:
    """Fold unique dotted paths that differ only in numeric index segments."""
    return list(_fold_indexed_dict(dict.fromkeys(paths)).keys())


def format_config(config: Configurable.Config) -> str:
    """Format a config tree as a compact, lossless flat JSON mapping."""
    flat: dict[str, Any] = {}
    _flatten(config, "", flat)
    return json.dumps(_fold_indexed_dict(flat), indent=2, ensure_ascii=False)
