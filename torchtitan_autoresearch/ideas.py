"""Load the human's advisory guidance from `Ideas.md`.

Ideas are the human -> agent advisory channel (ARCHITECTURE.md section 4.2):
hints, priors, soft constraints, references. They never bind a verdict. The
harness parses them and exposes them via `observe().ideas`; the agent
acknowledges an item by putting its id in `candidate.addresses`.

The file format is a markdown list of items, each a block of
``- id:/kind:/target:/weight:/text:`` lines; ``text`` may span multiple
continuation lines until the next field or item. Parsing is intentionally
forgiving so the human can write naturally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_KINDS = {"hint", "prior", "soft_constraint", "reference"}
_ITEM = re.compile(r"^- id:\s*(.+)$")
_FIELD = re.compile(r"^\s+(id|kind|target|weight|text):\s*(.*)$")


@dataclass
class Idea:
    id: str
    kind: str
    text: str
    target: str | None = None
    weight: float = 1.0


def load_ideas(path: str) -> list[Idea]:
    """Parse advisory items; malformed items are skipped, not fatal."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    items: list[Idea] = []
    cur: dict | None = None
    field: str | None = None

    def flush() -> None:
        if cur and cur.get("id") and cur.get("kind") in _VALID_KINDS:
            items.append(
                Idea(
                    id=cur["id"].strip(),
                    kind=cur["kind"].strip(),
                    text=cur.get("text", "").strip(),
                    target=(cur.get("target") or "").strip() or None,
                    weight=float(cur.get("weight", 1.0) or 1.0),
                )
            )

    for raw in lines:
        m_item = _ITEM.match(raw)
        if m_item:
            flush()
            cur = {"id": m_item.group(1).strip()}
            field = "id"
            continue
        if cur is None:
            continue
        m_field = _FIELD.match(raw)
        if m_field:
            field = m_field.group(1)
            cur[field] = m_field.group(2)
        elif field == "text" and raw.strip():
            # continuation of a multi-line text field
            cur["text"] = (cur.get("text", "") + " " + raw.strip()).strip()
        elif not raw.strip():
            field = None
    flush()
    return items
