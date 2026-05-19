#!/usr/bin/env python3
"""Compare two numerics activation logs and generate an interactive HTML diff.

Inputs are the per-rank text logs produced by
``torchtitan.tools.activation_tracer.dump_captures_to_file`` (typically
written to ``{dump_folder}/numerics/rank_{rank}_activations.log`` when
``--profiler.dump_numerics`` is set). Each log is a sequence of
``[module_fqn/op_N_opname]`` blocks with per-op stats (Shape, L1/L2 norm,
Min/Max/Mean), source ``Location``, and optional ``Phase``.

Pipeline:
    1. ``parse_log`` reads each file into a list of :class:`OpEntry`.
    2. ``match_entries`` aligns the two lists with three passes (exact key
       → fuzzy key → numeric stats) so that ops with different counters
       or FQNs across modes can still be compared.
    3. ``_compute_diffs`` flags per-stat differences, scaled by the
       tensor's L1 norm to suppress near-zero noise.
    4. ``generate_html`` renders an interactive page with collapsible
       Forward/Backward sections, per-cell diff coloring, filter toggles,
       and row hover-highlighting.

The naming "eager" vs "traced" is purely conventional — the tool is
symmetric, and any two logs (e.g. two ranks, two compile modes, two
checkpoints) can be compared.

Usage:
    python compare_numerics.py <eager_log> <traced_log> -o <output.html>
"""

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass, field


@dataclass
class OpEntry:
    """One captured op parsed from a numerics log file.

    Attributes:
        key: ``module_fqn/op_N_opname`` identifier from the ``[...]``
            header line. Used as the primary matching key.
        stats: Per-stat string values (Shape, L1 norm, L2 norm, Min, Max,
            Mean). Stored as strings so we can preserve formatting and
            detect non-numeric mismatches (e.g. shape).
        phase: ``"forward"`` or ``"backward"`` from the ``Phase:`` line.
            Forward is the default when the log omits the line.
        location: Source location like ``common/attention.py:534`` from
            the ``Location:`` line. Surfaced in the HTML for navigation.
    """

    key: str
    stats: dict[str, str] = field(default_factory=dict)
    phase: str = "forward"
    location: str = ""


def parse_log(path: str) -> tuple[list[OpEntry], set[str]]:
    """Parse an activation log file.

    The file format is the one produced by
    ``activation_tracer.dump_captures_to_file``: a small header (total
    count, optional ``Excluded ops dispatched:`` line) followed by a
    sequence of op blocks. Each op block begins with a ``[key]``
    header on its own line, followed by indented stat lines
    (``Shape:``, ``L1 norm:``, etc.), and optionally ``Location:`` and
    ``Phase:`` lines. Anything else is ignored.

    Args:
        path: Path to the log file (typically
            ``{dump_folder}/numerics/rank_{rank}_activations.log``).

    Returns:
        Tuple of ``(entries, skipped_excluded_ops)``.
        Entries are in the order they appear in the file (the
        op-dispatch order during the captured step, which is meaningful
        for the matching passes in ``match_entries``).
        ``skipped_excluded_ops`` is the set of op names that dispatched
        but were dropped during capture because they live in
        ``_EXCLUDED_OPS``; empty if the log doesn't carry the header
        line.
    """
    entries: list[OpEntry] = []
    skipped: set[str] = set()
    current = None
    with open(path) as f:
        for line in f:
            m = re.match(r"^\[(.+)\]", line)
            if m:
                if current:
                    entries.append(current)
                current = OpEntry(key=m.group(1))
            elif current:
                m2 = re.match(r"^\s+(Shape|L1 norm|L2 norm|Min|Max|Mean):\s+(.+)", line)
                if m2:
                    current.stats[m2.group(1)] = m2.group(2).strip()
                elif "Location:" in line:
                    current.location = line.strip().replace("Location: ", "")
                elif "Phase:" in line:
                    current.phase = line.strip().replace("Phase: ", "")
            elif line.startswith("Excluded ops dispatched:"):
                payload = line.split(":", 1)[1].strip()
                if payload and payload != "(none)":
                    skipped = {op.strip() for op in payload.split(",") if op.strip()}
    if current:
        entries.append(current)
    return entries, skipped


def _fuzzy_key(key: str) -> str:
    """Strip the per-module op counter from a key.

    Example: ``layers.0.w2/op_3_mm`` -> ``layers.0.w2/mm``.

    Used by ``match_entries`` Pass 2: between modes the same module may
    dispatch the same op N times but in different relative order (e.g.
    backward ops are interleaved differently), so the counter ``N`` is
    unstable. Stripping it lets us match by ``module_fqn + op_type``.
    """
    return re.sub(r"/op_\d+_", "/", key)


# Stat names recognized in the log and compared per-op. Must match the
# stats written by activation_tracer.dump_captures_to_file. Adding a new
# stat here also requires extending the parser regex in parse_log and
# the table headers in generate_html.
STAT_FIELDS = ["Shape", "L1 norm", "L2 norm", "Min", "Max", "Mean"]


def _rel_diff(a_str: str, b_str: str) -> float | None:
    """Compute symmetric relative difference between two stat strings.

    The denominator is ``max(|a|, |b|, 1e-30)`` so the result is bounded
    by 1 and never blows up when both values are tiny. The 1e-30 floor
    prevents division by zero when both inputs are exactly zero.

    Args:
        a_str: First stat string (e.g. ``"3.36e+04"``).
        b_str: Second stat string.

    Returns:
        Relative difference in ``[0, 1]`` for numeric inputs, or ``None``
        when either value cannot be parsed as a float (e.g. the
        ``Shape`` field, which is rendered as ``torch.Size([...])``).
    """
    try:
        a, b = float(a_str), float(b_str)
    except (ValueError, TypeError):
        return None
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _compute_diffs(e_entry, t_entry):
    """Compute per-stat differences between two matched op entries.

    Each stat is compared on its own scale via :func:`_rel_diff`
    (symmetric relative difference, ``abs(a-b) / max(|a|, |b|, 1e-30)``).
    L1 norm differences are judged against L1 magnitude, Min against
    Min magnitude, etc. — comparing across columns (e.g. scaling Min
    by L1) would over- or under-suppress diffs depending on stat.

    Special case: non-numeric stats (Shape, rendered as
    ``torch.Size([...])``) get a flat ``1.0`` intensity when they
    differ, since "the shape changed" is binary.

    Args:
        e_entry: Entry from the first log (typically eager).
        t_entry: Entry from the second log (typically traced).

    Returns:
        Dict mapping stat name to a relative-diff intensity in
        ``[0, 1]`` (numeric stats) or ``1.0`` (non-numeric mismatch).
        Only stats that actually differ are included; an empty dict
        means "matched".
    """
    diffs = {}
    for stat in STAT_FIELDS:
        ev = e_entry.stats.get(stat)
        tv = t_entry.stats.get(stat)
        if ev and tv and ev != tv:
            rd = _rel_diff(ev, tv)
            # _rel_diff returns None for non-numeric stats (Shape).
            diffs[stat] = 1.0 if rd is None else rd
    return diffs


def match_entries(eager: list[OpEntry], traced: list[OpEntry]):
    """Pair up ops between two logs using a three-pass matching strategy.

    Cross-mode comparisons (eager vs aot_fx_trace, etc.)
    can't rely on identical keys: op counters drift when ops are
    interleaved differently, and some ops have different module
    attributions between modes (e.g. gradient accumulation adds may
    surface under different FQNs). The three passes progressively relax
    the matching criteria so that ops still get paired:

    Pass 1 — Exact key match. Catches the bulk of forward ops where
        both runs dispatch the same op in the same module-relative
        order. Most-specific match; runs first to claim the
        "obviously the same op" cases.

    Pass 2 — Fuzzy key match (``module_fqn + op_type``, counter
        stripped via :func:`_fuzzy_key`). Handles ops where the
        per-module sequence number drifts between modes, common for
        backward ops which are interleaved differently.

    Pass 3 — Stats match (``op_type + Shape + L1 norm``). Last-resort
        match for ops whose module FQN differs between modes (e.g.
        gradient accumulation adds in autograd internals). Identifies
        ops by their tensor signature rather than by name.

    Anything still unmatched is returned with status ``eager_only`` or
    ``traced_only`` so it shows up in the HTML's "only" filter.

    Args:
        eager: Entries from the first log.
        traced: Entries from the second log.

    Returns:
        List of ``(eager_entry | None, traced_entry | None, status,
        diffs, match_strategy)`` tuples. ``status`` is one of
        ``"match"``, ``"diff"``, ``"eager_only"``, ``"traced_only"``.
        ``match_strategy`` records which pass paired the entries:
        ``"exact"``, ``"fuzzy"``, ``"stats"``, or ``""`` for only-side
        rows. Eager-side entries appear in their original log order;
        ``traced_only`` entries follow at the end in their original
        traced-log order.
    """
    traced_used = set()

    # Collect a (status, diffs, traced_entry, strategy) for each eager
    # entry, by its index in the eager list. Each pass fills in slots
    # that earlier passes left empty. Emitting in eager-index order at
    # the end keeps the output aligned with the eager log's op-dispatch
    # sequence.
    eager_results: list[tuple[str, dict, OpEntry, str] | None] = [None] * len(eager)

    def _try_pass(strategy: str, key_fn) -> None:
        # Build traced-side index from currently unclaimed entries, then
        # walk eager in order and claim the first available match per
        # entry that hasn't been filled by an earlier pass.
        index: dict[object, list[OpEntry]] = {}
        for e in traced:
            if id(e) not in traced_used:
                index.setdefault(key_fn(e), []).append(e)
        for i, e_entry in enumerate(eager):
            if eager_results[i] is not None:
                continue
            for c in index.get(key_fn(e_entry), []):
                if id(c) not in traced_used:
                    traced_used.add(id(c))
                    diffs = _compute_diffs(e_entry, c)
                    status = "diff" if diffs else "match"
                    eager_results[i] = (status, diffs, c, strategy)
                    break

    # Pass 1: exact match (forward ops, where keys agree).
    _try_pass("exact", lambda e: e.key)

    # Pass 2: fuzzy match (module_fqn + op_type, counter stripped).
    _try_pass("fuzzy", lambda e: _fuzzy_key(e.key))

    # Pass 3: stats match (op_type + Shape + L1 norm) — handles ops
    # whose FQN differs between modes.
    def _stats_key(entry: OpEntry) -> tuple[str, str, str]:
        op_type = _fuzzy_key(entry.key).split("/")[-1] if "/" in entry.key else entry.key
        return (op_type, entry.stats.get("Shape", ""), entry.stats.get("L1 norm", ""))

    _try_pass("stats", _stats_key)

    # Emit eager entries in their original order. Anything still None
    # is eager_only.
    results = []
    for e_entry, slot in zip(eager, eager_results, strict=True):
        if slot is None:
            results.append((e_entry, None, "eager_only", {}, ""))
        else:
            status, diffs, t_entry, strategy = slot
            results.append((e_entry, t_entry, status, diffs, strategy))

    # traced_only entries (never claimed by any pass) trail at the end
    # in traced-log order.
    for t_entry in traced:
        if id(t_entry) not in traced_used:
            results.append((None, t_entry, "traced_only", {}, ""))

    return results


def generate_html(
    results,
    eager_path,
    traced_path,
    name1: str = "run1",
    name2: str = "run2",
    skipped1: set[str] | None = None,
    skipped2: set[str] | None = None,
) -> str:
    """Render matching results as a self-contained interactive HTML page.

    The output has:

    * A summary bar with match / diff / only-side counts.
    * Filter checkboxes to hide each row category.
    * Two collapsible sections — Forward and Backward — each rendering
      a table with one row per (run1, run2) pair. Each row shows both
      sides' stats stacked, with diff cells colored on a white-to-red
      log scale, and the source ``Location`` for click-to-navigate
      context.
    * Hover-highlighting that tints both halves of a row pair together.

    The page is fully self-contained (inline CSS + JS, no external
    assets).

    Args:
        results: Output of :func:`match_entries`.
        eager_path: Path of the first log (currently unused in output;
            kept so callers can pass it for future provenance display).
        traced_path: Path of the second log.
        name1: Display name for the first log (defaults to ``"run1"``).
            Used in the page title, row-side label, and filter
            checkbox.
        name2: Display name for the second log (defaults to ``"run2"``).
        skipped1: Op names that dispatched in run 1 but were dropped by
            ``_EXCLUDED_OPS`` (from the log's ``Excluded ops
            dispatched:`` header). Rendered as a small chip list under
            the summary so the user sees which always-skipped ops the
            model actually exercises.
        skipped2: Same for run 2.

    Returns:
        Complete HTML document as a string.
    """
    # HTML-escape user-provided names since they go into the document
    # body and attribute values.
    name1_safe = html.escape(name1)
    name2_safe = html.escape(name2)

    def _skipped_block(label: str, ops: set[str] | None) -> str:
        if not ops:
            chips = '<span class="skipped-none">(none)</span>'
        else:
            chips = "".join(
                f'<span class="skipped-chip">{html.escape(op)}</span>'
                for op in sorted(ops)
            )
        return (
            f'<div class="skipped-row">'
            f'<span class="skipped-label">{html.escape(label)}:</span> {chips}'
            f"</div>"
        )

    skipped_section = (
        f'<div class="skipped-section">'
        f'<div class="skipped-title">Excluded ops dispatched '
        f'<span class="skipped-help">(present at runtime but filtered '
        f'by <code>_EXCLUDED_OPS</code>)</span></div>'
        f"{_skipped_block(name1, skipped1)}"
        f"{_skipped_block(name2, skipped2)}"
        f"</div>"
    )
    def _phase(e, t):
        return (e.phase if e else t.phase)
    fwd = [r for r in results if _phase(r[0], r[1]) == "forward"]
    bwd = [r for r in results if _phase(r[0], r[1]) == "backward"]

    total_match = sum(1 for r in results if r[2] == "match")
    total_diff = sum(1 for r in results if r[2] == "diff")
    total_eager = sum(1 for r in results if r[2] == "eager_only")
    total_traced = sum(1 for r in results if r[2] == "traced_only")

    def make_table(entries, section_id):
        rows = []
        for idx, (e, t, status, diffs, strategy) in enumerate(entries):
            row_id = f"{section_id}_{idx}"

            if status == "eager_only":
                cls = "eager-only"
            elif status == "traced_only":
                cls = "traced-only"
            elif status == "diff":
                cls = "has-diff"
            else:
                cls = "all-match"

            # Show both keys when they differ (fuzzy match)
            e_key = html.escape(e.key) if e else ""
            t_key = html.escape(t.key) if t else ""
            if e and t and e.key == t.key:
                key_html = e_key
            elif e and t:
                key_html = (
                    f'<span class="key-eager">{e_key}</span>'
                    f'<br><span class="key-traced">{t_key}</span>'
                )
            else:
                key_html = e_key or t_key

            # Build stat cells with intensity-based diff coloring.
            # Relative diff is mapped to opacity: tiny diffs are faint
            # red, large diffs and non-numeric mismatches (Shape) are
            # full red.
            eager_cells = []
            traced_cells = []
            for stat in STAT_FIELDS:
                ev = e.stats.get(stat, "") if e else ""
                tv = t.stats.get(stat, "") if t else ""
                rd = diffs.get(stat)
                if rd is not None:
                    # White-to-red gradient: intensity 0 = white text,
                    # intensity 1 = full red text + red background tint.
                    # Log scale: 1e-8 → 0.0, 1e-1+ → 1.0
                    if rd >= 0.1:
                        t_val = 1.0
                    elif rd <= 1e-8:
                        t_val = 0.0
                    else:
                        t_val = (math.log10(rd) + 8) / 7
                    # Text: white (220,220,220) → red (248,80,80)
                    r = int(220 + (248 - 220) * t_val)
                    g = int(220 - (220 - 80) * t_val)
                    b = int(220 - (220 - 80) * t_val)
                    # Background: transparent → faint red
                    bg_alpha = t_val * 0.15
                    style = (
                        f' style="color: rgb({r},{g},{b});'
                        f" background: rgba(248,80,80,{bg_alpha:.2f});"
                        f' font-weight: bold"'
                    )
                else:
                    style = ""
                eager_cells.append(f"<td{style}>{html.escape(ev)}</td>")
                traced_cells.append(f"<td{style}>{html.escape(tv)}</td>")

            e_loc = html.escape(e.location) if e else ""
            t_loc = html.escape(t.location) if t else ""
            e_phase = e.phase if e else ""
            t_phase = t.phase if t else ""

            strategy_label = {
                "exact": "same op key",
                "fuzzy": "fuzzy op key",
                "stats": "stats",
            }.get(strategy, "")
            strategy_html = (
                f'<span class="strategy strategy-{strategy}">{strategy_label}</span>'
                if strategy
                else ""
            )

            rows.append(f"""
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="key-cell" rowspan="2">{key_html}</td>
              <td class="match-cell" rowspan="2">{strategy_html}</td>
              <td class="side-label">{name1_safe}</td>
              {''.join(eager_cells)}
              <td>{e_loc}</td>
              <td>{e_phase}</td>
            </tr>
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="side-label">{name2_safe}</td>
              {''.join(traced_cells)}
              <td>{t_loc}</td>
              <td>{t_phase}</td>
            </tr>""")

        return "\n".join(rows)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Numerics Comparison</title>
<style>
  body {{ font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; font-size: 18px; }}
  h2 {{ color: #7ec8e3; font-size: 15px; margin-top: 30px; cursor: pointer; }}
  h2:hover {{ color: #00d4ff; }}
  .summary {{ background: #16213e; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; display: inline-block; }}
  .summary span {{ margin-right: 20px; }}
  .match-count {{ color: #4ade80; }}
  .diff-count {{ color: #f87171; }}
  .only-count {{ color: #fbbf24; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #0f3460; color: #e0e0e0; padding: 6px 10px; text-align: left; position: sticky; top: 0; z-index: 1; }}
  td {{ padding: 4px 10px; border-bottom: 1px solid #1a1a3e; white-space: nowrap; }}
  .key-cell {{ font-weight: bold; color: #7ec8e3; vertical-align: top; border-right: 2px solid #0f3460; max-width: 400px; overflow: hidden; text-overflow: ellipsis; line-height: 1.6; }}
  .key-eager {{ color: #c4b5fd; font-size: 11px; }}
  .key-traced {{ color: #67e8f9; font-size: 11px; }}
  .side-label {{ color: #888; font-size: 11px; width: 50px; }}
  .match-cell {{ vertical-align: middle; border-right: 2px solid #0f3460; }}
  .strategy {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .strategy-exact {{ color: #4ade80; background: rgba(74,222,128,0.15); }}
  .strategy-fuzzy {{ color: #fbbf24; background: rgba(251,191,36,0.15); }}
  .strategy-stats {{ color: #a78bfa; background: rgba(167,139,250,0.15); }}
  .op-row:nth-of-type(4n+1), .op-row:nth-of-type(4n+2) {{ background: #0f1629; }}
  .op-row:nth-of-type(4n+3), .op-row:nth-of-type(4n+4) {{ background: #1a1a2e; }}
  .op-row.highlighted {{ background: #1e3a5f !important; }}
  .section {{ margin-bottom: 30px; }}
  .section-content {{ display: block; }}
  .section-content.collapsed {{ display: none; }}
  .toggle {{ font-size: 12px; color: #888; }}
  .skipped-section {{ background: #16213e; padding: 10px 15px; border-radius: 8px; margin-bottom: 20px; }}
  .skipped-title {{ color: #7ec8e3; font-weight: bold; margin-bottom: 6px; }}
  .skipped-help {{ color: #888; font-weight: normal; font-size: 11px; }}
  .skipped-row {{ margin-bottom: 4px; }}
  .skipped-label {{ color: #aaa; margin-right: 6px; }}
  .skipped-chip {{ display: inline-block; background: rgba(167,139,250,0.15); color: #a78bfa; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right: 4px; }}
  .skipped-none {{ color: #666; font-style: italic; font-size: 11px; }}
  .filter-bar {{ margin-bottom: 15px; }}
  .filter-bar label {{ margin-right: 15px; color: #aaa; cursor: pointer; }}
  .filter-bar input {{ margin-right: 4px; }}
</style>
</head>
<body>
<h1>Numerics Comparison: {name1_safe} vs {name2_safe}</h1>
<div class="summary">
  <span class="match-count">✓ {total_match} matched</span>
  <span class="diff-count">✗ {total_diff} diffs</span>
  <span class="only-count">← {total_eager} {name1_safe} only</span>
  <span class="only-count">→ {total_traced} {name2_safe} only</span>
</div>

{skipped_section}

<div class="filter-bar">
  <label><input type="checkbox" checked onchange="toggleClass('all-match', this.checked)"> Show matched</label>
  <label><input type="checkbox" checked onchange="toggleClass('has-diff', this.checked)"> Show diffs</label>
  <label><input type="checkbox" checked onchange="toggleClass('eager-only', this.checked)"> Show {name1_safe}-only</label>
  <label><input type="checkbox" checked onchange="toggleClass('traced-only', this.checked)"> Show {name2_safe}-only</label>
</div>

<div class="section">
  <h2 onclick="toggleSection('fwd')">Forward ({len(fwd)} ops) <span class="toggle" id="fwd-toggle">▼</span></h2>
  <div class="section-content" id="fwd">
    <table>
      <tr><th>Op</th><th>Match method</th><th>Side</th><th>Shape</th><th>L1 norm</th><th>L2 norm</th><th>Min</th><th>Max</th><th>Mean</th><th>Location</th><th>Phase</th></tr>
      {make_table(fwd, "fwd")}
    </table>
  </div>
</div>

<div class="section">
  <h2 onclick="toggleSection('bwd')">Backward ({len(bwd)} ops) <span class="toggle" id="bwd-toggle">▼</span></h2>
  <div class="section-content" id="bwd">
    <table>
      <tr><th>Op</th><th>Match method</th><th>Side</th><th>Shape</th><th>L1 norm</th><th>L2 norm</th><th>Min</th><th>Max</th><th>Mean</th><th>Location</th><th>Phase</th></tr>
      {make_table(bwd, "bwd")}
    </table>
  </div>
</div>


<script>
function highlight(rowId) {{
  document.querySelectorAll('[data-row-id="' + rowId + '"]').forEach(
    el => el.classList.add('highlighted'));
}}
function unhighlight(rowId) {{
  document.querySelectorAll('[data-row-id="' + rowId + '"]').forEach(
    el => el.classList.remove('highlighted'));
}}
function toggleSection(id) {{
  const el = document.getElementById(id);
  const toggle = document.getElementById(id + '-toggle');
  if (el.classList.contains('collapsed')) {{
    el.classList.remove('collapsed');
    toggle.textContent = '▼';
  }} else {{
    el.classList.add('collapsed');
    toggle.textContent = '▶';
  }}
}}
function toggleClass(cls, show) {{
  document.querySelectorAll('.op-row.' + cls).forEach(el => {{
    el.style.display = show ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


def naive_compare_captures(
    captures_a: dict,
    captures_b: dict,
    *,
    rtol: float = 0,
    atol: float = 0,
    verbose: bool = True,
) -> dict[str, dict]:
    """Compare activations captured from two runs by raw key.

    Simple key-based comparison (requires identical key sets). For
    cross-mode comparison with fuzzy matching, use ``match_entries``
    on parsed log files instead.

    Args:
        captures_a: Captures from run A (dict of CapturedActivation).
        captures_b: Captures from run B.
        rtol: Relative tolerance (0 = bitwise).
        atol: Absolute tolerance (0 = bitwise).
        verbose: Print comparison results.

    Returns:
        Dict mapping keys to ``{"match": bool, "max_diff": float, ...}``.
    """
    import torch

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore[assignment, misc]

    results = {}
    common_keys = set(captures_a.keys()) & set(captures_b.keys())

    if verbose:
        print(f"\n{'=' * 80}")
        print("Activation Parity Comparison")
        print(f"A keys: {len(captures_a)}, B keys: {len(captures_b)}, "
              f"common: {len(common_keys)}")
        print(f"{'=' * 80}")

    for key in sorted(common_keys):
        ta = captures_a[key].tensor
        tb = captures_b[key].tensor

        if DTensor is not None:
            ta = ta._local_tensor if isinstance(ta, DTensor) else ta
            tb = tb._local_tensor if isinstance(tb, DTensor) else tb

        if ta.shape != tb.shape:
            if verbose:
                print(f"\n[{key}] SHAPE MISMATCH: {ta.shape} vs {tb.shape}")
            results[key] = {"match": False, "error": "shape mismatch"}
            continue

        diff = (ta.float() - tb.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if rtol == 0 and atol == 0:
            match = torch.equal(ta, tb)
        else:
            match = torch.allclose(ta, tb, rtol=rtol, atol=atol)

        results[key] = {
            "match": match,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }

        if verbose:
            status = "MATCH" if match else "DIFF"
            print(f"\n[{key}] {status}")
            print(f"  Shape: {ta.shape}, Dtype: {ta.dtype}")
            print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if verbose:
        total = len(results)
        matched = sum(1 for r in results.values() if r.get("match", False))
        print(f"\n{'=' * 80}")
        print(f"Summary: {matched}/{total} matched")
        print(f"{'=' * 80}")

    return results


def main():
    """CLI entry point: parse two logs, match entries, write HTML diff.

    Invoked via ``python -m torchtitan.tools.compare_numerics`` or
    directly. See module docstring for the typical workflow.
    """
    parser = argparse.ArgumentParser(description="Compare numerics activation logs")
    parser.add_argument("eager", help="First log file")
    parser.add_argument("traced", help="Second log file")
    parser.add_argument("-o", "--output", default="numerics_diff.html", help="Output HTML file")
    parser.add_argument(
        "--name1",
        default="run1",
        help="Display name for the first log (default: run1)",
    )
    parser.add_argument(
        "--name2",
        default="run2",
        help="Display name for the second log (default: run2)",
    )
    args = parser.parse_args()

    eager, skipped_eager = parse_log(args.eager)
    traced, skipped_traced = parse_log(args.traced)
    results = match_entries(eager, traced)
    html_content = generate_html(
        results,
        args.eager,
        args.traced,
        name1=args.name1,
        name2=args.name2,
        skipped1=skipped_eager,
        skipped2=skipped_traced,
    )

    with open(args.output, "w") as f:
        f.write(html_content)
    print(f"Wrote {args.output} ({len(results)} entries)")


if __name__ == "__main__":
    main()
