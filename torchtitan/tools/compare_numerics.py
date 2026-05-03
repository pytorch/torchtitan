#!/usr/bin/env python3
"""Compare two numerics activation logs and generate an interactive HTML diff.

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
    key: str
    stats: dict[str, str] = field(default_factory=dict)
    phase: str = "forward"
    location: str = ""


def parse_log(path: str) -> list[OpEntry]:
    entries = []
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
    if current:
        entries.append(current)
    return entries


def _fuzzy_key(key: str) -> str:
    """Strip op counter: 'layers.0.w2/op_3_mm' -> 'layers.0.w2/mm'"""
    return re.sub(r"/op_\d+_", "/", key)


STAT_FIELDS = ["Shape", "L1 norm", "L2 norm", "Min", "Max", "Mean"]


def _rel_diff(a_str: str, b_str: str) -> float | None:
    """Compute relative difference between two numeric strings.
    Returns None for non-numeric values (shape, dtype)."""
    try:
        a, b = float(a_str), float(b_str)
    except (ValueError, TypeError):
        return None
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _compute_diffs(e_entry, t_entry):
    """Returns dict mapping stat name to relative diff (0-1 scale for
    numeric fields, 1.0 for non-numeric mismatches like Shape).

    Skips differences that are negligible relative to the tensor's
    L1 norm (e.g. mean of 1e-20 vs -1e-20 when L1 is 1e+3).
    """
    # Use L1 norm as the reference scale for the tensor.
    e_l1 = e_entry.stats.get("L1 norm")
    t_l1 = t_entry.stats.get("L1 norm")
    try:
        ref_scale = max(float(e_l1), float(t_l1)) if e_l1 and t_l1 else 0
    except ValueError:
        ref_scale = 0

    diffs = {}
    for stat in STAT_FIELDS:
        ev = e_entry.stats.get(stat)
        tv = t_entry.stats.get(stat)
        if ev and tv and ev != tv:
            rd = _rel_diff(ev, tv)
            if rd is None:
                # Non-numeric (Shape) → always flag at max
                diffs[stat] = 1.0
            elif ref_scale > 0 and stat not in ("Shape",):
                # Use absolute difference relative to tensor scale
                # so near-zero noise gets a very faint color.
                try:
                    abs_diff = abs(float(ev) - float(tv))
                except ValueError:
                    abs_diff = 0
                scale_rd = abs_diff / ref_scale
                diffs[stat] = scale_rd
            else:
                diffs[stat] = rd
    return diffs


def match_entries(eager: list[OpEntry], traced: list[OpEntry]):
    """Match entries by key. Forward uses exact match, backward uses
    fuzzy match (ignoring op counter) to handle different op interleaving
    between eager and traced modes."""
    traced_used = set()
    results = []

    # Pass 1: exact match (forward ops)
    traced_exact = {}
    for e in traced:
        traced_exact.setdefault(e.key, []).append(e)

    unmatched_eager = []
    for e_entry in eager:
        candidates = traced_exact.get(e_entry.key, [])
        t_entry = None
        for c in candidates:
            if id(c) not in traced_used:
                t_entry = c
                traced_used.add(id(c))
                break
        if t_entry:
            diffs = _compute_diffs(e_entry, t_entry)
            status = "diff" if diffs else "match"
            results.append((e_entry, t_entry, status, diffs))
        else:
            unmatched_eager.append(e_entry)

    # Pass 2: fuzzy match for unmatched ops (backward/recompute)
    # Match by module_fqn + op_type, ignoring the counter.
    traced_fuzzy = {}
    for e in traced:
        if id(e) not in traced_used:
            traced_fuzzy.setdefault(_fuzzy_key(e.key), []).append(e)

    still_unmatched = []
    for e_entry in unmatched_eager:
        fk = _fuzzy_key(e_entry.key)
        candidates = traced_fuzzy.get(fk, [])
        t_entry = None
        for c in candidates:
            if id(c) not in traced_used:
                t_entry = c
                traced_used.add(id(c))
                break
        if t_entry:
            diffs = _compute_diffs(e_entry, t_entry)
            status = "diff" if diffs else "match"
            results.append((e_entry, t_entry, status, diffs))
        else:
            still_unmatched.append(e_entry)

    # Pass 3: numeric match for remaining unmatched ops.
    # Match by shape + L1 norm (for gradient accumulation adds etc.
    # that have different FQNs between modes).
    traced_by_stats = {}
    for e in traced:
        if id(e) not in traced_used:
            shape = e.stats.get("Shape", "")
            l1 = e.stats.get("L1 norm", "")
            op_type = _fuzzy_key(e.key).split("/")[-1] if "/" in e.key else e.key
            stats_key = (op_type, shape, l1)
            traced_by_stats.setdefault(stats_key, []).append(e)

    for e_entry in still_unmatched:
        shape = e_entry.stats.get("Shape", "")
        l1 = e_entry.stats.get("L1 norm", "")
        op_type = _fuzzy_key(e_entry.key).split("/")[-1] if "/" in e_entry.key else e_entry.key
        stats_key = (op_type, shape, l1)
        candidates = traced_by_stats.get(stats_key, [])
        t_entry = None
        for c in candidates:
            if id(c) not in traced_used:
                t_entry = c
                traced_used.add(id(c))
                break
        if t_entry:
            diffs = _compute_diffs(e_entry, t_entry)
            status = "diff" if diffs else "match"
            results.append((e_entry, t_entry, status, diffs))
        else:
            results.append((e_entry, None, "eager_only", {}))

    for t_entry in traced:
        if id(t_entry) not in traced_used:
            results.append((None, t_entry, "traced_only", {}))

    return results


def generate_html(results, eager_path, traced_path) -> str:
    # Group by phase — recompute goes into backward
    fwd = [(e, t, s, d) for e, t, s, d in results
           if (e and e.phase == "forward") or (t and t.phase == "forward")]
    bwd = [(e, t, s, d) for e, t, s, d in results
           if (e and e.phase == "backward") or (t and t.phase == "backward")]

    total_match = sum(1 for _, _, s, _ in results if s == "match")
    total_diff = sum(1 for _, _, s, _ in results if s == "diff")
    total_eager = sum(1 for _, _, s, _ in results if s == "eager_only")
    total_traced = sum(1 for _, _, s, _ in results if s == "traced_only")

    def make_table(entries, section_id):
        rows = []
        for idx, (e, t, status, diffs) in enumerate(entries):
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

            rows.append(f"""
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="key-cell" rowspan="2">{key_html}</td>
              <td class="side-label">eager</td>
              {''.join(eager_cells)}
              <td>{e_loc}</td>
              <td>{e_phase}</td>
            </tr>
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="side-label">traced</td>
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
  .op-row:nth-of-type(4n+1), .op-row:nth-of-type(4n+2) {{ background: #0f1629; }}
  .op-row:nth-of-type(4n+3), .op-row:nth-of-type(4n+4) {{ background: #1a1a2e; }}
  .op-row.highlighted {{ background: #1e3a5f !important; }}
  .section {{ margin-bottom: 30px; }}
  .section-content {{ display: block; }}
  .section-content.collapsed {{ display: none; }}
  .toggle {{ font-size: 12px; color: #888; }}
  .filter-bar {{ margin-bottom: 15px; }}
  .filter-bar label {{ margin-right: 15px; color: #aaa; cursor: pointer; }}
  .filter-bar input {{ margin-right: 4px; }}
</style>
</head>
<body>
<h1>Numerics Comparison: Eager vs aot_fx_trace</h1>
<div class="summary">
  <span class="match-count">✓ {total_match} matched</span>
  <span class="diff-count">✗ {total_diff} diffs</span>
  <span class="only-count">← {total_eager} eager only</span>
  <span class="only-count">→ {total_traced} traced only</span>
</div>

<div class="filter-bar">
  <label><input type="checkbox" checked onchange="toggleClass('all-match', this.checked)"> Show matched</label>
  <label><input type="checkbox" checked onchange="toggleClass('has-diff', this.checked)"> Show diffs</label>
  <label><input type="checkbox" checked onchange="toggleClass('eager-only', this.checked)"> Show eager-only</label>
  <label><input type="checkbox" checked onchange="toggleClass('traced-only', this.checked)"> Show traced-only</label>
</div>

<div class="section">
  <h2 onclick="toggleSection('fwd')">Forward ({len(fwd)} ops) <span class="toggle" id="fwd-toggle">▼</span></h2>
  <div class="section-content" id="fwd">
    <table>
      <tr><th>Op</th><th>Side</th><th>Shape</th><th>L1 norm</th><th>L2 norm</th><th>Min</th><th>Max</th><th>Mean</th><th>Location</th><th>Phase</th></tr>
      {make_table(fwd, "fwd")}
    </table>
  </div>
</div>

<div class="section">
  <h2 onclick="toggleSection('bwd')">Backward ({len(bwd)} ops) <span class="toggle" id="bwd-toggle">▼</span></h2>
  <div class="section-content" id="bwd">
    <table>
      <tr><th>Op</th><th>Side</th><th>Shape</th><th>L1 norm</th><th>L2 norm</th><th>Min</th><th>Max</th><th>Mean</th><th>Location</th><th>Phase</th></tr>
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
    parser = argparse.ArgumentParser(description="Compare numerics activation logs")
    parser.add_argument("eager", help="Eager mode log file")
    parser.add_argument("traced", help="Traced mode log file")
    parser.add_argument("-o", "--output", default="numerics_diff.html", help="Output HTML file")
    args = parser.parse_args()

    eager = parse_log(args.eager)
    traced = parse_log(args.traced)
    results = match_entries(eager, traced)
    html_content = generate_html(results, args.eager, args.traced)

    with open(args.output, "w") as f:
        f.write(html_content)
    print(f"Wrote {args.output} ({len(results)} entries)")


if __name__ == "__main__":
    main()
