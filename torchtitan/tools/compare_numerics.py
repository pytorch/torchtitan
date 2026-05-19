#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    # DebugMode norm hash of the op result.  Empty for in-place ops
    # (no result) and for logs that predate the hash fields.  Stored
    # as a single string — comma-separated when the op returns a tuple.
    output_hash: str = ""
    # Comma-separated norm hashes of input tensors, captured *before*
    # the op runs.  Useful for detecting in-place mutations: comparing
    # an op's ``input_hashes`` against the previous op's
    # ``input_hashes`` for the same tensor surfaces the change even
    # when ``output_hash`` is empty.
    input_hashes: str = ""
    # Semicolon-separated producing capture key per input tensor,
    # aligned positionally with ``input_hashes``.  Empty positions
    # mean the input came from outside the captured op stream
    # (uncaptured intermediate, parameter, dataloader).  Used to render
    # "produced by <op>" tooltips in the HTML.
    input_producers: str = ""


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
    current: OpEntry | None = None
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
                elif "Output hash:" in line:
                    current.output_hash = line.strip().replace("Output hash: ", "")
                elif "Input hashes:" in line:
                    current.input_hashes = line.strip().replace("Input hashes: ", "")
                elif "Input producers:" in line:
                    current.input_producers = line.strip().replace(
                        "Input producers: ", ""
                    )
            elif line.startswith("Excluded ops dispatched:"):
                payload = line.split(":", 1)[1].strip()
                if payload and payload != "(none)":
                    skipped = {op.strip() for op in payload.split(",") if op.strip()}
    if current:
        entries.append(current)
    return entries, skipped


# Leading CamelCase class-name segment in a key, e.g.
# ``FSDPLlama3Model.layers.0...``.  ModTracker (used by the DebugMode
# eager backend) roots FQNs at the wrapped model class, while
# FQNInterpreter (used during traced replay) uses the unprefixed
# ``layers.0...`` form.  The pattern requires the segment to start with
# an uppercase letter and contain another uppercase letter, so common
# all-lowercase attribute names (``layers``, ``feed_forward``) are
# never accidentally stripped.
_ROOT_CLASS_PREFIX_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*\.")


def _strip_root_class_prefix(key: str) -> str:
    """Remove a leading ``ClassName.`` segment from an FQN.

    Backend-agnostic match normalization: the DebugMode eager path
    produces ``FSDPLlama3Model.layers.0.attention.wo/op_0_mm`` while
    the traced path produces ``layers.0.attention.wo/op_0_mm``.
    Stripping the class segment lets them match by key.

    Generalizes across models — ``FSDPQwenModel.``, ``Llama2Model.``,
    etc. are all matched by the same heuristic.
    """
    return _ROOT_CLASS_PREFIX_RE.sub("", key, count=1)


def _fuzzy_key(key: str) -> str:
    """Strip the per-module op counter from a key.

    Example: ``layers.0.w2/op_3_mm`` -> ``layers.0.w2/mm``.

    Used by ``match_entries`` Pass 2: between modes the same module may
    dispatch the same op N times but in different relative order (e.g.
    backward ops are interleaved differently), so the counter ``N`` is
    unstable. Stripping it lets us match by ``module_fqn + op_type``.

    Also strips a leading ``ClassName.`` segment so eager DebugMode
    keys (rooted at the model class) match traced keys (unrooted).
    """
    return _strip_root_class_prefix(re.sub(r"/op_\d+_", "/", key))


# Stat names recognized in the log and compared per-op. Must match the
# stats written by activation_tracer.dump_captures_to_file. Adding a new
# stat here also requires extending the parser regex in parse_log and
# the table headers in generate_html.  L1 norm is intentionally absent —
# it duplicates ``output_hash`` (the DebugMode norm hash is L1 in
# float64), so we render that column as "Output L1 norm" instead.
STAT_FIELDS = ["Shape", "L2 norm", "Min", "Max", "Mean"]

# Column-header labels for the rendered HTML.  Keeping the underlying
# OpEntry.stats keys short (matching the log) while exposing
# user-friendly headers that make it clear these stats describe the
# *output* tensor.  Shape is left bare since it's the only one not
# obviously an aggregate.
_STAT_DISPLAY_LABELS = {
    "Shape": "Shape",
    "L2 norm": "Output L2 norm",
    "Min": "Output Min",
    "Max": "Output Max",
    "Mean": "Output Mean",
}


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


def _hash_intensity(e_val: str, t_val: str) -> float | None:
    """Max relative diff between two comma-separated hash lists.

    Returns:
        - ``None`` when both sides are empty.
        - ``0.0`` when the strings are identical or numerically equal.
        - ``1.0`` for structural mismatches (different list length, or
          one side empty, or non-numeric tokens that don't match).
        - The largest position-wise relative diff (``_rel_diff``)
          otherwise.

    Used for both the HTML cell coloring intensity and the
    ``_compute_match_status`` threshold.
    """
    if not (e_val or t_val):
        return None
    if not e_val or not t_val:
        return 1.0
    if e_val == t_val:
        return 0.0
    ev = [v.strip() for v in e_val.split(",")]
    tv = [v.strip() for v in t_val.split(",")]
    if len(ev) != len(tv):
        return 1.0
    max_rd = 0.0
    for a_s, b_s in zip(ev, tv):
        rd = _rel_diff(a_s, b_s)
        if rd is None:
            if a_s != b_s:
                return 1.0
            continue
        if rd > max_rd:
            max_rd = rd
    return max_rd


def _compute_match_status(e_entry, t_entry) -> str:
    """Classify a paired row as ``"match"`` or ``"diff"``.

    The classification considers only ``Shape``, ``output_hash``, and
    ``input_hashes`` — the three fields that uniquely identify "this
    is the same activation".  L2 norm / Min / Max / Mean are
    sub-statistics of the same tensor; they get displayed (with
    intensity coloring) for context, but they don't drive the toggle.

    Threshold: hash relative-diff > 1e-8 counts as a real difference,
    matching the floor used by the hash-cell color gradient (smaller
    diffs are pure float64 reduction noise).
    """
    if e_entry.stats.get("Shape", "") != t_entry.stats.get("Shape", ""):
        return "diff"
    for e_val, t_val in (
        (e_entry.output_hash, t_entry.output_hash),
        (e_entry.input_hashes, t_entry.input_hashes),
    ):
        rd = _hash_intensity(e_val, t_val)
        if rd is not None and rd > 1e-8:
            return "diff"
    return "match"


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


def match_entries(
    eager: list[OpEntry],
    traced: list[OpEntry],
    overrides: dict[str, str] | None = None,
):
    """Pair up ops between two logs using a four-pass matching strategy.

    Cross-mode comparisons (eager vs aot_fx_trace, etc.)
    can't rely on identical keys: op counters drift when ops are
    interleaved differently, and some ops have different module
    attributions between modes (e.g. gradient accumulation adds may
    surface under different FQNs). The passes progressively relax the
    matching criteria so that ops still get paired:

    Pass 0 — Manual overrides (only if ``overrides`` is provided).
        Force-pairs the named keys regardless of stats / signatures.
        Use this to fix-up specific cases where the FQN scheme drifts
        between runs in a way the automated passes can't recover (e.g.
        an AC-recomputed op surfacing as bare ``feed_forward/op_7_mul``
        in eager but ``layers.2.feed_forward/op_3_mul`` in traced).

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
        overrides: Optional ``{eager_key: traced_key}`` dict.  Each
            entry forces a pair regardless of stat similarity; any
            entry whose target key isn't present on the traced side is
            silently ignored (so override files can outlive log
            regenerations).

    Returns:
        List of ``(eager_entry | None, traced_entry | None, status,
        diffs, match_strategy)`` tuples. ``status`` is one of
        ``"match"``, ``"diff"``, ``"eager_only"``, ``"traced_only"``.
        ``match_strategy`` records which pass paired the entries:
        ``"override"``, ``"exact"``, ``"fuzzy"``, ``"stats"``, or
        ``""`` for only-side rows. Eager-side entries appear in their
        original log order; ``traced_only`` entries follow at the end
        in their original traced-log order.
    """
    traced_used = set()
    overrides = overrides or {}

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
                    status = _compute_match_status(e_entry, c)
                    eager_results[i] = (status, diffs, c, strategy)
                    break

    # Pass 0: manual overrides.  Each entry forces a specific
    # eager_key -> traced_key pair and consumes both sides so later
    # passes don't reassign them.  Index traced by raw key for lookup.
    if overrides:
        traced_by_key: dict[str, list[OpEntry]] = {}
        for t in traced:
            traced_by_key.setdefault(t.key, []).append(t)
        for i, e_entry in enumerate(eager):
            target_key = overrides.get(e_entry.key)
            if target_key is None:
                continue
            for c in traced_by_key.get(target_key, []):
                if id(c) not in traced_used:
                    traced_used.add(id(c))
                    diffs = _compute_diffs(e_entry, c)
                    status = _compute_match_status(e_entry, c)
                    eager_results[i] = (status, diffs, c, "override")
                    break

    # Pass 1: exact match (forward ops, where keys agree).  The root
    # class prefix is normalized here too — DebugMode eager keys are
    # rooted at the model class (``FSDPLlama3Model.``), traced keys
    # are not, so a literal e.key equality would never match across
    # backends.
    _try_pass("exact", lambda e: _strip_root_class_prefix(e.key))

    # Pass 2: fuzzy match (module_fqn + op_type, counter stripped).
    _try_pass("fuzzy", lambda e: _fuzzy_key(e.key))

    # Pass 3: stats match (op_type + Shape + output L1-norm hash) —
    # handles ops whose FQN differs between modes.  Uses ``output_hash``
    # rather than the now-removed ``L1 norm`` stat field; both encode
    # the same float64 L1 reduction, so the key is unchanged in spirit.
    def _stats_key(entry: OpEntry) -> tuple[str, str, str]:
        op_type = (
            _fuzzy_key(entry.key).split("/")[-1] if "/" in entry.key else entry.key
        )
        return (op_type, entry.stats.get("Shape", ""), entry.output_hash)

    _try_pass("stats", _stats_key)

    # Emit eager entries in their original order. Anything still None
    # is eager_only.  Typed as ``list[tuple]`` so pyrefly doesn't narrow
    # element type to the first append and reject the others.
    results: list[tuple] = []
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
        f"by <code>_EXCLUDED_OPS</code>)</span></div>"
        f"{_skipped_block(name1, skipped1)}"
        f"{_skipped_block(name2, skipped2)}"
        f"</div>"
    )

    def _phase(e, t):
        return e.phase if e else t.phase

    fwd = [r for r in results if _phase(r[0], r[1]) == "forward"]
    bwd = [r for r in results if _phase(r[0], r[1]) == "backward"]

    total_match = sum(1 for r in results if r[2] == "match")
    total_diff = sum(1 for r in results if r[2] == "diff")
    total_eager = sum(1 for r in results if r[2] == "eager_only")
    total_traced = sum(1 for r in results if r[2] == "traced_only")

    def make_table(entries, section_id):
        rows: list[str] = []
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
            eager_cells: list[str] = []
            traced_cells: list[str] = []
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

            # Hash cells: input and output norm hashes from
            # log_tensor_hashes.  Comma-separated lists of floats — we
            # compare element-wise with the same log-scale relative-
            # diff intensity used for the L1/L2/Min/Max columns, so a
            # last-ULP drift (~1e-7) renders as nearly invisible faint
            # pink and a true divergence (~1e-1) renders as full red.
            # A structural mismatch (different list length) renders as
            # full red — that genuinely means the op signatures differ.
            #
            # For input hashes we additionally wrap each individual
            # value in a <span title="..."> when a producer key is
            # available — hovering shows "produced by <op key>".
            def _format_hash_values(val_str: str, prod_str: str) -> str:
                if not val_str:
                    return ""
                vals = [v.strip() for v in val_str.split(",")]
                prods = [p.strip() for p in prod_str.split(";")] if prod_str else []
                parts: list[str] = []
                for i, v in enumerate(vals):
                    prod = prods[i] if i < len(prods) else ""
                    if prod:
                        # Both the CSS tooltip (data-tooltip via ::after)
                        # and native title are emitted — title is the
                        # universal fallback if CSS gets stripped by the
                        # host environment.
                        tip = html.escape(f"produced by {prod}")
                        parts.append(
                            f'<span class="prod-link"'
                            f' data-tooltip="{tip}"'
                            f' title="{tip}">'
                            f"{html.escape(v)}</span>"
                        )
                    else:
                        parts.append(html.escape(v))
                return ", ".join(parts)

            def _hash_cell(
                e_val: str,
                t_val: str,
                e_prod: str = "",
                t_prod: str = "",
                collapsible: bool = False,
            ):
                e_inner = _format_hash_values(e_val, e_prod)
                t_inner = _format_hash_values(t_val, t_prod)
                if collapsible:
                    # Wrap in a clickable div that starts collapsed
                    # (max-height limited) and expands on click.  Lets
                    # rows with hundreds of input hashes (e.g.
                    # _fused_adamw_) stay readable.
                    def _wrap(inner: str) -> str:
                        if not inner:
                            return ""
                        return (
                            '<div class="ih-cell collapsed"'
                            ' onclick="toggleIH(this)">'
                            f"{inner}</div>"
                        )

                    e_inner = _wrap(e_inner)
                    t_inner = _wrap(t_inner)
                if not (e and t):
                    return f"<td>{e_inner}</td>", f"<td>{t_inner}</td>"
                rd = _hash_intensity(e_val, t_val)
                if rd is None or rd == 0.0:
                    return f"<td>{e_inner}</td>", f"<td>{t_inner}</td>"
                if rd >= 0.1:
                    t_val_intensity = 1.0
                elif rd <= 1e-8:
                    t_val_intensity = 0.0
                else:
                    t_val_intensity = (math.log10(rd) + 8) / 7
                r = int(220 + (248 - 220) * t_val_intensity)
                g = int(220 - (220 - 80) * t_val_intensity)
                b = int(220 - (220 - 80) * t_val_intensity)
                bg_alpha = t_val_intensity * 0.15
                style = (
                    f' style="color: rgb({r},{g},{b});'
                    f" background: rgba(248,80,80,{bg_alpha:.2f});"
                    f' font-weight: bold"'
                )
                return f"<td{style}>{e_inner}</td>", f"<td{style}>{t_inner}</td>"

            e_oh = e.output_hash if e else ""
            t_oh = t.output_hash if t else ""
            e_oh_cell, t_oh_cell = _hash_cell(e_oh, t_oh)
            e_ih = e.input_hashes if e else ""
            t_ih = t.input_hashes if t else ""
            e_ip = e.input_producers if e else ""
            t_ip = t.input_producers if t else ""
            e_ih_cell, t_ih_cell = _hash_cell(e_ih, t_ih, e_ip, t_ip, collapsible=True)

            strategy_label = {
                "override": "manual override",
                "exact": "same op key",
                "fuzzy": "fuzzy op key",
                "stats": "stats",
            }.get(strategy, "")
            strategy_html = (
                f'<span class="strategy strategy-{strategy}">{strategy_label}</span>'
                if strategy
                else ""
            )

            rows.append(
                f"""
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="key-cell" rowspan="2">{key_html}</td>
              <td class="match-cell" rowspan="2">{strategy_html}</td>
              <td class="side-label">{name1_safe}</td>
              {eager_cells[0]}
              {e_oh_cell}
              {e_ih_cell}
              <td>{e_loc}</td>
              <td>{e_phase}</td>
              {''.join(eager_cells[1:])}
            </tr>
            <tr class="op-row {cls}" data-row-id="{row_id}"
                onmouseenter="highlight('{row_id}')"
                onmouseleave="unhighlight('{row_id}')">
              <td class="side-label">{name2_safe}</td>
              {traced_cells[0]}
              {t_oh_cell}
              {t_ih_cell}
              <td>{t_loc}</td>
              <td>{t_phase}</td>
              {''.join(traced_cells[1:])}
            </tr>"""
            )

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
  .key-cell {{
    font-weight: bold; color: #7ec8e3;
    vertical-align: top; border-right: 2px solid #0f3460;
    max-width: 400px; overflow: hidden;
    text-overflow: ellipsis; line-height: 1.6;
  }}
  .key-eager {{ color: #c4b5fd; font-size: 11px; }}
  .key-traced {{ color: #67e8f9; font-size: 11px; }}
  .side-label {{ color: #888; font-size: 11px; width: 50px; }}
  .prod-link {{
    cursor: help;
    border-bottom: 1px dotted #58a6ff;
    position: relative;
  }}
  .prod-link:hover {{ background: rgba(88,166,255,0.18); }}
  /* CSS tooltip via ::after — the browser's native title attribute is
     unreliable inside iframes (Pixelcloud) and has a multi-second
     reveal delay.  We render the producer key as a styled bubble that
     appears instantly. */
  .prod-link::after {{
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 4px);
    left: 0;
    background: #0f3460;
    color: #c4b5fd;
    border: 1px solid #58a6ff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: normal;
    white-space: nowrap;
    z-index: 100;
    pointer-events: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    opacity: 0;
    transition: opacity 0.08s ease-in;
  }}
  .prod-link:hover::after {{ opacity: 1; }}
  /* Collapsible input-hashes cell.  Default state shows ~3 lines and
     clips the rest; clicking expands.  word-break: break-all lets long
     comma-separated lists wrap mid-token so the column stays narrow. */
  .ih-cell {{
    /* Min-width fits ~2 hashes per line (each value is ~13 chars in
       12px mono ~ 90px + ", " separator).  Max-width caps growth so a
       295-entry _fused_adamw_ row doesn't push the rest of the table
       off-screen.  overflow-wrap: anywhere lets long lists wrap at
       commas naturally but still break a single oversized token if it
       would otherwise overflow. */
    min-width: 280px;
    max-width: 520px;
    white-space: normal;
    word-break: normal;
    overflow-wrap: anywhere;
    line-height: 1.5;
    cursor: pointer;
  }}
  .ih-cell.collapsed {{ max-height: 3em; overflow: hidden; position: relative; }}
  .ih-cell.collapsed::after {{
    content: '▾ click to expand';
    position: absolute;
    bottom: 0; right: 0;
    font-size: 10px;
    color: #58a6ff;
    background: rgba(15,22,41,0.92);
    padding: 0 4px;
    border-radius: 3px;
  }}
  .ih-cell.expanded::after {{
    content: '▴ click to collapse';
    display: inline-block;
    margin-top: 4px;
    font-size: 10px;
    color: #58a6ff;
  }}
  .match-cell {{ vertical-align: middle; border-right: 2px solid #0f3460; }}
  .strategy {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .strategy-exact {{ color: #4ade80; background: rgba(74,222,128,0.15); }}
  .strategy-fuzzy {{ color: #fbbf24; background: rgba(251,191,36,0.15); }}
  .strategy-stats {{ color: #a78bfa; background: rgba(167,139,250,0.15); }}
  .strategy-override {{ color: #fb7185; background: rgba(251,113,133,0.18); }}
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
  .skipped-chip {{
    display: inline-block; background: rgba(167,139,250,0.15);
    color: #a78bfa; font-size: 11px; padding: 2px 6px;
    border-radius: 4px; margin-right: 4px;
  }}
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
      <tr>
        <th>Op</th><th>Match method</th><th>Side</th><th>Shape</th>
        <th>Output L1 norm</th><th>Input L1 norm</th>
        <th>Location</th><th>Phase</th>
        <th>Output L2 norm</th><th>Output Min</th>
        <th>Output Max</th><th>Output Mean</th>
      </tr>
      {make_table(fwd, "fwd")}
    </table>
  </div>
</div>

<div class="section">
  <h2 onclick="toggleSection('bwd')">Backward ({len(bwd)} ops) <span class="toggle" id="bwd-toggle">▼</span></h2>
  <div class="section-content" id="bwd">
    <table>
      <tr>
        <th>Op</th><th>Match method</th><th>Side</th><th>Shape</th>
        <th>Output L1 norm</th><th>Input L1 norm</th>
        <th>Location</th><th>Phase</th>
        <th>Output L2 norm</th><th>Output Min</th>
        <th>Output Max</th><th>Output Mean</th>
      </tr>
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
function toggleIH(el) {{
  el.classList.toggle('collapsed');
  el.classList.toggle('expanded');
}}
</script>
</body>
</html>"""


def main():
    """CLI entry point: parse two logs, match entries, write HTML diff.

    Invoked via ``python -m torchtitan.tools.compare_numerics`` or
    directly. See module docstring for the typical workflow.
    """
    parser = argparse.ArgumentParser(description="Compare numerics activation logs")
    parser.add_argument("eager", help="First log file")
    parser.add_argument("traced", help="Second log file")
    parser.add_argument(
        "-o", "--output", default="numerics_diff.html", help="Output HTML file"
    )
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
    parser.add_argument(
        "--override",
        default=None,
        help=(
            "Path to a CSV of forced matches: each non-empty line is "
            "'<log1_key>, <log2_key>'.  Applied before the automated "
            "matching passes; matching pairs are rendered with the "
            "'manual override' strategy chip.  Lines starting with '#' "
            "are ignored."
        ),
    )
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    if args.override:
        with open(args.override) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                overrides[parts[0].strip()] = parts[1].strip()
        print(f"Loaded {len(overrides)} override(s) from {args.override}")

    eager, skipped_eager = parse_log(args.eager)
    traced, skipped_traced = parse_log(args.traced)
    results = match_entries(eager, traced, overrides=overrides)
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
