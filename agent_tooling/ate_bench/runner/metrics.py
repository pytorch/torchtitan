"""Parse Claude Code stream-json transcripts into ATE-Bench effort metrics.

ATE-Bench (PithTrain, arXiv:2605.31463) reports five *effort* metrics per task:
session duration, active GPU time, agent turns, per-turn context, and output
tokens. For the Q&A category only agent turns / per-turn context / output tokens
are reported (paper Table 5); active GPU time is N/A because Q&A does not run
training.

Computed from the JSONL emitted by:
    claude -p --output-format stream-json --verbose

IMPORTANT — where the numbers come from. The streamed per-message ``usage`` blocks
are *partial snapshots*: one assistant turn shows up as several ``assistant``
events sharing a ``message.id``, and their ``output_tokens`` are tiny incremental
values that do NOT sum to the real total. The authoritative, billed usage is the
final ``result`` event's ``modelUsage`` (this is what cost is computed from). So:

    agent_turns      = result.num_turns
    output_tokens    = sum(modelUsage[*].outputTokens)
    per_turn_context = (sum of input-side tokens across all turns) / num_turns
                       where input-side = inputTokens + cacheReadInputTokens
                                          + cacheCreationInputTokens
    session_duration = result.duration_ms / 1000

We cross-check agent_turns against the count of unique assistant ``message.id``s.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_FINAL_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)


def _extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def _model_usage_totals(model_usage: dict) -> tuple[int, int]:
    """Return (total_output_tokens, total_input_side_tokens) from modelUsage."""
    out = 0
    inp = 0
    for v in (model_usage or {}).values():
        out += int(v.get("outputTokens", 0) or 0)
        inp += (
            int(v.get("inputTokens", 0) or 0)
            + int(v.get("cacheReadInputTokens", 0) or 0)
            + int(v.get("cacheCreationInputTokens", 0) or 0)
        )
    return out, inp


@dataclass
class TaskMetrics:
    # The three Q&A metrics (paper Table 5):
    agent_turns: int | None
    per_turn_context: float | None  # mean input-side context per turn
    output_tokens: int  # total billed output tokens
    # The other two ATE metrics (reported for non-Q&A categories):
    session_duration_s: float | None
    active_gpu_time_s: float | None  # None for Q&A (no training run)
    # Provenance / cross-checks (not paper metrics):
    unique_assistant_messages: int = 0
    total_cost_usd: float | None = None
    model: str | None = None
    is_error: bool | None = None
    final_answer: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_transcript(path: str | Path) -> TaskMetrics:
    """Parse one stream-json transcript file into a :class:`TaskMetrics`."""
    message_ids: set[str] = set()
    result: dict | None = None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "assistant":
                mid = (ev.get("message") or {}).get("id")
                if mid:
                    message_ids.add(mid)
            elif etype == "result":
                result = ev

    n_unique = len(message_ids)
    if result is None:
        # No final result event (timeout/crash mid-stream). Fall back to what we
        # can: turn count from unique messages; totals unknown.
        return TaskMetrics(
            agent_turns=n_unique or None,
            per_turn_context=None,
            output_tokens=0,
            session_duration_s=None,
            active_gpu_time_s=None,
            unique_assistant_messages=n_unique,
            is_error=True,
        )

    num_turns = result.get("num_turns") or n_unique or None
    duration_ms = result.get("duration_ms")
    duration_s = (duration_ms / 1000.0) if duration_ms else None
    model_usage = result.get("modelUsage") or {}
    out_tokens, in_side = _model_usage_totals(model_usage)
    per_turn_context = (in_side / num_turns) if (in_side and num_turns) else None
    model = next(iter(model_usage.keys()), None)

    return TaskMetrics(
        agent_turns=num_turns,
        per_turn_context=per_turn_context,
        output_tokens=out_tokens,
        session_duration_s=duration_s,
        active_gpu_time_s=None,
        unique_assistant_messages=n_unique,
        total_cost_usd=result.get("total_cost_usd"),
        model=model,
        is_error=result.get("is_error"),
        final_answer=_extract_final_answer(result.get("result")),
    )


def _k(x: float | None) -> str:
    return "-" if x is None else f"{x / 1000:.1f}K"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Parse a Claude stream-json transcript")
    ap.add_argument("transcript", help="path to a *.jsonl stream-json transcript")
    args = ap.parse_args(argv)

    m = parse_transcript(args.transcript)
    print(json.dumps(m.to_dict(), indent=2))
    dur = f"{m.session_duration_s:.0f}s" if m.session_duration_s else "-"
    print(
        f"\nAgent Turns: {m.agent_turns}  |  Per-Turn Context: "
        f"{_k(m.per_turn_context)}  |  Output Tokens: {_k(m.output_tokens)}  |  "
        f"Duration: {dur}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
