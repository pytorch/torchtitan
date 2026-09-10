"""Independent LLM judge for new-feature rule-based correctness (paper B.3.2).

Each new-feature attempt is judged by an *independent* agent session given the
three fixed rules and the agent's ``git diff`` against ``main``; the judge returns
PASS/FAIL per rule (overall PASS iff all three pass). The paper used a
``claude-opus-4-7`` session at xhigh effort. The judge sees only the rules and the
diff — not the repo, and not the implementing agent's reasoning.

Returns: {"overall": "PASS"|"FAIL", "rules": [{"id","verdict","reason"}, ...], ...}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Keep the diff within the judge's context; note truncation if it exceeds this.
MAX_DIFF_CHARS = 200_000

_PROMPT = """\
You are an independent, skeptical code reviewer. You are given THREE required \
rules that a code change must satisfy, and a git diff. Decide PASS or FAIL for \
each rule STRICTLY from the diff: mark PASS only if the diff clearly implements \
the rule; if the evidence is absent or you are uncertain, mark FAIL.

Respond with ONLY a JSON object (no prose, no code fence) of exactly this shape:
{{"rules":[{{"id":1,"verdict":"PASS","reason":"..."}},{{"id":2,"verdict":"FAIL","reason":"..."}},{{"id":3,"verdict":"PASS","reason":"..."}}],"overall":"FAIL"}}
"overall" is "PASS" iff all three rules are "PASS".

=== RULES ===
{rules}

=== GIT DIFF ===
{diff}
"""


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Try to salvage the largest valid {...} via a non-greedy scan.
        for m in re.finditer(r"\{.*\}", text[start : end + 1], re.DOTALL):
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def judge_feature(
    rules_text: str,
    diff_text: str,
    model: str | None = None,
    timeout: int = 600,
) -> dict:
    truncated = False
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n... [diff truncated] ...\n"
        truncated = True

    prompt = _PROMPT.format(rules=rules_text.strip(), diff=diff_text.strip())
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout
    )
    raw_text = ""
    if proc.stdout.strip():
        try:
            raw_text = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            raw_text = proc.stdout
    verdict = _extract_json(raw_text)
    if verdict is None:
        return {
            "overall": "ERROR",
            "rules": [],
            "error": "could not parse judge JSON",
            "raw": raw_text[:2000],
            "diff_truncated": truncated,
        }
    # Normalize / enforce overall = all PASS.
    rules = verdict.get("rules", [])
    all_pass = bool(rules) and all(
        str(r.get("verdict", "")).upper() == "PASS" for r in rules
    )
    verdict["overall"] = "PASS" if all_pass else "FAIL"
    verdict["diff_truncated"] = truncated
    return verdict


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", required=True, type=Path, help="rules markdown file")
    ap.add_argument("--diff", required=True, type=Path, help="git diff file (or '-' for stdin)")
    ap.add_argument("--model", default=None, help="judge model (paper: claude-opus-4-7 @ xhigh)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    rules_text = args.rules.read_text(encoding="utf-8")
    diff_text = sys.stdin.read() if str(args.diff) == "-" else args.diff.read_text(errors="ignore")
    verdict = judge_feature(rules_text, diff_text, args.model, args.timeout)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
