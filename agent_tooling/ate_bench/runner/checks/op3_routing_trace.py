"""OP3 check: per-token MoE routing trace npz files are valid.

Paper (B.2.2): verify that the expected per-step ``step-<id:08d>.npz`` files are
present and each carries expert-ID and gating-weight arrays of the correct shape
for every MoE layer over every token in the global batch; expert-IDs within
``[0, num_experts)``; gating weights non-negative and summing to 1 over the top-k
selected experts for every token.

Expected schema per .npz (see op3_collect_routing_trace.md):
  layer{L:02d}_expert_ids      int   [num_tokens, top_k]
  layer{L:02d}_gating_weights  float [num_tokens, top_k]
  num_experts                  scalar

Default expected step count is 4: 8M tokens / (global_batch 1024 * seq_len 2048)
= 4 steps.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checks import CheckResult
else:
    from . import CheckResult

_LAYER_RE = re.compile(r"layer(\d+)_expert_ids")


def _check_one_npz(path: Path, sum_tol: float) -> tuple[bool, str]:
    try:
        z = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"{path.name}: cannot load npz ({exc})"
    keys = set(z.files)
    if "num_experts" not in keys:
        return False, f"{path.name}: missing 'num_experts' scalar"
    num_experts = int(np.asarray(z["num_experts"]).reshape(-1)[0])
    layers = sorted(int(m.group(1)) for k in keys if (m := _LAYER_RE.fullmatch(k)))
    if not layers:
        return False, f"{path.name}: no 'layer{{L:02d}}_expert_ids' arrays found"

    n_tokens_ref = None
    for layer in layers:
        ids_key = f"layer{layer:02d}_expert_ids"
        w_key = f"layer{layer:02d}_gating_weights"
        if w_key not in keys:
            return False, f"{path.name}: layer {layer} missing {w_key}"
        ids = np.asarray(z[ids_key])
        w = np.asarray(z[w_key])
        if ids.ndim != 2 or w.ndim != 2:
            return False, f"{path.name}: layer {layer} arrays must be 2D [tokens, top_k]"
        if ids.shape != w.shape:
            return False, f"{path.name}: layer {layer} id/weight shape mismatch {ids.shape} vs {w.shape}"
        n_tokens, top_k = ids.shape
        if n_tokens_ref is None:
            n_tokens_ref = n_tokens
        elif n_tokens != n_tokens_ref:
            return False, f"{path.name}: layer {layer} token count {n_tokens} != {n_tokens_ref}"
        if ids.min() < 0 or ids.max() >= num_experts:
            return False, (
                f"{path.name}: layer {layer} expert id out of range "
                f"[0,{num_experts}): min={ids.min()} max={ids.max()}"
            )
        if w.min() < -sum_tol:
            return False, f"{path.name}: layer {layer} has negative gating weight {w.min()}"
        sums = w.astype(np.float64).sum(axis=1)
        if not np.allclose(sums, 1.0, atol=sum_tol):
            bad = int(np.count_nonzero(np.abs(sums - 1.0) > sum_tol))
            return False, (
                f"{path.name}: layer {layer} gating weights do not sum to 1 for "
                f"{bad}/{n_tokens} tokens (e.g. {sums[np.argmax(np.abs(sums-1.0))]:.4f})"
            )
    return True, f"{path.name}: {len(layers)} layers, {n_tokens_ref} tokens OK"


def check(
    routing_traces_dir: str | Path,
    expected_steps: int = 4,
    sum_tol: float = 1e-3,
) -> CheckResult:
    d = Path(routing_traces_dir)
    if not d.is_dir():
        return CheckResult(False, f"routing-traces dir not found: {d}")
    files = sorted(d.glob("step-*.npz"))
    if len(files) < expected_steps:
        return CheckResult(
            False,
            f"expected >= {expected_steps} step-*.npz files, found {len(files)}",
            {"files": [f.name for f in files]},
        )
    details = []
    for f in files:
        ok, msg = _check_one_npz(f, sum_tol)
        details.append(msg)
        if not ok:
            return CheckResult(False, msg, {"per_file": details})
    return CheckResult(
        True,
        f"{len(files)} routing-trace files valid",
        {"per_file": details},
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("routing_traces_dir", help="dir containing step-*.npz")
    ap.add_argument("--expected-steps", type=int, default=4)
    ap.add_argument("--sum-tol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    res = check(args.routing_traces_dir, args.expected_steps, args.sum_tol)
    print(("PASS " if res.passed else "FAIL ") + res.detail)
    for line in res.extra.get("per_file", []):
        print("  ", line)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
