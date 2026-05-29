"""Reductions and tolerance logic for the single-batch numerical probe.

We never store reference tensors raw: at seq128/batch176 the logits are ~13.7GB
and ``lm_head.weight.grad`` is ~3.1GB. Instead we keep magnitude (L2 norm,
scalar loss/grad_norm) and direction (a feature-hashing sketch that preserves
cosine), so a snapshot is a few hundred KB. All comparisons are differential
against a frozen high-precision golden and, optionally, the same-config
champion; nothing here depends on the dataset being realistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Multiplicative-hash constants for the chunk-invariant feature-hashing sketch.
# Using a pure index hash (not an RNG stream) makes the projection identical
# across processes and chunk sizes, which is what lets capture and compare agree.
_P_BUCKET = 2654435761
_P_SIGN = 2246822519


def sketch(flat: torch.Tensor, out_dim: int, seed: int) -> torch.Tensor:
    """Signed feature-hashing projection of a flat tensor to ``out_dim`` dims.

    Each element ``i`` is hashed to a bucket and a sign and accumulated; this is
    an unbiased random projection so cosine of two sketches estimates cosine of
    the originals. Runs chunked in float64 to bound memory and keep huge
    gradients (hundreds of millions of elements) numerically stable.
    """
    out = torch.zeros(out_dim, dtype=torch.float64, device=flat.device)
    flat = flat.reshape(-1)
    n = flat.numel()
    chunk = 1 << 24
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        idx = torch.arange(s, e, device=flat.device, dtype=torch.int64) + seed
        bucket = (idx * _P_BUCKET) % out_dim
        sign = torch.where((idx * _P_SIGN) % 2 == 0, 1.0, -1.0).to(torch.float64)
        out.index_add_(0, bucket, flat[s:e].to(torch.float64) * sign)
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).reshape(-1)
    b = b.to(torch.float64).reshape(-1)
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return float((a @ b) / denom)


def sqnr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB of ``cand`` against ``ref``.

    Higher is closer. Computed over a fixed sampled subset of logits so it
    never needs the full reference logits tensor materialized or stored.
    """
    ref = ref.to(torch.float64).reshape(-1)
    cand = cand.to(torch.float64).reshape(-1)
    noise = (ref - cand).norm() ** 2
    if noise == 0:
        return math.inf
    return float(10.0 * torch.log10((ref.norm() ** 2) / noise))


def relerr(a: float, b: float) -> float:
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


@dataclass
class Tolerance:
    """Per-class acceptance thresholds.

    ``vs_champion_*`` gate the increment against the same-config champion;
    ``vs_golden_sqnr_db`` is the absolute drift floor against the frozen
    high-precision golden. Defaults are seeds for calibration, not gospel:
    they should be re-derived from the golden's own seed/reduction-order spread.
    """

    vs_champion_cosine_min: float
    vs_champion_loss_relerr_max: float
    vs_champion_gradnorm_relerr_max: float
    vs_golden_sqnr_db_min: float
    vs_golden_cosine_min: float  # only enforced when no champion is available
    requires_gradcheck: bool = False


TOLERANCES: dict[str, Tolerance] = {
    # Scheduling/wrapping/compile-boundary changes must not move the math:
    # near-identical to the same-config champion (only ulp-level collective noise).
    "schedule": Tolerance(0.9999, 1e-3, 1e-3, 18.0, 0.9999),
    # Reduction-order changes (loss chunking, FSDP wrap, reduce dtype) shift bits
    # but not direction; allow a slightly looser band.
    "reduction": Tolerance(0.999, 5e-3, 5e-3, 18.0, 0.999),
    # Precision changes (FP8/MXFP8) diverge from the bf16 golden by design; the
    # SQNR floor is the primary guard, with a wide cosine band.
    "precision": Tolerance(0.95, 0.05, 0.10, 22.0, 0.95),
    # Custom kernels add the standalone gradcheck on top of the precision band.
    "kernel": Tolerance(0.95, 0.05, 0.10, 22.0, 0.95, requires_gradcheck=True),
}


def select_param_names(
    named: list[tuple[str, torch.Tensor]], patterns: tuple[str, ...]
) -> list[str]:
    """Resolve fingerprint params by substring, falling back to largest 2D.

    Keeping selection robust means a candidate that renames an internal module
    still gets verified on comparably important weights rather than silently
    fingerprinting nothing.
    """
    chosen: list[str] = []
    for pat in patterns:
        match = next((n for n, _ in named if pat in n), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    if not chosen:
        twod = sorted(
            (p.numel(), n) for n, p in named if p.dim() == 2
        )
        chosen = [n for _, n in twod[-3:]]
    return chosen


def compare(
    snap: dict, golden: dict, champion: dict | None, cls: str
) -> tuple[bool, dict]:
    """Differential verdict for a candidate snapshot. Returns (passed, metrics).

    Always enforces the absolute SQNR floor against the golden. When a champion
    snapshot is present, enforces the per-class incremental tolerance against it;
    otherwise falls back to the golden cosine band.
    """
    tol = TOLERANCES[cls]
    m: dict = {"cls": cls, "checks": {}}
    ok = True

    if snap["fingerprint"] != golden["fingerprint"]:
        return False, {
            "cls": cls,
            "error": "fingerprint mismatch (shape/model differs from golden)",
            "candidate": snap["fingerprint"],
            "golden": golden["fingerprint"],
        }

    logit_sqnr = sqnr_db(golden["logit_sample"], snap["logit_sample"])
    m["vs_golden_sqnr_db"] = round(logit_sqnr, 2)
    if logit_sqnr < tol.vs_golden_sqnr_db_min:
        ok = False
    m["checks"]["golden_sqnr"] = logit_sqnr >= tol.vs_golden_sqnr_db_min

    ref = champion if champion is not None else golden
    anchor = "champion" if champion is not None else "golden"
    m["anchor"] = anchor

    cos_min = 1.0
    for name, sk in snap["grads"].items():
        if name not in ref["grads"]:
            continue
        c = cosine(torch.as_tensor(sk), torch.as_tensor(ref["grads"][name]))
        cos_min = min(cos_min, c)
    m["grad_cosine_min"] = round(cos_min, 6)

    loss_re = relerr(snap["loss"], ref["loss"])
    gn_re = relerr(snap["grad_norm"], ref["grad_norm"])
    m["loss_relerr"] = round(loss_re, 6)
    m["grad_norm_relerr"] = round(gn_re, 6)

    if champion is not None:
        cos_pass = cos_min >= tol.vs_champion_cosine_min
        loss_pass = loss_re <= tol.vs_champion_loss_relerr_max
        gn_pass = gn_re <= tol.vs_champion_gradnorm_relerr_max
    else:
        cos_pass = cos_min >= tol.vs_golden_cosine_min
        loss_pass = gn_pass = True  # bf16-vs-quantized loss differs by design
    m["checks"]["grad_cosine"] = cos_pass
    m["checks"]["loss_relerr"] = loss_pass
    m["checks"]["grad_norm_relerr"] = gn_pass
    ok = ok and cos_pass and loss_pass and gn_pass

    if not math.isfinite(snap["loss"]) or not math.isfinite(snap["grad_norm"]):
        ok = False
        m["checks"]["finite"] = False
    else:
        m["checks"]["finite"] = True

    m["passed"] = ok
    m["requires_gradcheck"] = tol.requires_gradcheck
    return ok, m
