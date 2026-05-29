"""Structured failure taxonomy and idea-family time-boxing.

Post-mortem findings 1e/1f/1g: almost every crash was a *toolchain* integration
failure (Inductor post-grad RecursionError, ``triton_kernel_wrapper_functional
is not an OpOverload``, TorchAO dim1 128-row tile assert, Flex/CUTE API
mismatches), not a parallelism mistake — and the Flex/CUTE family alone burned
~42 runs (~6% of the budget) with no mechanism to give up. This module turns a
run log into a structured (crash_class, stage) verdict and tracks per-family
attempt budgets so the driver auto-defers a family that keeps failing the same
way ("repeated failures for the same idea should be abandoned quickly").
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# Ordered: first match wins, most specific first. Each entry maps a class name to
# a signature regex over the (rank-prefixed) log text.
_CLASS_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("OOM", re.compile(r"OutOfMemoryError|CUDA out of memory|num_ooms|allocator", re.I)),
    ("INDUCTOR_RECURSION", re.compile(r"RecursionError|maximum recursion", re.I)),
    ("INDUCTOR_CODEGEN", re.compile(
        r"triton_kernel_wrapper_functional is not an OpOverload|"
        r"reinplace_inplaceable_ops|post_grad_passes|fake_tensor", re.I)),
    ("TRITON_TILE", re.compile(r"n_rows % 128|tile|dim1|to_mxfp8|invalid argument", re.I)),
    ("FLEX_CUTE_API", re.compile(r"flash_attn\.cute|cute|flex_attention|FlexAttention", re.I)),
    ("NCCL", re.compile(r"NCCL|ncclInternalError|nccl error|fabric|NVLS", re.I)),
    ("COMPILE", re.compile(r"torch\._dynamo|Inductor|fullgraph|graph break|BackendCompiler", re.I)),
    ("TIMEOUT", re.compile(r"timed out|Watchdog|timeout|hang", re.I)),
    ("DATALOADER", re.compile(r"DataLoader|DataloaderExhausted|dataset", re.I)),
]

# Stage in the train flow where the failure surfaced (best-effort, from frames).
_STAGE_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("model_build", re.compile(r"meta|init_weights|materializ|build", re.I)),
    ("parallelize", re.compile(r"parallelize_qwen3|fully_shard|apply_compile|apply_ac", re.I)),
    ("forward", re.compile(r"\bforward\b|model_parts\[0\]\(", re.I)),
    ("backward", re.compile(r"\.backward\(|autograd|grad_fn", re.I)),
    ("loss", re.compile(r"loss_fn|ChunkedCELoss|cross_entropy", re.I)),
    ("optimizer", re.compile(r"optimizer\.step|clip_grad", re.I)),
]


@dataclass
class CrashVerdict:
    crash_class: str
    stage: str
    is_substrate: bool  # toolchain/integration vs. a real parallelism logic error
    excerpt: str


# Classes that mean "the substrate broke", not "the candidate's idea is wrong".
_SUBSTRATE = {
    "INDUCTOR_RECURSION", "INDUCTOR_CODEGEN", "TRITON_TILE",
    "FLEX_CUTE_API", "COMPILE", "NCCL",
}


def classify(log_text: str) -> CrashVerdict:
    """Classify a failed run's log into (crash_class, stage)."""
    cls = "UNKNOWN"
    for name, rx in _CLASS_SIGNATURES:
        if rx.search(log_text):
            cls = name
            break
    stage = "unknown"
    for name, rx in _STAGE_SIGNATURES:
        if rx.search(log_text):
            stage = name
            break
    m = re.search(r"(Error|Exception|assert)[^\n]{0,160}", log_text, re.I)
    excerpt = m.group(0).strip() if m else ""
    return CrashVerdict(cls, stage, cls in _SUBSTRATE, excerpt)


@dataclass
class FamilyBudget:
    """Tracks consecutive same-class failures per idea family.

    A "family" is the agent's idea tag (e.g. ``flex-cute``, ``mxfp8-enable``).
    When the same family fails the same substrate class ``max_substrate`` times
    in a row, the driver should auto-defer it to a blocked-on-substrate list
    rather than keep paying full GPU runs to rediscover the same wall.
    """

    max_substrate: int = 3
    max_other: int = 4
    _streak: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    deferred: set[str] = field(default_factory=set)

    def record(self, family: str, verdict: CrashVerdict | None) -> str:
        """Record one outcome; return action: ``continue`` | ``defer``.

        ``verdict`` is None on success, which resets the family's streak.
        """
        if verdict is None:
            for key in list(self._streak):
                if key[0] == family:
                    self._streak[key] = 0
            return "continue"
        key = (family, verdict.crash_class)
        self._streak[key] += 1
        cap = self.max_substrate if verdict.is_substrate else self.max_other
        if self._streak[key] >= cap:
            self.deferred.add(family)
            return "defer"
        return "continue"

    def is_deferred(self, family: str) -> bool:
        return family in self.deferred
