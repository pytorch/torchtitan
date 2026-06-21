# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WSD-S / Chinchilla scaling policy, ported from OLMo-core.

Faithful port of ``WSDSChinchillaRunConfigurator`` (see DESIGN.md References).
``N`` denotes ladder (non-embedding) parameters throughout; every token budget
is converted to steps by the planner.
"""

import math
from dataclasses import dataclass

# The policy fields overridable per run (the hillclimb knobs). Single source of
# truth for override validation, the run-dir slug, and the CLI flags -- consumers
# derive their tables from this instead of re-listing the field names.
OVERRIDABLE_FIELDS = (
    "chinchilla_multiple",
    "decay_fraction",
    "tokens_per_param",
    "lr_multiplier",
    "weight_decay",
)

# Constants from the over-training scaling fits (SemanticScholar
# CorpusID:270764838) and OLMo-core. ``_ANCHOR_SEQ_LEN`` is the 2048-token
# anchor baked into the batch-size fit; it is NOT the ladder seq_len.
_ANCHOR_SEQ_LEN = 2048
_BATCH_CONST = 160
_PARAM_ANCHOR = 108_000_000
_LR_CONST = 0.0047
# Beta2 switch point (in tokens): smaller batches want a larger beta2.
_LARGE_BATCH_TOKENS = 524_288


@dataclass(kw_only=True, frozen=True)
class WSDSChinchillaPolicy:
    """Scaling-law policy producing batch size, duration, LR, and periods.

    All fields are overridable per run (the hillclimb knobs). ``weight_decay``
    is applied to the non-embedding parameter group; embeddings stay at 0.0.
    """

    chinchilla_multiple: float = 4.0
    decay_fraction: float = 0.1
    tokens_per_param: int = 20
    lr_multiplier: float = 1.0
    weight_decay: float = 0.1
    stepped_schedule: bool = False
    seq_len: int = 4096

    def __post_init__(self) -> None:
        cm = self.chinchilla_multiple
        if cm < 0.5 or not math.log2(cm).is_integer():
            raise ValueError("chinchilla_multiple must be >= 0.5 and a power of 2.")
        if not (0.0 < self.decay_fraction < 0.5):
            raise ValueError("decay_fraction must be in (0, 0.5).")

    def target_token_batch(self, N: int) -> int:
        # Returns a TOKEN count; convert to sequences with ``/ seq_len``.
        return round(_ANCHOR_SEQ_LEN * _BATCH_CONST * (N / _PARAM_ANCHOR) ** (2 / 3))

    def train_tokens(self, N: int) -> int:
        return int(self.chinchilla_multiple * self.tokens_per_param * N)

    def warmup_tokens(self, N: int) -> int:
        return N

    def peak_lr(self, N: int) -> float:
        # The /2 is OLMo-core's empirical halving (near optimal for the stepped
        # WSD-S schedule); kept verbatim, see DESIGN.md References.
        return _LR_CONST * (N / _PARAM_ANCHOR) ** (-1 / 3) / 2.0 * self.lr_multiplier

    def beta2(self, actual_token_batch: int) -> float:
        return 0.95 if actual_token_batch >= _LARGE_BATCH_TOKENS else 0.99

    def chinchilla_periods(self) -> list[float]:
        # Decay periods as powers of two: [0.5, 1, ..., chinchilla_multiple].
        max_pow = int(math.log2(self.chinchilla_multiple))
        return [2.0**p for p in range(-1, max_pow + 1)]
