# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AlphabetSort: multi-turn name-sorting env (port of prime-rl/verifiers).

Task per episode: the model is shown a list of CamelCase names (first
turn) and asked to return them sorted alphabetically by first or last
name. Turns 2+ add a small batch of new names and ask for the
cumulative sorted list, marking the new names with `// new name!`.

Used as the acceptance gate for the v6 RL refactor (multi-turn
training; rollout/train overlap; loss-mask; FIFO replay). Reproduces
the prime-rl `examples/alphabet_sort/README.md` setup with a
deterministic local name list so CPU smoke tests don't need network.
"""

from torchtitan.experiments.rl.alphabet_sort.env import (
    AlphabetSortBuilder,
    AlphabetSortDataset,
    AlphabetSortEnv,
)

__all__ = ["AlphabetSortBuilder", "AlphabetSortDataset", "AlphabetSortEnv"]
