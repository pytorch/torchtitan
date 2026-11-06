# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for copy-engine FSDP communication."""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class CECollectivesConfig:
    all_gather: bool = True
    reduce_scatter: bool = True
    ranks_per_domain: int | None = None
    """Consecutive ranks per NVLink domain, in each FSDP shard group's order.

    None means the whole shard group is one domain. All-gather requires one
    domain; disable it to use hierarchical reduce-scatter across domains.
    """
    scope: Literal["all", "dense"] = "all"

    def __post_init__(self) -> None:
        if not self.all_gather and not self.reduce_scatter:
            raise ValueError("Enable at least one CE collective.")
        if self.ranks_per_domain is not None and not 2 <= self.ranks_per_domain <= 72:
            raise ValueError("ranks_per_domain must be between 2 and 72.")
        if self.scope not in ("all", "dense"):
            raise ValueError("CE collective scope must be 'all' or 'dense'.")
