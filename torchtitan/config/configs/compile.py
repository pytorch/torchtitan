# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compilation configuration."""

from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    enable_async_tensor_parallel: bool = False
    """Whether to pipeline tensor-parallel collectives with matrix multiplications."""

    components: list[str] = field(default_factory=lambda: ["model", "loss"])
    """Which components to compile"""

    backend: str = "inductor"

    def __post_init__(self) -> None:
        allowed = frozenset({"model", "loss"})
        unknown = [c for c in self.components if c not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown compile.components entries {unknown}; "
                f"allowed values are {sorted(allowed)}"
            )
        if self.enable_async_tensor_parallel and "model" not in self.components:
            raise ValueError("Async TP requires 'model' in --compile.components.")
