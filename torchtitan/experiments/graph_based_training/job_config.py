# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Compile:
    """
    Compiler configuration for graph-based training.

    - mode: Compilation mode. "jit" uses torch.compile with a custom backend.
      "aot" uses AOT joint graph capture with a configurable pass pipeline.
      None disables compilation entirely.

    - passes: List of compiler pass names to apply. Passes are automatically
      classified as pre-partition or post-partition. Some passes are only
      supported in certain modes; an error is raised for unsupported combinations.

      Available passes:
        - auto_bucketing: Automatic comm/compute overlap bucketing (jit, aot)
        - transformer_block_bucketing: Manual per-block bucketing (jit, aot)
        - regional_inductor: Regional Inductor compilation (aot only)
        - cudagraph: CUDA graph capture and replay (aot only)
        - full_inductor_compilation: Full Inductor code generation (aot only)
        - inductor_decomposition: Inductor decompositions on joint graph (aot only)

      Example: --compile.passes auto_bucketing,cudagraph
    """

    mode: Literal["jit", "aot"] | None = None
    passes: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
