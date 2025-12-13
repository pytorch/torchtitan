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
    Compiler configuration for the compiler toolkit workflow.

    - backend: The compilation backend to use. Options are:
        - "aot_eager": AOT Autograd with eager backend (graph transformations only)
        - "inductor": Full Inductor compilation with optimized code generation

    - passes: List of compiler pass names to apply in the compiler toolkit workflow.

      Example: --compile.passes autobucketing_reordering
    """

    backend: Literal["aot_eager", "inductor"] = "aot_eager"
    passes: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
