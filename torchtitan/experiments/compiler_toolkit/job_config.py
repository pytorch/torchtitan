# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Compile:
    """
    List of compiler pass names to apply in the compiler toolkit workflow.
    By default, no passes are applied.
    Example: --compile.passes autobucketing_reordering,regional_inductor
    """

    passes: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
