# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Compile:
    """
    Compiler configuration for the compiler toolkit workflow.

    - joint_passes: List of joint graph pass names to apply on the joint forward-backward
      graph before partitioning.

      Example: --compile.joint_passes inductor_decomposition

    - passes: List of compiler pass names to apply to the partitioned forward/backward graphs.

      Example: --compile.passes full_inductor_compilation

      Note: If "full_inductor_compilation" is specified, "inductor_decomposition" must
      be included in joint_passes.
    """

    joint_passes: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
