# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Compile:
    model_backend_override: str | None = None
    """Override backend to compile in simplefsdp. Additional backend includes aot_eager_autobucketing"""


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
