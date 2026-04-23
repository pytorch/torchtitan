# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class AutoParallelCompileConfig(CompileConfig):
    comms_bucket_reorder_strategy: str = "aten"
    """Options: "aten" (default), "inductor", "none" """

    autop_force_bf16: bool = False


@dataclass(kw_only=True, slots=True)
class AutoParallelConfig(Trainer.Config):
    compile: AutoParallelCompileConfig = field(
        default_factory=AutoParallelCompileConfig
    )


def to_autoparallel_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
    flavor: str | None = None,
) -> AutoParallelConfig:
    """Convert a base Trainer.Config to an AutoParallelConfig.

    Copies all fields from the base config and replaces the model_spec with one
    from the autoparallel model_registry. The compile field is removed and
    left as the AutoParallelConfig default; callers should explicitly set it.

    Args:
        base_config: The base Trainer.Config to convert.
        model_registry: A callable that returns a ModelSpec for a given flavor.
        flavor: Optional flavor override. If None, uses the base config's flavor.
    """
    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["model_spec"] = model_registry(flavor or base_config.model_spec.flavor)
    d.pop("compile")

    ac = d.get("activation_checkpoint")
    if ac is not None and ac.mode != "none":
        d["activation_checkpoint"] = ActivationCheckpointConfig(mode="selective")

    return AutoParallelConfig(**d)
