# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public bucket configuration for FlexShard."""

from __future__ import annotations

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh


__all__ = ["BucketConfig"]


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Static bucket configuration resolved after runtime meshes exist.

    ``mesh_axis`` is the storage mesh axis used for redistribution. Compute
    distributions and any single-rank assignments come from the optimizer's
    parameter metadata. Parameters requiring redistribution determine the
    concrete communication mesh. An entirely local bucket does not bind a
    communication mesh.
    """

    patterns: tuple[str, ...]
    mesh_axis: str
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))

    def _bind(self, mesh: DeviceMesh | None) -> _BucketSpec:
        return _BucketSpec(
            patterns=self.patterns,
            mesh=mesh,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    """One ordered optimizer-work bucket selected by canonical FQN.

    Patterns use case-sensitive ``fnmatch`` syntax. Every optimizer FQN must
    match exactly one bucket, and sequence order controls execution order.
    ``mesh`` is the bucket's exact one-dimensional communication mesh, or
    ``None`` when every matched parameter is already compute-ready. ``name`` is
    diagnostic metadata only.
    """

    patterns: tuple[str, ...]
    mesh: DeviceMesh | None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mesh is not None and self.mesh.ndim != 1:
            raise ValueError("bucket mesh must be one-dimensional")
        object.__setattr__(self, "patterns", tuple(self.patterns))
