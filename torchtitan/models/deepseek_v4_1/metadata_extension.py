# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Identity metadata extension for DeepSeek V4.1.

The default post-process of the built attention masks is the identity.
Backend-specific extensions can be swapped via the override mechanism.
"""

from dataclasses import dataclass, field, fields
from typing import Any, cast

from torch.utils._pytree import GetAttrKey, KeyEntry, register_pytree_node
from torchtitan.config.configurable import Configurable


@dataclass(kw_only=True, slots=True)
class LightningIndexerKernelConfig:
    layout_q: str = "TND"
    layout_k: str = "TND"
    mask_mode: int = 3
    cmp_ratio: int = 4


class MetadataExtension(Configurable):
    """The default identity post-process of the built attention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 0
        num_heads: int | None = None
        head_dim: int | None = None
        index_n_heads: int | None = None
        index_head_dim: int | None = None
        index_topk: int | None = None
        compress_ratios: tuple[int, ...] = ()
        li_kernel_config: LightningIndexerKernelConfig = field(
            default_factory=LightningIndexerKernelConfig,
        )

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadata):
        return metadata


def register_pytree_node_for_dataclass(cls: type) -> None:
    from torch.utils._pytree import SUPPORTED_NODES

    if cls in SUPPORTED_NODES:
        return
    field_names = [f.name for f in fields(cls)]

    def flatten(obj):
        return [getattr(obj, name) for name in field_names], None

    def flatten_with_keys(obj):
        keys = cast(
            "list[tuple[KeyEntry, Any]]",
            [(GetAttrKey(name), getattr(obj, name)) for name in field_names],
        )
        return keys, None

    def unflatten(values, context):
        return cls(**dict(zip(field_names, values, strict=True)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
    )
