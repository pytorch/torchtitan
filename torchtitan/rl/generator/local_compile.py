# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared ``local_compile`` regions for generator-only functions."""

from collections.abc import Callable
from typing import Any

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.local_compile import local_compile

INFERENCE_BATCH_INVARIANT_REGION = "inference_batch_invariant"
INFERENCE_BATCH_VARIANT_REGION = "inference_batch_variant"

_registered_regions: set[str] = set()


def inference_local_compile(
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a generator-only function under the shared region that matches
    ``batch_invariant``: ``INFERENCE_BATCH_INVARIANT_REGION`` or
    ``INFERENCE_BATCH_VARIANT_REGION``.
    """
    name = (
        INFERENCE_BATCH_INVARIANT_REGION
        if batch_invariant
        else INFERENCE_BATCH_VARIANT_REGION
    )
    _registered_regions.add(name)
    return local_compile(name, batch_invariant=batch_invariant, **compile_kwargs)


def inference_regions() -> list[str]:
    """Return the shared inference regions that have registered functions, leaving
    out the batch-variant one in batch-invariant mode."""
    return [
        name
        for name in (INFERENCE_BATCH_INVARIANT_REGION, INFERENCE_BATCH_VARIANT_REGION)
        if name in _registered_regions
        and not (
            name == INFERENCE_BATCH_VARIANT_REGION and is_in_batch_invariant_mode()
        )
    ]
