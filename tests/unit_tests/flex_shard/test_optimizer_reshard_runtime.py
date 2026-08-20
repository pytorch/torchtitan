# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _foreach_copy_or_fallback_,
)


class TestForeachCopyOrFallback(unittest.TestCase):
    def test_rejects_mismatched_lists(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            _foreach_copy_or_fallback_((torch.empty(1),), ())

    def test_empty_and_single_copy_bypass_foreach(self):
        _foreach_copy_or_fallback_((), ())

        source = torch.arange(6).reshape(2, 3)
        destination = torch.empty_like(source)
        with patch.object(torch, "_foreach_copy_") as foreach_copy:
            _foreach_copy_or_fallback_((destination,), (source,))
        foreach_copy.assert_not_called()
        torch.testing.assert_close(destination, source)

    def test_foreach_copy_supports_mixed_sizes(self):
        sources = (
            torch.arange(6).reshape(2, 3),
            torch.arange(8).reshape(2, 4),
        )
        destinations = tuple(torch.empty_like(source) for source in sources)
        original_foreach_copy = torch._foreach_copy_
        with patch.object(
            torch,
            "_foreach_copy_",
            wraps=original_foreach_copy,
        ) as foreach_copy:
            _foreach_copy_or_fallback_(destinations, sources)
        foreach_copy.assert_called_once()
        for destination, source in zip(destinations, sources, strict=True):
            torch.testing.assert_close(destination, source)

    def test_foreach_copy_supports_noncontiguous_views(self):
        source_base = torch.arange(24).reshape(4, 6)
        destination_base = torch.zeros_like(source_base)
        sources = (source_base[:, ::2], source_base[:, 1::2])
        destinations = (destination_base[:, ::2], destination_base[:, 1::2])

        _foreach_copy_or_fallback_(destinations, sources)

        torch.testing.assert_close(destination_base, source_base)

    def test_incompatible_dtype_uses_copy_fallback(self):
        sources = (torch.arange(3), torch.arange(4))
        destinations = (
            torch.empty(3, dtype=torch.float32),
            torch.empty(4, dtype=torch.float32),
        )
        with patch.object(torch, "_foreach_copy_") as foreach_copy:
            _foreach_copy_or_fallback_(destinations, sources)
        foreach_copy.assert_not_called()
        for destination, source in zip(destinations, sources, strict=True):
            torch.testing.assert_close(destination, source.to(destination.dtype))


if __name__ == "__main__":
    unittest.main()
