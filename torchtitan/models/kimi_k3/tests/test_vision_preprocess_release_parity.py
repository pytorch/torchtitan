# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Our vision preprocessing against the RELEASE's own, module for module.

`test_vision_preprocess_pil_parity.py` judges one setting (antialias) against PIL. This
judges the pipeline against the released implementation itself: `/workspace/k3qat_mm_hf`
ships `media_utils.py` and `kimi_k3_vision_processing.py`, so the reference is on disk
rather than inferred from the report.

The decisions are integer arithmetic, so they are compared for EXACT equality -- any
difference there is a bug, not a tolerance question. That is the part worth having: a
silent disagreement in `new_width` / `pad_height` / `num_tokens` changes the patch grid,
and every downstream shape check would still pass.

Skipped when the release files are absent, since they are a downloaded artifact rather
than part of the repo.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

from torchtitan.models.kimi_k3.vision_preprocess import (
    IMAGE_MEAN,
    IMAGE_STD,
    IN_PATCH_LIMIT,
    MERGE_KERNEL_SIZE,
    navit_resize,
    PATCH_LIMIT_ON_ONE_SIDE,
    PATCH_SIZE,
)


_RELEASE = pathlib.Path("/workspace/k3qat_mm_hf")


def _release_available() -> bool:
    return (_RELEASE / "media_utils.py").is_file() and (
        _RELEASE / "preprocessor_config.json"
    ).is_file()


# Shapes chosen to hit every branch of the scale computation: no-op, pad-only, the
# in_patch_limit branch, the one-side limit branch, degenerate sizes, and primes that
# make the ceiling arithmetic visible.
_SHAPES = (
    (224, 224),
    (256, 256),
    (768, 512),
    (1024, 1024),
    (4096, 4096),
    (8192, 256),
    (28672, 224),
    (37, 53),
    (1, 1),
    (3591, 3591),
    (100, 7000),
    (13, 4099),
)


@unittest.skipUnless(_release_available(), f"release preprocessing not at {_RELEASE}")
class TestReleaseResizeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_RELEASE))
        from media_utils import navit_resize_image

        cls.release_resize = staticmethod(navit_resize_image)
        cls.cfg = json.loads(
            (_RELEASE / "preprocessor_config.json").read_text()
        )["media_proc_cfg"]

    def test_our_limits_are_the_released_ones(self):
        """A parity test driven with different constants proves nothing."""
        self.assertEqual(PATCH_SIZE, self.cfg["patch_size"])
        self.assertEqual(MERGE_KERNEL_SIZE, self.cfg["merge_kernel_size"])
        self.assertEqual(IN_PATCH_LIMIT, self.cfg["in_patch_limit"])
        self.assertEqual(
            PATCH_LIMIT_ON_ONE_SIDE, self.cfg["patch_limit_on_one_side"]
        )

    def test_resize_decisions_match_exactly(self):
        for width, height in _SHAPES:
            theirs = self.release_resize(
                width,
                height,
                PATCH_SIZE,
                MERGE_KERNEL_SIZE,
                IN_PATCH_LIMIT,
                PATCH_LIMIT_ON_ONE_SIDE,
                None,
            )
            ours = navit_resize(width, height)
            with self.subTest(size=(width, height)):
                self.assertEqual(
                    (
                        ours.new_width,
                        ours.new_height,
                        ours.pad_width,
                        ours.pad_height,
                    ),
                    (
                        theirs["new_width"],
                        theirs["new_height"],
                        theirs["pad_width"],
                        theirs["pad_height"],
                    ),
                )
                self.assertEqual(ours.num_tokens, theirs["num_tokens"])

    def test_normalisation_constants_match(self):
        """Same mean/std, different input domain, and the difference is intended.

        The release's `normalize` takes 0-255 and scales internally; ours documents
        [0, 1] input. Both land in [-1, 1], so what has to agree is the mean and std,
        not the call signature.
        """
        self.assertEqual(list(self.cfg["image_mean"]), [IMAGE_MEAN] * 3)
        self.assertEqual(list(self.cfg["image_std"]), [IMAGE_STD] * 3)

    def test_the_comparison_can_fail(self):
        """Guard the guard: perturb one limit and the decisions must diverge.

        Without this, a parity test that silently drove both sides through the same
        code path would pass forever.
        """
        theirs = self.release_resize(
            4096,
            4096,
            PATCH_SIZE,
            MERGE_KERNEL_SIZE,
            IN_PATCH_LIMIT // 4,
            PATCH_LIMIT_ON_ONE_SIDE,
            None,
        )
        ours = navit_resize(4096, 4096)
        self.assertNotEqual(
            theirs["new_width"],
            ours.new_width,
            "a quartered patch budget produced the same size, so this comparison is "
            "not actually reading the release's arithmetic",
        )


if __name__ == "__main__":
    unittest.main()
