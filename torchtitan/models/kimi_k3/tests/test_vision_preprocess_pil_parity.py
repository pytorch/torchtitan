# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Our downscale against the released preprocessing's, which is PIL bicubic.

Finding 62 was recorded as "fixed, weakly judged": `prepare_image` was missing
`antialias=True`, and the note said antialiasing "cannot be detected by any test this
repo can write on its own -- it needs an external reference". The reference is PIL, which
is what the released `media_utils.image_to_np` calls
(`image.resize(..., resample=Image.Resampling.BICUBIC)`), and PIL's bicubic always
prefilters on downscale. So the judge does exist. This is it.

It cannot be an equality test. `F.interpolate(mode="bicubic", antialias=True)` is
documented to closely match PIL rather than to reproduce it bit-for-bit, so what is
pinned is a DIFFERENTIAL: with antialiasing on, our result is orders of magnitude closer
to PIL than with it off. Measured 256 -> 64, mean absolute error against PIL:

    2px checkerboard   antialias=True 0.4994   antialias=False 120.0293   240x
    uniform noise      antialias=True 0.2661   antialias=False  39.2317   147x
    smooth gradient    antialias=True 0.4945   antialias=False   0.5000   1.0x

The third row is why this stayed unjudged for so long: on smooth content the two are
indistinguishable, so any test built from a gradient or a flat fill passes either way and
proves nothing. The content has to carry energy above the target Nyquist rate.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from torchtitan.models.kimi_k3.vision_preprocess import navit_resize, prepare_image


try:
    from PIL import Image

    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _HAVE_PIL = False


def _checkerboard(size: int, cell: int = 2) -> np.ndarray:
    return ((np.indices((size, size)).sum(0) // cell) % 2 * 255).astype(np.uint8)


def _checkerboard_wide(height: int, width: int, cell: int = 2) -> np.ndarray:
    return ((np.indices((height, width)).sum(0) // cell) % 2 * 255).astype(np.uint8)


def _noise(size: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (size, size), dtype=np.uint8)


def _gradient(size: int) -> np.ndarray:
    return np.linspace(0, 255, size)[None, :].repeat(size, 0).astype(np.uint8)


def _pil_reference(rgb_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    """What the release produces: PIL bicubic, in [0, 1], as [C, H, W]."""
    resized = Image.fromarray(rgb_hwc).resize(
        (width, height), resample=Image.Resampling.BICUBIC
    )
    return np.asarray(resized, dtype=np.float64).transpose(2, 0, 1) / 255.0


def _ours(rgb_hwc: np.ndarray, height: int, width: int, *, antialias: bool):
    """Our interpolate call, with antialiasing as the single variable."""
    chw = torch.from_numpy(rgb_hwc.transpose(2, 0, 1)).double() / 255.0
    out = F.interpolate(
        chw.unsqueeze(0),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=antialias,
    ).clamp(0.0, 1.0)
    return out[0].numpy()


@unittest.skipUnless(_HAVE_PIL, "PIL is the external reference this test needs")
class TestPILParity(unittest.TestCase):
    def test_antialiasing_is_what_closes_the_gap_to_pil(self):
        for name, plane in (
            ("checkerboard", _checkerboard(256)),
            ("noise", _noise(256)),
        ):
            rgb = np.stack([plane] * 3, -1)
            ref = _pil_reference(rgb, 64, 64)
            on = np.abs(_ours(rgb, 64, 64, antialias=True) - ref).mean()
            off = np.abs(_ours(rgb, 64, 64, antialias=False) - ref).mean()
            with self.subTest(image=name):
                # Generous bounds: the point is the order of magnitude, not the digits.
                self.assertLess(on, 0.01, f"{name}: antialiased result is far from PIL")
                self.assertGreater(
                    off / max(on, 1e-12),
                    20.0,
                    f"{name}: dropping antialiasing barely changed the distance to PIL, "
                    "so this image cannot judge the setting",
                )

    def test_a_smooth_image_cannot_judge_it(self):
        """Pinned so nobody 'simplifies' the fixtures into something that proves nothing."""
        rgb = np.stack([_gradient(256)] * 3, -1)
        ref = _pil_reference(rgb, 64, 64)
        on = np.abs(_ours(rgb, 64, 64, antialias=True) - ref).mean()
        off = np.abs(_ours(rgb, 64, 64, antialias=False) - ref).mean()
        self.assertLess(
            off / max(on, 1e-12), 1.5, "a smooth gradient unexpectedly discriminates"
        )

    def test_prepare_image_downscales_through_the_antialiased_path(self):
        """The real entry point, not just the interpolate call the other tests isolate.

        Reconstructs the resized image from the patches prepare_image returns and compares
        it to PIL at the size prepare_image itself chose. Padding rows are excluded: they
        are zeros pre-normalization by design and have no counterpart in the reference.
        """
        # A WIDE image at the production patch size, and one that DECIMATES: 28672x224
        # hits the one-side patch limit (512 * 14 = 7168) for a 4x reduction, with no
        # padding to exclude. Both parts matter. The patch_size override route looked
        # cheaper and does not work -- ResizePlan carries no patch_size, so its
        # patch_grid property divides by the module constant and disagrees with the plan
        # it belongs to, and prepare_image dies in the view (recorded separately). And a
        # mild reduction judges nothing: at 8192x256 (0.875x) this same comparison gives
        # only 1.9x, because antialiasing hardly matters when you are barely decimating.
        height, width = 224, 28672
        rgb = np.stack([_checkerboard_wide(height, width)] * 3, -1)
        chw = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        patches, grid = prepare_image(chw, already_normalized=True)
        plan = navit_resize(width, height)
        self.assertLess(
            plan.new_width / width,
            0.5,
            "fixture stopped decimating, so this test would no longer discriminate",
        )
        self.assertEqual(
            (plan.pad_height, plan.pad_width),
            (0, 0),
            "fixture now needs padding, which has no counterpart in the reference",
        )

        _, h, w = grid
        patch = patches.shape[-1]
        canvas = (
            patches.reshape(h, w, 3, patch, patch)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, h * patch, w * patch)
            .double()
            .numpy()
        )
        got = canvas[:, : plan.new_height, : plan.new_width]
        ref = _pil_reference(rgb, plan.new_height, plan.new_width)
        # The same differential the isolated tests use, on the real entry point: what
        # prepare_image produces against PIL, versus the same resize with antialiasing
        # off. Measured 0.00196 vs 0.47070, a factor of 240.
        alt = _ours(rgb, plan.new_height, plan.new_width, antialias=False)
        d_ours = np.abs(got - ref).mean()
        d_alt = np.abs(alt - ref).mean()
        self.assertLess(d_ours, 0.01, "prepare_image is not tracking PIL bicubic")
        self.assertGreater(
            d_alt / max(d_ours, 1e-12),
            20.0,
            "prepare_image is no closer to PIL than an un-antialiased resize would be",
        )


if __name__ == "__main__":
    unittest.main()
