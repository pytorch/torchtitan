# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video preprocessing for one model.

A video is a stack of frames the vision encoder patches like images (with
``temporal_patch_size`` frames per patch), so ``VideoProcessor`` extends
``ImageProcessor`` with the frame-specific first stage and keeps its own config:
a model can give videos a different resolution budget than images. The trainer
builds one from ``Trainer.Config.video_processor`` and hands it to the
dataloader.
"""

from dataclasses import dataclass

import torch

from .image_processor import ImageProcessor


class VideoProcessor(ImageProcessor):
    """Turns video frames into a model's vision-encoder inputs."""

    # Its own Config, so that ``VideoProcessor.Config(...).build()`` builds a
    # ``VideoProcessor`` rather than an ``ImageProcessor``.
    @dataclass(kw_only=True, slots=True)
    class Config(ImageProcessor.Config):
        pass

    def resize_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize decoded uint8 ``(T, H, W, C)`` frames: uint8 ``(T, H', W', C)``.

        Runs per video on the CPU; ``normalize_and_patchify`` normalizes and
        patches the result.
        """
        frames_TCHW = self._resize(frames.permute(0, 3, 1, 2))
        return frames_TCHW.permute(0, 2, 3, 1)
