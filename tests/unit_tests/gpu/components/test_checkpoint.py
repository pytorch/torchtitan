# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import fsspec
import torch
import torch.distributed as dist
import torch.nn as nn
from fsspec.implementations.local import LocalFileSystem

from torchtitan.components.checkpointer import CheckpointManager


class _ListingCachingLocalFileSystem(LocalFileSystem):
    """``file://`` that caches directory listings, as gcsfs, s3fs and adlfs do."""

    def ls(self, path, detail=True, **kwargs):
        listing = self.dircache.get((path, detail))
        if listing is None:
            listing = super().ls(path, detail, **kwargs)
            self.dircache[(path, detail)] = listing
        return listing


class _NoState:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


@unittest.skipUnless(torch.cuda.is_available(), "pinned-memory staging needs CUDA")
class TestAsyncWithPinnedMemRetention(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        store = dist.FileStore(os.path.join(self.folder, "store"), 1)
        dist.init_process_group("nccl", store=store, rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)
        # Only this process lists the folder through the caching class. DCP's
        # save process is spawned with a fresh fsspec registry, so it writes
        # through the plain LocalFileSystem.
        fsspec.register_implementation(
            "file", _ListingCachingLocalFileSystem, clobber=True
        )
        self.addCleanup(
            fsspec.register_implementation, "file", LocalFileSystem, clobber=True
        )

    def test_keep_latest_k_purges_checkpoints_saved_by_the_save_process(self):
        # async_with_pinned_mem saves each checkpoint from a subprocess while
        # the trainer process purges. On a remote folder read through fsspec's
        # listing cache, the purge never saw the new checkpoints and
        # keep_latest_k never deleted one.
        manager = CheckpointManager(
            CheckpointManager.Config(
                folder=f"file://{self.folder}/ckpt",  # a remote URI to torchtitan
                interval=1,
                keep_latest_k=2,
                last_save_model_only=False,
                async_mode="async_with_pinned_mem",
            ),
            model_parts=[nn.Linear(4, 4, device="cuda")],
            optimizers=_NoState(),
            lr_schedulers=_NoState(),
            ema=None,
            extra_states={},
            sd_adapter=None,
        )
        for step in range(1, 5):
            manager.save(curr_step=step)
            manager.maybe_wait_for_staging()
        manager.close()

        self.assertEqual(
            sorted(os.listdir(f"{self.folder}/ckpt")), ["step-3", "step-4"]
        )
