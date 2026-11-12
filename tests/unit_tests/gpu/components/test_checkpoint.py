# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from fsspec.implementations.local import LocalFileSystem

from torchtitan.components.checkpointer import CheckpointManager

_uncached_ls = LocalFileSystem.ls


def _listing_caching_ls(self, path, detail=True, **kwargs):
    # Cache listings per instance, as object-store backends (gcsfs, s3fs,
    # adlfs) do. Patched in the trainer process only; the save process lists
    # nothing.
    key = (self._strip_protocol(path), detail)
    try:
        return self.dircache[key]
    except KeyError:
        self.dircache[key] = listing = _uncached_ls(self, path, detail, **kwargs)
        return listing


class _NoState:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


@unittest.skipUnless(torch.cuda.is_available(), "pinned-memory staging needs CUDA")
class TestAsyncWithPinnedMemRetention(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        store = dist.FileStore(os.path.join(self.folder, "store"), 1)
        dist.init_process_group("nccl", store=store, rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)

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
        with mock.patch.object(LocalFileSystem, "ls", _listing_caching_ls):
            for step in range(1, 5):
                manager.save(curr_step=step)
                manager.maybe_wait_for_staging()
            manager.close()

        self.assertEqual(
            sorted(os.listdir(f"{self.folder}/ckpt")), ["step-3", "step-4"]
        )
