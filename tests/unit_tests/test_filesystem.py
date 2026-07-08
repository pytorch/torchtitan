# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
import uuid

import fsspec

from torchtitan.tools import filesystem


class TestIsRemote(unittest.TestCase):
    def test_remote_uris(self):
        self.assertTrue(filesystem.is_remote("gs://bucket/run"))
        self.assertTrue(filesystem.is_remote("s3://bucket/run"))

    def test_local_paths(self):
        self.assertFalse(filesystem.is_remote("/abs/path"))
        self.assertFalse(filesystem.is_remote("./relative"))
        self.assertFalse(filesystem.is_remote("plain"))


class TestJoin(unittest.TestCase):
    def test_local_folder_is_prefixed(self):
        self.assertEqual(filesystem.join("/dump", "checkpoint"), "/dump/checkpoint")

    def test_remote_folder_is_not_prefixed(self):
        self.assertEqual(
            filesystem.join("/dump", "gs://bucket/run/ckpt"),
            "gs://bucket/run/ckpt",
        )


class TestLocalOps(unittest.TestCase):
    def test_local_filesystem_operations(self):
        with tempfile.TemporaryDirectory() as d:
            step = os.path.join(d, "step-10")
            os.makedirs(step)
            meta = os.path.join(step, ".metadata")
            with open(meta, "w") as f:
                f.write("x")

            self.assertTrue(filesystem.exists(step))
            self.assertTrue(filesystem.isdir(step))
            self.assertTrue(filesystem.isfile(meta))
            self.assertFalse(filesystem.isfile(step))
            self.assertEqual(filesystem.listdir(d), ["step-10"])

            filesystem.rmtree(step)
            self.assertFalse(filesystem.exists(step))

            # rmtree on a missing path is a no-op (ignore_errors=True).
            filesystem.rmtree(step)


class TestRemoteOps(unittest.TestCase):
    """Exercise the fsspec path with the in-memory backend (no gcsfs/network)."""

    def setUp(self):
        # Unique root so tests do not interfere via the process-shared memory fs.
        self.name = f"test-{uuid.uuid4().hex}"
        self.root = f"memory://{self.name}"
        self._fs = fsspec.filesystem("memory")

    def tearDown(self):
        if self._fs.exists(f"/{self.name}"):
            self._fs.rm(f"/{self.name}", recursive=True)

    def _write(self, url):
        with fsspec.open(url, "wb") as f:
            f.write(b"x")

    def test_exists_isdir_isfile(self):
        self._write(f"{self.root}/step-10/.metadata")
        self.assertTrue(filesystem.exists(f"{self.root}/step-10"))
        self.assertTrue(filesystem.isdir(f"{self.root}/step-10"))
        self.assertTrue(filesystem.isfile(f"{self.root}/step-10/.metadata"))
        self.assertFalse(filesystem.exists(f"{self.root}/step-99"))

    def test_listdir_returns_basenames(self):
        self._write(f"{self.root}/step-10/.metadata")
        self._write(f"{self.root}/step-20/.metadata")
        self.assertEqual(sorted(filesystem.listdir(self.root)), ["step-10", "step-20"])

    def test_rmtree(self):
        self._write(f"{self.root}/step-10/.metadata")
        filesystem.rmtree(self.root)
        self.assertFalse(filesystem.exists(self.root))

    def test_rmtree_missing_is_noop(self):
        # A missing remote path must not raise (parity with local rmtree),
        # otherwise a failed purge would kill the background purge thread.
        filesystem.rmtree(f"{self.root}/does-not-exist")
