# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import tempfile
import unittest

from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter


class TestDiskStorageAdapter(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            data = pickle.dumps({"hello": "world", "values": [1, 2, 3]})

            path = storage.save("test_key", data)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(storage.exists("test_key"))

            loaded = storage.load("test_key")
            self.assertEqual(data, loaded)

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            with self.assertRaises(FileNotFoundError):
                storage.load("nonexistent")

    def test_exists_false_for_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            self.assertFalse(storage.exists("missing"))

    def test_save_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            storage = DiskStorageAdapter(nested)
            data = b"test"
            path = storage.save("key", data)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(storage.load("key"), data)

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("key", b"data")
            self.assertTrue(storage.exists("key"))
            storage.delete("key")
            self.assertFalse(storage.exists("key"))

    def test_delete_nonexistent_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.delete("nonexistent")

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("key", b"original")
            storage.save("key", b"updated")
            self.assertEqual(storage.load("key"), b"updated")

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            with self.assertRaises(ValueError):
                storage.save("../../escape", b"data")


if __name__ == "__main__":
    unittest.main()
