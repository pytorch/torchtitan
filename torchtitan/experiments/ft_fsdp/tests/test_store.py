# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
from datetime import timedelta

import torch.distributed as dist

from torchtitan.experiments.ft_fsdp.store import TimedStore


class _HangingStore(dist.Store):
    """Delegates to ``inner`` but blocks until ``release`` once ``hang`` is
    set."""

    def __init__(
        self, inner: dist.Store, hang: threading.Event, release: threading.Event
    ) -> None:
        super().__init__()
        self.inner = inner
        self.hang = hang
        self.release = release

    def _maybe_hang(self) -> None:
        if self.hang.is_set():
            self.release.wait()

    def set(self, key, value):
        self._maybe_hang()
        self.inner.set(key, value)

    def get(self, key):
        self._maybe_hang()
        return self.inner.get(key)

    def add(self, key, amount):
        self._maybe_hang()
        return self.inner.add(key, amount)

    def check(self, keys):
        self._maybe_hang()
        return self.inner.check(keys)


class TimedStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inner = dist.HashStore()
        self.hangs: list[threading.Event] = []
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def _connect(self) -> dist.Store:
        hang = threading.Event()
        self.hangs.append(hang)
        return _HangingStore(self.inner, hang, self.release)

    def _store(self) -> TimedStore:
        return TimedStore(
            self._connect,
            op_timeout=timedelta(seconds=0.2),
            timeout=timedelta(seconds=1),
        )

    def test_ops(self) -> None:
        store = self._store()
        store.set("a", "1")
        self.assertEqual(store.get("a"), b"1")
        self.assertTrue(store.check(["a"]))
        self.assertEqual(store.add("n", 2), 2)
        self.assertEqual(store.multi_get(["a"]), [b"1"])

    def test_reconnects_on_hang(self) -> None:
        store = self._store()
        store.set("a", "1")
        self.hangs[-1].set()
        self.assertTrue(store.check(["a"]))
        self.assertEqual(len(self.hangs), 2)

    def test_add_does_not_retry(self) -> None:
        store = self._store()
        self.hangs[-1].set()
        with self.assertRaises(TimeoutError):
            store.add("n", 1)
        self.assertEqual(store.add("n", 1), 1)

    def test_wait_times_out(self) -> None:
        store = self._store()
        with self.assertRaises(dist.DistStoreError):
            store.wait(["missing"], timedelta(seconds=0.3))

    def test_prefix_store(self) -> None:
        # Keep a reference: PrefixStore does not keep a Python store alive.
        client = self._store().clone()
        store = dist.PrefixStore("p", client)
        threading.Timer(0.2, lambda: store.set("k", "v")).start()
        store.wait(["k"], timedelta(seconds=2))
        self.assertEqual(store.get("k"), b"v")
        self.assertTrue(self.inner.check(["p/k"]))


if __name__ == "__main__":
    unittest.main()
