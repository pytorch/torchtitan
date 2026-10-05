# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
import unittest
from unittest import mock

import torch.distributed as dist

from torchtitan.experiments.ft_fsdp import snapshot


class _FakeTransport:
    _ids = itertools.count()

    def __init__(self) -> None:
        self.id = next(self._ids)
        self.peer: bytes | None = None
        self.closed = False

    def bind(self, *, timeout: float) -> bytes:
        return f"agent{self.id}".encode()

    def connect(self, peer_url: bytes, *, timeout: float) -> int:
        self.peer = peer_url
        return 0

    def close(self, *, timeout: float) -> None:
        self.closed = True


class TransportPoolTest(unittest.TestCase):
    def test_get_refills(self) -> None:
        created = []

        def new():
            created.append(_FakeTransport())
            return created[-1]

        with mock.patch.object(snapshot, "_new_nixl_transport", new):
            pool = snapshot.TransportPool(2)
            first = pool.get()
            second = pool.get()
            pool._ready.get().result()
        self.assertIsNot(first, second)
        self.assertEqual(created[:2], [first, second])
        # Each get queues a replacement.
        self.assertEqual(len(created), 4)

    def test_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            snapshot.TransportPool(0)

    def test_creation_error_raised_on_get(self) -> None:
        def new():
            raise RuntimeError("no NICs")

        with mock.patch.object(snapshot, "_new_nixl_transport", new):
            pool = snapshot.TransportPool(1)
            with self.assertRaisesRegex(RuntimeError, "no NICs"):
                pool.get()


class BootstrapTest(unittest.TestCase):
    def test_exchanges_metadata(self) -> None:
        store = dist.HashStore()
        a, b = _FakeTransport(), _FakeTransport()
        t = threading.Thread(
            target=snapshot._bootstrap,
            args=(b, store),
            kwargs={"rank": 1, "peer_rank": 0, "timeout": 5.0},
        )
        t.start()
        snapshot._bootstrap(a, store, rank=0, peer_rank=1, timeout=5.0)
        t.join()
        self.assertEqual(a.peer, b.bind(timeout=1.0))
        self.assertEqual(b.peer, a.bind(timeout=1.0))

    def test_closes_on_timeout(self) -> None:
        tr = _FakeTransport()
        with self.assertRaises(dist.DistStoreError):
            snapshot._bootstrap(tr, dist.HashStore(), rank=0, peer_rank=1, timeout=0.1)
        self.assertTrue(tr.closed)


if __name__ == "__main__":
    unittest.main()
