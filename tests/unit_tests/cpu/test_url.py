# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for SSRF-safe URL fetching in multimodal image loading.

The image decoder accepts HTTP(S) URLs from untrusted dataset samples, so
``_url.is_safe_url`` must reject any URL that resolves to a private,
loopback, link-local, multicast, reserved, or unspecified address. These
tests mock DNS and the HTTP call so nothing touches the network.
"""

import socket
import unittest
from unittest.mock import MagicMock, patch

from torchtitan.hf_datasets.multimodal.utils import _url


def _mock_getaddrinfo(*addresses):
    """Return a ``getaddrinfo`` replacement yielding the given IP strings.

    Each entry becomes one ``getaddrinfo`` result tuple; the fifth element's
    first field is the resolved IP.
    """
    infos = [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0)) for ip in addresses
    ]

    def fake_getaddrinfo(hostname, *args, **kwargs):
        if hostname is None:
            return []
        return infos

    return fake_getaddrinfo


class TestIsSafeUrl(unittest.TestCase):
    def test_public_ip_allowed(self):
        with patch(
            "socket.getaddrinfo", side_effect=_mock_getaddrinfo("1.2.3.4")
        ):
            self.assertTrue(_url.is_safe_url("https://example.com/a.png"))

    def test_loopback_blocked(self):
        with patch(
            "socket.getaddrinfo", side_effect=_mock_getaddrinfo("127.0.0.1")
        ):
            self.assertFalse(_url.is_safe_url("http://127.0.0.1/x.png"))

    def test_link_local_blocked(self):
        with patch(
            "socket.getaddrinfo",
            side_effect=_mock_getaddrinfo("169.254.169.254"),
        ):
            self.assertFalse(
                _url.is_safe_url("http://169.254.169.254/latest/meta-data/")
            )


if __name__ == "__main__":
    unittest.main()