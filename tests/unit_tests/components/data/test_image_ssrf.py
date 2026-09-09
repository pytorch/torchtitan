# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ipaddress
from unittest import mock

import pytest
import requests

from torchtitan.hf_datasets.multimodal.utils.image import _is_blocked_ip, _is_safe_url


class TestIsBlockedIP:
    """Tests for _is_blocked_ip with IPv4 and IPv6."""

    def test_private_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("10.0.0.1"))
        assert _is_blocked_ip(ipaddress.ip_address("172.16.0.1"))
        assert _is_blocked_ip(ipaddress.ip_address("192.168.1.1"))

    def test_loopback_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))

    def test_link_local_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("169.254.1.1"))
        assert _is_blocked_ip(ipaddress.ip_address("169.254.169.254"))  # metadata

    def test_multicast_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("224.0.0.1"))

    def test_unspecified_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("0.0.0.0"))

    def test_reserved_ipv4(self):
        assert _is_blocked_ip(ipaddress.ip_address("240.0.0.1"))

    def test_loopback_ipv6(self):
        assert _is_blocked_ip(ipaddress.ip_address("::1"))

    def test_link_local_ipv6(self):
        assert _is_blocked_ip(ipaddress.ip_address("fe80::1"))

    def test_unique_local_ipv6(self):
        assert _is_blocked_ip(ipaddress.ip_address("fd00::1"))

    def test_unicast_ipv6_not_blocked(self):
        assert not _is_blocked_ip(ipaddress.ip_address("2001:4860:4860::8888"))

    def test_ipv4_mapped_ipv6(self):
        # ::ffff:127.0.0.1 should be blocked as loopback
        assert _is_blocked_ip(ipaddress.ip_address("::ffff:127.0.0.1"))

    def test_public_ipv4_not_blocked(self):
        assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))

    def test_public_ipv6_not_blocked(self):
        assert not _is_blocked_ip(ipaddress.ip_address("2606:4700:4700::1111"))


class TestIsSafeURL:
    """Tests for _is_safe_url with mocked DNS resolution."""

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    def test_https_url_safe(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket_AF_INET := __import__("socket").AF_INET, None, None, None,
             ("8.8.8.8", 0)),
        ]
        assert _is_safe_url("https://example.com/image.png")

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    def test_http_url_safe(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (__import__("socket").AF_INET, None, None, None, ("8.8.4.4", 0)),
        ]
        assert _is_safe_url("http://example.com/page")

    def test_localhost_blocked(self):
        assert not _is_safe_url("http://localhost/image.png")

    def test_metadata_url_blocked(self):
        assert not _is_safe_url("http://169.254.169.254/latest/meta-data/")

    def test_private_ip_blocked(self):
        assert not _is_safe_url("http://192.168.1.1/image.png")

    def test_file_scheme_blocked(self):
        assert not _is_safe_url("file:///etc/passwd")

    def test_ftp_scheme_blocked(self):
        assert not _is_safe_url("ftp://example.com/file")

    def test_empty_hostname_blocked(self):
        assert not _is_safe_url("http:///path")

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    def test_dns_resolution_failure_blocked(self, mock_getaddrinfo):
        import socket as sock
        mock_getaddrinfo.side_effect = sock.gaierror("DNS resolution failed")
        assert not _is_safe_url("http://nonexistent.invalid/image.png")

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    def test_ipv6_resolved_safe(self, mock_getaddrinfo):
        import socket as sock
        mock_getaddrinfo.return_value = [
            (sock.AF_INET6, None, None, None,
             ("2001:4860:4860::8888", 0, 0, 0)),
        ]
        assert _is_safe_url("https://example.com/image.png")


class TestFetchURLSafe:
    """Tests for _fetch_url_safe redirect handling."""

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._is_safe_url")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.requests.Session")
    def test_redirect_chain_validated(
        self, mock_session_cls, mock_getaddrinfo, mock_is_safe
    ):
        import socket as sock
        mock_getaddrinfo.return_value = [
            (sock.AF_INET, None, None, None, ("8.8.8.8", 0)),
        ]
        mock_is_safe.return_value = True

        session = mock_session_cls.return_value
        # Simulate redirect chain: 302 -> 302 -> 200
        redirect_resp = mock.MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.headers = {"Location": "https://cdn.example.com/img.png"}
        redirect_resp.url = "https://example.com/img.png"

        final_resp = mock.MagicMock()
        final_resp.is_redirect = False
        final_resp.content = b"image-bytes"

        session.get.side_effect = [redirect_resp, final_resp]

        from torchtitan.hf_datasets.multimodal.utils.image import _fetch_url_safe
        content = _fetch_url_safe("https://example.com/img.png")
        assert content == b"image-bytes"
        assert session.get.call_count == 2

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._is_safe_url")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.requests.Session")
    def test_unsafe_redirect_blocked(
        self, mock_session_cls, mock_getaddrinfo, mock_is_safe
    ):
        import socket as sock
        mock_getaddrinfo.return_value = [
            (sock.AF_INET, None, None, None, ("8.8.8.8", 0)),
        ]
        # Allow initial URL, block redirect
        mock_is_safe.side_effect = [True, False]

        session = mock_session_cls.return_value
        redirect_resp = mock.MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.headers = {"Location": "http://169.254.169.254/steal"}
        redirect_resp.url = "https://safe.example.com/img.png"

        session.get.return_value = redirect_resp

        from torchtitan.hf_datasets.multimodal.utils.image import _fetch_url_safe
        with pytest.raises(ValueError, match="Blocked redirect"):
            _fetch_url_safe("https://safe.example.com/img.png")

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._is_safe_url")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.socket.getaddrinfo")
    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image.requests.Session")
    def test_redirect_loop_bounded(
        self, mock_session_cls, mock_getaddrinfo, mock_is_safe
    ):
        """Verify infinite redirect loops are bounded (max_redirects = 10)."""
        import socket as sock
        mock_getaddrinfo.return_value = [
            (sock.AF_INET, None, None, None, ("8.8.8.8", 0)),
        ]
        mock_is_safe.return_value = True

        session = mock_session_cls.return_value
        redirect_resp = mock.MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.headers = {"Location": "https://safe-redirect.example.com/page"}
        redirect_resp.url = "https://example.com/img.png"
        redirect_resp.content = b"redirect-bytes"

        # Every call returns a redirect response → would loop forever without cap
        session.get.return_value = redirect_resp

        from torchtitan.hf_datasets.multimodal.utils.image import _fetch_url_safe
        _fetch_url_safe("https://example.com/img.png")
        # First call is the initial request, then up to 10 redirect hops
        assert session.get.call_count == 11
