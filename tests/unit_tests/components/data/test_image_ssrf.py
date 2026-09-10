# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from requests_hardened.ip_filter import InvalidIPAddress

from torchtitan.hf_datasets.multimodal.utils.image import _fetch_url_safe, _http_session


class TestFetchURLSafe:
    """Tests for _fetch_url_safe using the requests-hardened session."""

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._http_session")
    def test_url_fetch_returns_content(self, mock_session):
        mock_response = mock.MagicMock()
        mock_response.content = b"image-bytes"
        mock_session.get.return_value = mock_response

        content = _fetch_url_safe("https://example.com/img.png")
        assert content == b"image-bytes"
        mock_session.get.assert_called_once_with(
            "https://example.com/img.png", timeout=10.0
        )

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._http_session")
    def test_custom_timeout_passed(self, mock_session):
        mock_response = mock.MagicMock()
        mock_response.content = b"image-bytes"
        mock_session.get.return_value = mock_response

        _fetch_url_safe("https://example.com/img.png", timeout=30.0)
        mock_session.get.assert_called_once_with(
            "https://example.com/img.png", timeout=30.0
        )

    @mock.patch("torchtitan.hf_datasets.multimodal.utils.image._http_session")
    def test_invalid_ip_error_propagates(self, mock_session):
        """Private-IP / metadata errors from requests-hardened must surface."""
        mock_session.get.side_effect = InvalidIPAddress(
            "Blocked: 169.254.169.254 is a link-local address"
        )

        with pytest.raises(InvalidIPAddress):
            _fetch_url_safe("http://169.254.169.254/latest/meta-data/")

    def test_session_config_blocks_loopback(self):
        """The hardened session must have IP filtering enabled."""
        assert _http_session.config.ip_filter_enable is True
        assert _http_session.config.ip_filter_allow_loopback_ips is False

    def test_session_redirect_bounded(self):
        """Redirect chains can never exceed 10 hops."""
        assert _http_session.max_redirects == 10
