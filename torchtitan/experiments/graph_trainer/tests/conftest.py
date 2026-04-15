# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "sm90_only: mark test as requiring sm_90+ GPU (e.g. H100)",
    )
