# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' Harbor taskset for Terminal-Bench and TMax."""

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalTasksetConfig(HarborConfig):
    """Config of ``TerminalTaskset``: Harbor's fields, unchanged."""


class TerminalTaskset(HarborTaskset, vf.Taskset[HarborTask, TerminalTasksetConfig]):
    """Verifiers' Harbor taskset, unchanged."""

    config: TerminalTasksetConfig
