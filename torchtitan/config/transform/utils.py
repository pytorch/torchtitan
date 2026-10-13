# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by model config transforms."""


def matches_fqn(fqn: str, target: str) -> bool:
    """Match an exact FQN or a dot-delimited suffix, without glob expansion."""
    return fqn == target or fqn.endswith(f".{target}")
