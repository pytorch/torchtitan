# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro


@dataclass
class CustomArgs:
    how_is_your_day: str = "good" # Just an example.


@dataclass
class CustomArgsTwo:
    how_is_your_week: str = "great"
    """Just another example."""

    num_days: int = 7
    """Number of days in a week."""


@dataclass
class ExtendedConfig:
    """
    This is an example of how to extend the tyro parser with custom config classes.
    """

    custom_args: CustomArgs = field(default_factory=CustomArgs)
    custom_args_two: CustomArgsTwo = field(default_factory=CustomArgsTwo)


def extend_parser(parser: argparse.ArgumentParser) -> None:
    # TODO: Remove
    parser.add_argument(
        "--custom_args.how-is-your-day",
        type=str,
        default="good",
        help="Just an example.",
    )
