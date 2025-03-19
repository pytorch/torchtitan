# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--custom_args.how-is-your-day",
        type=str,
        default="good",
        help="Just an example.",
    )
