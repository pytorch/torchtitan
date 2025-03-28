# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder.t5_encoder",
        type=str,
        default="google/t5-v1_1-small",
        help="T5 encoder to use, hugging face model name.",
    )
    parser.add_argument(
        "--encoder.clip_encoder",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Clip encoder to use, hugging face model name.",
    )
    parser.add_argument(
        "--encoder.encoder_device",
        type=str,
        default="cpu",
        help="Where to load the encoder, cpu or cuda.",
    )
    parser.add_argument(
        "--encoder.max_encoding_len",
        type=int,
        default=512,
        help="Maximum length of the T5 encoding.",
    )
