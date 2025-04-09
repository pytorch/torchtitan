# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training.guidance",
        type=float,
        default=3.5,
        help="guidance value used for guidance distillation",
    )
    parser.add_argument(
        "--encoder.t5_encoder",
        type=str,
        default="google/t5-v1_1-small",
        help="T5 encoder to use, HuggingFace model name.",
    )
    parser.add_argument(
        "--encoder.clip_encoder",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Clip encoder to use, HuggingFace model name.",
    )
    parser.add_argument(
        "--encoder.encoder_dtype",
        type=torch.dtype,
        default=torch.bfloat16,
        help="Which dtype to load for autoencoder. ",
    )
    parser.add_argument(
        "--encoder.max_t5_encoding_len",
        type=int,
        default=512,
        help="Maximum length of the T5 encoding.",
    )
    parser.add_argument(
        "--encoder.encoder_data_parallel_shard",
        action="store_true",
        help="Whether to shard the encoder using FSDP",
    )
