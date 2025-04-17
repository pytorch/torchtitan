# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training.classifer_free_guidance_prob",
        type=float,
        default=0.0,
        help="Classifier-free guidance with probability p to dropout the text conditioning",
    )
    parser.add_argument(
        "--training.img_size",
        type=int,
        default=256,
        help="Image width to sample",
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
        "--encoder.offload_encoder",
        action="store_true",
        help="Whether to shard the encoder using FSDP",
    )
    # eval configs
    parser.add_argument(
        "--eval.enable_classifer_free_guidance",
        action="store_true",
        help="Whether to use classifier-free guidance during sampling",
    )
    parser.add_argument(
        "--eval.classifier_free_guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale when sampling",
    )
    parser.add_argument(
        "--eval.denoising_steps",
        type=int,
        default=50,
        help="How many denoising steps to sample when generating an image",
    )
    parser.add_argument(
        "--eval.eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluation/sampling during training",
    )
    parser.add_argument(
        "--eval.save_img_folder",
        type=str,
        default="img",
        help="Directory to save image generated/sampled from the model",
    )
