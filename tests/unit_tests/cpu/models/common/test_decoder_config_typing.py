# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, fields, replace
from typing import get_type_hints

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import Attention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.models.llama3 import build_model_config
from torchtitan.models.llama3.model import Llama3TransformerBlock


def test_decoder_config_uses_concrete_base_annotations():
    assert get_type_hints(TransformerBlock.Config)["attention"] is Attention.Config
    assert get_type_hints(Decoder.Config)["layers"] == list[TransformerBlock.Config]
    assert get_type_hints(MTPDecoder.Config)["layers"] == list[TransformerBlock.Config]


def test_decoder_subclass_preserves_mutable_layers_serialization_and_build_owner():
    class TypedDecoder(Decoder):
        @dataclass(kw_only=True, slots=True)
        class Config(Decoder.Config):
            pass

    source = build_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
    config = TypedDecoder.Config(
        **copy.deepcopy(
            {
                field.name: getattr(source, field.name)
                for field in fields(Decoder.Config)
            }
        )
    )
    source_count = len(source.layers)
    config.layers.append(replace(config.layers[0]))
    config.layers[0] = replace(config.layers[0])
    assert len(source.layers) == source_count
    assert len(config.to_dict()["layers"]) == source_count + 1

    with torch.device("meta"):
        model = config.build()
    assert isinstance(model, TypedDecoder)
    assert model.config is not config
    assert len(model.layers) == source_count + 1
    assert all(
        isinstance(layer, Llama3TransformerBlock) for layer in model.layers.values()
    )


def test_mtp_decoder_subclass_keeps_main_and_prediction_layer_lists_separate():
    class TypedMTPDecoder(MTPDecoder):
        @dataclass(kw_only=True, slots=True)
        class Config(MTPDecoder.Config):
            pass

    source = build_deepseek_config(
        "debugmodel",
        seq_len=128,
        attn_backend="flex",
        parallelism=ParallelismConfig(),
        num_mtp_layers=1,
    )
    config = TypedMTPDecoder.Config(
        **copy.deepcopy(
            {
                field.name: getattr(source, field.name)
                for field in fields(MTPDecoder.Config)
            }
        )
    )
    config.layers.append(replace(config.layers[0]))
    serialized = config.to_dict()
    assert len(serialized["layers"]) == len(source.layers) + 1
    assert len(serialized["mtp_layers"]) == 1

    with torch.device("meta"):
        model = config.build()
    assert isinstance(model, TypedMTPDecoder)
    assert len(model.layers) == len(config.layers)
    assert len(model.mtp_layers) == 1
    assert len(source.mtp_layers) == 1
