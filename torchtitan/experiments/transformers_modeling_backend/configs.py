# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.transformers_modeling_backend.trainer import SFTTrainer

# The HF-backend config is just SFTTrainer's own Config. SFTTrainer defines a
# nested Config, so Configurable.__init_subclass__ auto-wires _owner=SFTTrainer;
# the inherited Config.build() then constructs an SFTTrainer (and keeps the
# config-field vs build()-kwarg overlap check) -- no reimplementation needed.
TransformersBackendConfig = SFTTrainer.Config
