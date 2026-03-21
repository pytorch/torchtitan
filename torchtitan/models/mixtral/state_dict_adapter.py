# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Implement HF state dict conversion in step 4.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import MixtralModel


class MixtralStateDictAdapter(StateDictAdapter):
    """Stub — will be replaced with HF weight conversion."""

    def __init__(
        self, model_config: MixtralModel.Config, hf_assets_path: str | None
    ):
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Mixtral to_hf not yet implemented")

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Mixtral from_hf not yet implemented")
