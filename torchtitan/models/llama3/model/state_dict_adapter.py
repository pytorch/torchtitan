# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class Llama3StateDictAdapter(StateDictAdapter):
    @staticmethod
    def to_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement this
        return state_dict

    @staticmethod
    def from_hf(hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement this
        return hf_state_dict
