# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger

from .args import Nemotron3ModelArgs


class Nemotron3StateDictAdapter(MoEStateDictAdapter):
    """State dict adapter for NemotronH models (hybrid Mamba + Attention + MoE).

    Handles conversion between torchtitan's native state dict format
    and HuggingFace's NemotronH format.

    HuggingFace NemotronH format:
        - backbone.embeddings.weight
        - backbone.layers.{}.norm.weight
        - backbone.layers.{}.mixer.* (Mamba, Attention, MLP, or MoE)
        - backbone.norm_f.weight
        - lm_head.weight

    Torchtitan native format:
        - tok_embeddings.weight
        - layers.{}.norm.weight
        - layers.{}.mixer.* (Mamba2Mixer, Attention, MLP, or MoE)
        - norm.weight
        - output.weight
    """

    def __init__(
        self,
        model_args: Nemotron3ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        # Static mappings (non-layer keys)
        self.static_from_hf_map: dict[str, str | None] = {
            "backbone.embeddings.weight": "tok_embeddings.weight",
            "backbone.norm_f.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Layer-level mappings with {} placeholder for layer number
        # Format: HF key pattern -> Native key pattern

        # Layer norm (pre-normalization)
        self.layer_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.norm.weight": "layers.{}.norm.weight",
        }

        # Mamba2 mixer components
        self.mamba_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.in_proj.weight": "layers.{}.mixer.in_proj.weight",
            "backbone.layers.{}.mixer.conv1d.weight": "layers.{}.mixer.conv1d.weight",
            "backbone.layers.{}.mixer.conv1d.bias": "layers.{}.mixer.conv1d.bias",
            "backbone.layers.{}.mixer.out_proj.weight": "layers.{}.mixer.out_proj.weight",
            "backbone.layers.{}.mixer.dt_bias": "layers.{}.mixer.dt_bias",
            "backbone.layers.{}.mixer.D": "layers.{}.mixer.D",
            "backbone.layers.{}.mixer.A_log": "layers.{}.mixer.A_log",
            "backbone.layers.{}.mixer.norm.weight": "layers.{}.mixer.norm.weight",
        }

        # Attention mixer components (HF uses q/k/v/o_proj, native uses wq/wk/wv/wo)
        self.attention_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.q_proj.weight": "layers.{}.mixer.wq.weight",
            "backbone.layers.{}.mixer.k_proj.weight": "layers.{}.mixer.wk.weight",
            "backbone.layers.{}.mixer.v_proj.weight": "layers.{}.mixer.wv.weight",
            "backbone.layers.{}.mixer.o_proj.weight": "layers.{}.mixer.wo.weight",
        }

        # MLP mixer components (same names in both formats)
        self.mlp_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.up_proj.weight": "layers.{}.mixer.up_proj.weight",
            "backbone.layers.{}.mixer.down_proj.weight": "layers.{}.mixer.down_proj.weight",
        }

        # MoE mixer components (gate/router)
        self.moe_gate_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.gate.weight": "layers.{}.mixer.gate.weight",
            "backbone.layers.{}.mixer.gate.e_score_correction_bias": "layers.{}.mixer.gate.e_score_correction_bias",
        }

        # MoE shared experts
        self.moe_shared_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.shared_experts.up_proj.weight": "layers.{}.mixer.shared_experts.up_proj.weight",
            "backbone.layers.{}.mixer.shared_experts.down_proj.weight": "layers.{}.mixer.shared_experts.down_proj.weight",
        }

        # MoE routed experts (uses {} for layer number and expert number)
        self.moe_experts_from_hf_map: dict[str, str | None] = {
            "backbone.layers.{}.mixer.experts.{}.up_proj.weight": "layers.{}.mixer.experts.{}.up_proj.weight",
            "backbone.layers.{}.mixer.experts.{}.down_proj.weight": "layers.{}.mixer.experts.{}.down_proj.weight",
        }

        # Combined map for all layer components (excluding MoE experts which need special handling)
        self.from_hf_map: dict[str, str | None] = {
            **self.static_from_hf_map,
            **self.layer_from_hf_map,
            **self.mamba_from_hf_map,
            **self.attention_from_hf_map,
            **self.mlp_from_hf_map,
            **self.moe_gate_from_hf_map,
            **self.moe_shared_from_hf_map,
        }

    def _get_abstract_key(self, key: str) -> str:
        """Convert a concrete key to an abstract key with {} placeholders for layer/expert numbers.

        Only replaces numbers that appear between dots (e.g., layer indices, expert indices),
        not numbers that are part of component names (e.g., 'conv1d').
        """
        # Replace numbers that are surrounded by dots (layer.X.mixer or experts.X.up_proj)
        # Pattern: matches numbers that come after a dot and before a dot
        return re.sub(r"\.(\d+)\.", ".{}.", key)

    def _extract_layer_and_expert_nums(self, key: str) -> tuple[str | None, str | None]:
        """Extract layer number and optionally expert number from a key."""
        # Match patterns like backbone.layers.5.mixer.experts.42.up_proj.weight
        numbers = re.findall(r"(\d+)", key)
        if len(numbers) >= 2:
            return numbers[0], numbers[1]  # layer_num, expert_num
        elif len(numbers) == 1:
            return numbers[0], None  # layer_num only
        return None, None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict

        Raises:
            ValueError: If any keys could not be converted or if extraction fails.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}

        # Also build reverse map for MoE experts
        moe_experts_to_hf_map = {
            v: k for k, v in self.moe_experts_from_hf_map.items() if v is not None
        }

        # Track unprocessed keys - we'll pop from this set as we process
        remaining_keys = set(state_dict.keys())
        hf_state_dict = {}

        for key, value in state_dict.items():
            # Handle MoE expert weights (two {} placeholders)
            if ".experts." in key and ".shared_experts." not in key:
                abstract_key = self._get_abstract_key(key)
                layer_num, expert_num = self._extract_layer_and_expert_nums(key)

                if abstract_key not in moe_experts_to_hf_map:
                    raise ValueError(
                        f"to_hf: No mapping for MoE expert key '{abstract_key}'"
                    )
                if layer_num is None or expert_num is None:
                    raise ValueError(
                        f"to_hf: Failed to extract layer/expert nums from '{key}'"
                    )

                new_key = moe_experts_to_hf_map[abstract_key]
                new_key = new_key.format(layer_num, expert_num)
                hf_state_dict[new_key] = value
                remaining_keys.remove(key)
                continue

            # Handle layer-level keys (one {} placeholder for layer number)
            if "layers" in key:
                abstract_key = self._get_abstract_key(key)
                layer_num, _ = self._extract_layer_and_expert_nums(key)

                if abstract_key not in to_hf_map:
                    raise ValueError(
                        f"to_hf: No mapping for layer key '{abstract_key}'"
                    )
                if layer_num is None:
                    raise ValueError(f"to_hf: Failed to extract layer num from '{key}'")

                new_key = to_hf_map[abstract_key]
                if new_key is not None:
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
                remaining_keys.remove(key)
                continue

            # Handle static keys (no {} placeholders)
            if key in to_hf_map:
                new_key = to_hf_map[key]
                if new_key is not None:
                    hf_state_dict[new_key] = value
                remaining_keys.remove(key)
            # Note: keys not in to_hf_map will remain in remaining_keys

        # Fail if any keys were not processed
        if remaining_keys:
            raise ValueError(
                f"to_hf: {len(remaining_keys)} keys not converted: {list(remaining_keys)[:5]}"
            )

        logger.info(
            f"StateDictAdapter.to_hf: Converted {len(hf_state_dict)} keys (native -> HF)"
        )
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict

        Raises:
            ValueError: If any HF keys could not be converted or if extraction fails.
        """
        state_dict = {}

        for key, value in hf_state_dict.items():
            # Handle MoE expert weights (two {} placeholders)
            if ".experts." in key and ".shared_experts." not in key:
                abstract_key = self._get_abstract_key(key)
                layer_num, expert_num = self._extract_layer_and_expert_nums(key)

                if abstract_key not in self.moe_experts_from_hf_map:
                    raise ValueError(
                        f"from_hf: No mapping for MoE expert key '{abstract_key}'"
                    )
                if layer_num is None or expert_num is None:
                    raise ValueError(
                        f"from_hf: Failed to extract layer/expert nums from '{key}'"
                    )

                new_key = self.moe_experts_from_hf_map[abstract_key]
                if new_key is not None:
                    new_key = new_key.format(layer_num, expert_num)
                    state_dict[new_key] = value
                continue

            # Handle layer-level keys (one {} placeholder for layer number)
            if "layers" in key:
                abstract_key = self._get_abstract_key(key)
                layer_num, _ = self._extract_layer_and_expert_nums(key)

                if abstract_key not in self.from_hf_map:
                    raise ValueError(
                        f"from_hf: No mapping for layer key '{abstract_key}'"
                    )
                if layer_num is None:
                    raise ValueError(
                        f"from_hf: Failed to extract layer num from '{key}'"
                    )

                new_key = self.from_hf_map[abstract_key]
                if new_key is not None:
                    new_key = new_key.format(layer_num)
                    state_dict[new_key] = value
                continue

            # Handle static keys (no {} placeholders)
            if key in self.from_hf_map:
                new_key = self.from_hf_map[key]
                if new_key is not None:
                    state_dict[new_key] = value

        logger.info(
            f"StateDictAdapter.from_hf: Converted {len(state_dict)} keys (HF -> native)"
        )
        return state_dict
