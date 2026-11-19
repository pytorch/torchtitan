# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next generation wrapper that feeds PLE its token IDs and positions."""

from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper


class VLLMFlashNextModelWrapper(VLLMModelWrapper):
    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        if input_ids is None or positions is None or inputs_embeds is not None:
            raise ValueError(
                "Flash-Next generation requires token IDs and text positions"
            )
        with self.parallelism_context.activate_spmd():
            return self.model(
                input_ids,
                positions=positions,
                document_positions=positions,
                ple_input_ids=input_ids,
                attention_metadata=None,
                **self._prepare_step_kwargs(input_ids.shape[0]),
            )
