# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.lfm2.model.args import LFM2ModelArgs
from torchtitan.experiments.lfm2.model.model import LFM2Model

from ..simple_fsdp import disable_active_parametrization


class SimpleFSDPLFM2Model(LFM2Model):
    def __init__(self, model_args: LFM2ModelArgs):
        # IMPORTANT: Disable weight tying for SimpleFSDP compatibility
        # SimpleFSDP's per-module parametrization doesn't support weight tying
        # because it creates separate autograd nodes for each module, causing
        # double gradient reduction on tied weights.
        original_tie_setting = model_args.tie_word_embeddings
        model_args.tie_word_embeddings = False

        super().__init__(model_args)

        # Restore original setting (though it's not used after init)
        model_args.tie_word_embeddings = original_tie_setting

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)
