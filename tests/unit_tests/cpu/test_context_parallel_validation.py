# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Decoder.Config.update_from_config`` validates CP attention support."""

    @staticmethod
    def _config(*, cp: int, varlen: bool = False):
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        config.parallelism.context_parallel_degree = cp
        config.training.max_context_length = 512
        return config

    def test_allows_flex_cp(self):
        config = self._config(cp=2)
        config.model_spec.model.update_from_config(config=config)

    def test_rejects_varlen_cp(self):
        # Only FlexAttention's BlockMask represents global key positions for CP.
        config = self._config(cp=2, varlen=True)
        with self.assertRaisesRegex(NotImplementedError, "VarlenAttention"):
            config.model_spec.model.update_from_config(config=config)


if __name__ == "__main__":
    unittest.main()
