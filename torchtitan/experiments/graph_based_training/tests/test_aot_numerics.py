# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from .numerics_utils import run_numerics_test


class TestAOTNumerics(unittest.TestCase):
    """Test numerics equivalence between JIT and AOT compilation modes."""

    def test_llama3_fsdp_tp(self):
        """Test Llama3 with FSDP + TP configuration."""
        result = run_numerics_test(
            ngpu=4,
            config_file="./torchtitan/models/llama3/train_configs/debug_model.toml",
            dp_shard_degree=2,
            tp_degree=2,
            cp_degree=1,
            ep_degree=1,
            ac_mode="selective",
            steps=10,
            seed=42,
            eager_tb_folder="tb/test_llama3_fsdp_tp_jit",
            compiled_tb_folder="tb/test_llama3_fsdp_tp_aot",
            metrics=["loss_metrics/global_avg_loss", "grad_norm"],
        )
        self.assertTrue(result, "Llama3 FSDP+TP numerics test failed")

    def test_llama3_fsdp_tp_autobucketing(self):
        result = run_numerics_test(
            ngpu=4,
            config_file="./torchtitan/models/llama3/train_configs/debug_model.toml",
            dp_shard_degree=2,
            tp_degree=2,
            cp_degree=1,
            ep_degree=1,
            ac_mode="selective",
            steps=10,
            seed=42,
            eager_tb_folder="tb/test_llama3_fsdp_tp_autobucketing_jit",
            compiled_tb_folder="tb/test_llama3_fsdp_tp_autobucketing_aot",
            metrics=["loss_metrics/global_avg_loss", "grad_norm"],
            passes="auto_bucketing",
        )
        self.assertTrue(result, "Llama3 FSDP+TP+autobucketing numerics test failed")

    def test_llama3_fsdp_tp_manualbucketing(self):
        result = run_numerics_test(
            ngpu=4,
            config_file="./torchtitan/models/llama3/train_configs/debug_model.toml",
            dp_shard_degree=2,
            tp_degree=2,
            cp_degree=1,
            ep_degree=1,
            ac_mode="selective",
            steps=10,
            seed=42,
            eager_tb_folder="tb/test_llama3_fsdp_tp_manualbucketing_jit",
            compiled_tb_folder="tb/test_llama3_fsdp_tp_manualbucketing_aot",
            metrics=["loss_metrics/global_avg_loss", "grad_norm"],
            passes="transformer_block_bucketing",
        )
        self.assertTrue(result, "Llama3 FSDP+TP+manualbucketing numerics test failed")

    def test_deepseek_v3_fsdp_tp_ep(self):
        """Test DeepSeek V3 with FSDP + TP + EP configuration."""
        result = run_numerics_test(
            ngpu=4,
            config_file="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml",
            dp_shard_degree=2,
            tp_degree=2,
            cp_degree=1,
            ep_degree=4,
            ac_mode="none",
            steps=10,
            seed=42,
            eager_tb_folder="tb/test_deepseek_v3_fsdp_tp_ep_jit",
            compiled_tb_folder="tb/test_deepseek_v3_fsdp_tp_ep_aot",
            metrics=["loss_metrics/global_avg_loss", "grad_norm"],
        )
        self.assertTrue(result, "DeepSeek V3 FSDP+TP+EP numerics test failed")


if __name__ == "__main__":
    unittest.main()
