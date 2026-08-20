import unittest

from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_mtp_debugmodel


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model_spec.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)


if __name__ == "__main__":
    unittest.main()
