# If these tests fail, the merge is messing up grpo at minimum. please fix it.
import unittest


class TestDownstreamCanaries(unittest.TestCase):
    def test_qwen2_exists(self):
        from torchtitan.models import _supported_models

        assert "qwen2" in _supported_models, "You deleted qwen2 :(."

    def test_qwen2_loads(self):
        import torchtitan.protocols.train_spec as ts

        spec = ts.get_train_spec("qwen2")
        assert spec is not None, "qwen2 train spec is broken."

    def test_wandb_group_autogenerates(self):
        import inspect

        from torchtitan.components.metrics import WandBLogger

        src = inspect.getsource(WandBLogger.__init__)
        assert (
            "generate_id" in src
        ), "WandB group auto-generation was removed. Runs will be ungrouped."


if __name__ == "__main__":
    unittest.main()
