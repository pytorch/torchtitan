import pytest
from torchtrain.config_manager import JobConfig

class TestJobConfig():
    def test_command_line_args(self):
        config = JobConfig([])
        assert config.model.name == "llama"

    def test_command_line_args_with_override(self):
        config = JobConfig(["--metrics_log_freq" , "2", "--metrics_enable_tensorboard"])
        assert config.metrics.log_freq == 2
        assert config.metrics.enable_tensorboard

    def test_job_config_file(self):
        config = JobConfig(["--global_config_file", "./torchtrain/train_configs/train_config.toml"])
        assert config.model.name == "llama"

    def test_job_config_file_with_override(self):
        config = JobConfig(["--global_config_file",
                            "./torchtrain/train_configs/train_config.toml",
                            "--metrics_log_freq" , "2"])
        assert config.metrics.log_freq == 2

    def test_job_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            JobConfig(["--global_config_file", "ohno.toml"])
