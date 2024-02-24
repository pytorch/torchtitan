import pytest
from torchtrain.config_manager import JobConfig


class TestJobConfig:
    def test_command_line_args(self):
        config = JobConfig()
        config.parse_args([])
        assert config.model.name == "llama"

    def test_job_config_file(self):
        config = JobConfig()
        config.parse_args(
            ["--job.config_file", "./torchtrain/train_configs/train_config.toml"]
        )
        assert config.model.name == "llama"

    def test_job_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            config = JobConfig()
            config.parse_args(["--job.config_file", "ohno.toml"])
