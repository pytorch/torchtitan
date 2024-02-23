import pytest
from torchtrain.config_manager import JobConfig

class TestJobConfig():
    def test_job_config(self):
        config = JobConfig()
        assert config.model.name == "llama"

    def test_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            JobConfig("ohno.toml")
