from torchtitan.config_manager import JobConfig
from torchtitan.train import Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(self, job_config=job_config)
