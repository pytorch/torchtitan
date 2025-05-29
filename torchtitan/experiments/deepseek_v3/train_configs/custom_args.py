from dataclasses import dataclass, field


@dataclass
class Parallelism:
    expert_parallel_degree: int = 2
    """ degree to parallelize experts """


@dataclass
class Training:
    steps: int = 22222222


@dataclass
class JobConfig:
    parallelism: Parallelism = field(default_factory=Parallelism)
    training: Training = field(default_factory=Training)
