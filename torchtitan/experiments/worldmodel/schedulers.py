from collections.abc import Callable

import torch


def linear_quadratic_schedule(x: torch.Tensor, num_timesteps_multiplier: float = 2.5) -> torch.Tensor:
    num_timesteps = len(x)
    timesteps_lin = torch.linspace(0, 1, int(num_timesteps * num_timesteps_multiplier), dtype=torch.float32)[: num_timesteps // 2 + 1]
    timesteps_quad = torch.square(torch.linspace(torch.sqrt(timesteps_lin[-1]), torch.tensor(1.0), num_timesteps - num_timesteps // 2))[1:]
    return 1 - torch.cat([timesteps_lin, timesteps_quad])


SCHEDULER_FUNCTIONS: dict[str, Callable] = {
    "linear": lambda x: x,
    "sqrt": lambda x: torch.sqrt(x),
    "cosine": lambda x, exponent=2: torch.sin((torch.pi * x) / 2) ** exponent,
    "pow": lambda x, exponent: torch.pow(x, exponent=exponent),
    "lin_quad": linear_quadratic_schedule,
}


class RFScheduler(torch.nn.Module):
    no_noise_timestep_value: float = 0.0
    full_noise_timestep_value: float = 1.0
    timesteps: torch.Tensor
    dt: torch.Tensor
    no_noise_timestep: torch.Tensor
    full_noise_timestep: torch.Tensor

    def __init__(self, steps: int = 15, inference_schedule: str = "linear", **kwargs):
        super().__init__()
        self.num_timesteps = steps + 1
        scheduler_function = SCHEDULER_FUNCTIONS[inference_schedule]
        timesteps = torch.linspace(1, 0, self.num_timesteps, dtype=torch.float32)
        self.register_buffer("timesteps", scheduler_function(timesteps, **kwargs))
        self.register_buffer("dt", -torch.diff(self.timesteps))
        self.register_buffer("no_noise_timestep", torch.tensor(self.no_noise_timestep_value, dtype=torch.float32))
        self.register_buffer("full_noise_timestep", torch.tensor(self.full_noise_timestep_value, dtype=torch.float32))

    def sample_timestep(self, shape: tuple) -> torch.Tensor:
        nt = torch.randn(shape, device=self.timesteps.device)
        return torch.sigmoid(nt)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)
        return (1 - timesteps) * original_samples + timesteps * noise

    def step(self, model_output: torch.Tensor, timestep_idx: int, sample: torch.Tensor) -> torch.Tensor:
        dt = self.dt[timestep_idx]
        dt = dt.view(1, *([1] * (sample.ndim - 1)))
        return sample + dt * model_output


class IdentityScheduler(torch.nn.Module):
    no_noise_timestep_value: float = 0.0
    full_noise_timestep_value: float = 1.0
    no_noise_timestep: torch.Tensor
    full_noise_timestep: torch.Tensor
    timesteps: torch.Tensor

    def __init__(self, steps: int = 15, *args, **kwargs):
        super().__init__()
        self.num_timesteps = steps + 1
        self.register_buffer("no_noise_timestep", torch.tensor(self.no_noise_timestep_value, dtype=torch.float32))
        self.register_buffer("full_noise_timestep", torch.tensor(self.full_noise_timestep_value, dtype=torch.float32))
        self.register_buffer("timesteps", torch.full((self.num_timesteps,), self.no_noise_timestep_value, dtype=torch.float32))

    def sample_timestep(self, shape: tuple, *args, **kwargs) -> torch.Tensor:
        return torch.ones(shape, device=self.no_noise_timestep.device) * self.no_noise_timestep

    def add_noise(self, original_samples: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return original_samples

    def step(self, *args, **kwargs) -> None:
        raise ValueError("IdentityScheduler does not support step method")
