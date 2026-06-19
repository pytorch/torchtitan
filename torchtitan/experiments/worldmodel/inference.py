from dataclasses import dataclass
from typing import Any

import torch

from xx.training.diffusion.schedulers import RFScheduler

from torchtitan.experiments.worldmodel.model import WorldModel


@dataclass(frozen=True, slots=True)
class ModelIO:
    in_shape: dict[str, torch.Size]
    in_dtype: dict[str, torch.dtype]
    out_shape: dict[str, torch.Size]
    out_dtype: dict[str, torch.dtype]

    def to_dict(self) -> dict[str, dict[str, torch.Size] | dict[str, torch.dtype]]:
        return {
            "in_shape": self.in_shape,
            "in_dtype": self.in_dtype,
            "out_shape": self.out_shape,
            "out_dtype": self.out_dtype,
        }


class WorldModelForInference(WorldModel):
    @staticmethod
    def input_shapes(config: WorldModel.Config, batch_size: int = 1) -> dict[str, tuple[int, ...]]:
        frames, height, width = config.input_size
        pose_size = config.pose_size // 2
        return {
            "latents": (batch_size, frames, config.in_channels, height, width),
            "augments_pos_ref_augment": (batch_size, frames, pose_size),
            "ref_augment_from_augments_euler": (batch_size, frames, pose_size),
            "pose_mask": (batch_size, frames),
            "fidxs": (batch_size, frames),
        }

    @staticmethod
    def input_dtypes(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.dtype]:
        return {
            "latents": dtype,
            "augments_pos_ref_augment": dtype,
            "ref_augment_from_augments_euler": dtype,
            "pose_mask": torch.int64,
            "fidxs": torch.int64,
        }

    @classmethod
    def example_inputs(
        cls,
        config: WorldModel.Config,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "meta",
    ) -> dict[str, torch.Tensor]:
        dtypes = cls.input_dtypes(dtype)
        return {
            name: torch.randn(shape, dtype=dtypes[name], device=device) if dtypes[name].is_floating_point else torch.ones(shape, dtype=dtypes[name], device=device)
            for name, shape in cls.input_shapes(config, batch_size=batch_size).items()
        }

    def get_model_io(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        steps: int = 1,
        num_conditioning_frames: int = 14,
    ) -> dict[str, dict[str, torch.Size] | dict[str, torch.dtype]]:
        inputs = self.example_inputs(self.config, batch_size=batch_size, dtype=dtype, device=self.pos_embed.device)
        outputs = self.generate(
            **inputs,
            dtype=dtype,
            steps=steps,
            num_conditioning_frames=num_conditioning_frames,
        )
        return ModelIO(
            in_shape={key: value.shape for key, value in inputs.items()},
            in_dtype={key: value.dtype for key, value in inputs.items()},
            out_shape={key: value.shape for key, value in outputs.items()},
            out_dtype={key: value.dtype for key, value in outputs.items()},
        ).to_dict()

    def compile_for_inference(self) -> None:
        for block in self.blocks:
            block.compile(mode="max-autotune-no-cudagraphs")

    @torch.inference_mode()
    def generate(
        self,
        latents: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        *,
        steps: int = 15,
        num_conditioning_frames: int = 14,
        dtype: torch.dtype = torch.bfloat16,
        inference_schedule: str = "linear",
        cfg: float = 0.0,
        return_trajectory: bool = False,
        **scheduler_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        latents = latents.to(dtype=dtype)
        batch, frames = latents.shape[:2]
        if not 0 <= num_conditioning_frames <= frames:
            raise ValueError(f"num_conditioning_frames={num_conditioning_frames} must be in [0, {frames}]")

        if steps <= 0:
            return self._prefill_only(
                latents=latents,
                augments_pos_ref_augment=augments_pos_ref_augment,
                ref_augment_from_augments_euler=ref_augment_from_augments_euler,
                pose_mask=pose_mask,
                fidxs=fidxs,
                num_conditioning_frames=num_conditioning_frames,
                return_trajectory=return_trajectory,
            )

        if self.final_layer is None:
            raise ValueError("diffusion sampling requires model.out_channels > 0")

        scheduler = RFScheduler(steps=steps, inference_schedule=inference_schedule, **scheduler_kwargs).to(device=latents.device)
        conditioning = self._prefill_conditioning(latents, num_conditioning_frames)
        future = latents[:, num_conditioning_frames:]
        trajectory = [self.unscale_latents(future)] if return_trajectory else None
        model_output: dict[str, torch.Tensor] = {}

        for step_idx in range(steps):
            full_latents, timesteps = self._decode_inputs(
                conditioning=conditioning,
                future=future,
                batch=batch,
                frames=frames,
                num_conditioning_frames=num_conditioning_frames,
                timestep=scheduler.timesteps[step_idx],
            )
            model_output = self(
                x=full_latents,
                t=timesteps,
                augments_pos_ref_augment=augments_pos_ref_augment,
                ref_augment_from_augments_euler=ref_augment_from_augments_euler,
                pose_mask=pose_mask,
                fidx=fidxs,
            )
            velocity = model_output["sample"][:, num_conditioning_frames:]
            if cfg > 0.0:
                unconditional_output = self(
                    x=full_latents,
                    t=timesteps,
                    augments_pos_ref_augment=torch.zeros_like(augments_pos_ref_augment),
                    ref_augment_from_augments_euler=torch.zeros_like(ref_augment_from_augments_euler),
                    pose_mask=torch.ones_like(pose_mask),
                    fidx=fidxs,
                )
                unconditional_velocity = unconditional_output["sample"][:, num_conditioning_frames:]
                velocity = unconditional_velocity + cfg * (velocity - unconditional_velocity)
            future = (future + scheduler.dt[step_idx].view(1, *([1] * (future.ndim - 1))) * velocity).to(dtype=dtype)
            if trajectory is not None:
                trajectory.append(self.unscale_latents(future))

        if not latents.is_meta and not all(torch.isfinite(value).all() for value in model_output.values()):
            raise ValueError("model outputs contain inf/nan")

        outputs = {"latents": self.unscale_latents(future)}
        if "plan" in model_output:
            outputs["plan"] = model_output["plan"]
        if trajectory is not None:
            outputs["trajectory"] = torch.stack(trajectory, dim=1)
        return outputs

    def _prefill_only(
        self,
        *,
        latents: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        num_conditioning_frames: int,
        return_trajectory: bool,
    ) -> dict[str, torch.Tensor]:
        batch, frames = latents.shape[:2]
        timesteps = torch.full(
            (batch, frames),
            RFScheduler.no_noise_timestep_value,
            device=latents.device,
            dtype=torch.float32,
        )
        scaled_latents = self.scale_latents(latents)
        model_output = self(
            x=scaled_latents,
            t=timesteps,
            augments_pos_ref_augment=augments_pos_ref_augment,
            ref_augment_from_augments_euler=ref_augment_from_augments_euler,
            pose_mask=pose_mask,
            fidx=fidxs,
        )
        start = max(0, num_conditioning_frames - 1)
        output_latents = self.unscale_latents(scaled_latents[:, start:])
        outputs = {"latents": output_latents}
        if "plan" in model_output:
            outputs["plan"] = model_output["plan"]
        if return_trajectory:
            outputs["trajectory"] = output_latents.unsqueeze(1)
        return outputs

    def _prefill_conditioning(self, latents: torch.Tensor, num_conditioning_frames: int) -> torch.Tensor:
        if num_conditioning_frames == 0:
            return latents[:, :0]
        return self.scale_latents(latents[:, :num_conditioning_frames])

    def _decode_inputs(
        self,
        *,
        conditioning: torch.Tensor,
        future: torch.Tensor,
        batch: int,
        frames: int,
        num_conditioning_frames: int,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_latents = torch.cat([conditioning, future], dim=1) if num_conditioning_frames else future
        timesteps = torch.ones((batch, frames), device=future.device, dtype=torch.float32) * timestep
        if num_conditioning_frames:
            timesteps[:, :num_conditioning_frames] = RFScheduler.no_noise_timestep_value
        return full_latents, timesteps


def main() -> None:
    from torchtitan.experiments.worldmodel.config_registry import model_registry

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = model_registry("debugmodel").model
    model = WorldModelForInference(config).to(device=device, dtype=torch.bfloat16).eval()
    inputs = WorldModelForInference.example_inputs(
        config,
        batch_size=2,
        dtype=torch.bfloat16,
        device=device,
    )
    outputs = model.generate(
        **inputs,
        dtype=torch.bfloat16,
        steps=10,
    )
    print(
        {
            "device": str(device),
            "inputs": {key: (tuple(value.shape), str(value.dtype)) for key, value in inputs.items()},
            "outputs": {key: (tuple(value.shape), str(value.dtype)) for key, value in outputs.items()},
        }
    )


if __name__ == "__main__":
    main()
