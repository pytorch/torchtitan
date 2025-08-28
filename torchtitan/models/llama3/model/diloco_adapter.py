from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.train_spec import OptimizersBuilder
from torchtitan.tools.logging import logger

class DiLoCoAdapter:
    """DiLoCo optimizer adapter that manages global optimizer and pseudo-gradient computation."""

    def __init__(
        self,
        model_parts: List[nn.Module],
        local_optimizer: Optimizer,
        num_local_steps: int = 1,
        global_lr: float = 0.7,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        # Store references to model parts
        self._model_parts = model_parts

        self._worker_model_params = []
        for model in self._model_parts:
            # Handle distributed and regular parameters
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self._worker_model_params.append(p)

        # Store local SGD specific fields.
        self._num_local_steps = num_local_steps
        self._local_step_counter = 0
        self._global_step_counter = 0

        # Create global optimizer
        self._global_model_params = [
            (p._local_tensor if hasattr(p, "_local_tensor") else p).clone().detach().requires_grad_(False)
            for p in self._worker_model_params
        ]
        self._global_optimizer = optim.SGD(
            self._global_model_params,
            lr=global_lr,
            momentum=momentum,
            nesterov=nesterov,
        )
        self._local_optimizer = local_optimizer

    @torch.no_grad()
    def _prepare_flat_buffers_from_model_params(
        self,
        params: list[torch.Tensor],
        copy: bool = False,
        device: torch.device | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Prepares the flat buffer based on the worker model parameters.

        Args:
            params (list[torch.Tensor]): List of worker model parameters.
            copy (bool): Whether to copy the parameters into flat buffer.
            device (torch.device | None): Device for flat buffer (default: params device).
        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: Tuple containing the list of
            pseudo-gradient buffers that are views into the flat buffer.
        """

        # Assert that all tensors are on the same device and have the same dtype.
        expected_device = params[0].device
        expected_dtype = params[0].dtype
        for i, param in enumerate(params):
            assert (
                param.device == expected_device
            ), f"All tensors must be on the same device, tensor {i} is on {param.device}, expecting {expected_device}"
            assert (
                param.dtype == expected_dtype
            ), f"All tensors must have the same dtype, tensor {i} is of dtype {param.dtype}, expecting {expected_dtype}"

        if not device:
            device = expected_device

        # Constructs flattened pseudo-gradient buffer based on parameter sizes.
        param_numels = [p.numel() for p in params]
        buffer = torch.empty((sum(param_numels),), device=device, dtype=expected_dtype)

        buffer_views = [
            v.view(p.size())
            for p, v in zip(
                params,
                torch.split(buffer, param_numels),
                strict=True,
            )
        ]

        if copy:
            torch._foreach_copy_(buffer_views, src=params)

        return buffer_views, buffer

    @torch.no_grad()
    def _compute_pseudo_gradients(
        self,
        pseudo_gradients: list[torch.Tensor],
        global_model_params: list[torch.Tensor],
        worker_model_params: list[torch.Tensor],
    ) -> None:
        """
        Computes the pseudo-gradients by subtracting the worker model parameters from
        the global model parameters.
        Args:
            global_model_params (list[torch.Tensor]): List of global model parameters.
            worker_model_params (list[torch.Tensor]): List of worker model parameters.
            pseudo_gradients (list[torch.Tensor]): Where computed pseudo-gradients
                must be stored.
        """
        torch._foreach_copy_(pseudo_gradients, src=global_model_params)
        torch._foreach_sub_(pseudo_gradients, other=worker_model_params)

    @torch.no_grad()
    def _set_pseudo_gradients(
        self, params: List[torch.Tensor], pseudo_gradients: List[torch.Tensor]
    ) -> None:
        """Set gradients of parameters to pseudo-gradients, handling distributed tensors."""
        for param, pseudo_gradient in zip(params, pseudo_gradients):
            if hasattr(param, "grad"):
                param.grad = pseudo_gradient
    
    @torch.no_grad()
    def post_step_hook(self, optimizer, args, kwargs):
        """
        Post optimization hook that implements DiLoCo logic.
        Adapted from nested_optimizer.py step() method.
        """
        self._local_step_counter += 1
        trigger_global_step = self._local_step_counter > 0 and (
            self._local_step_counter % self._num_local_steps == 0
        )

        if trigger_global_step:
            self._global_step_counter += 1

            # Step 1: Create buffer for pseudo-gradients
            pseudo_gradients, _ = self._prepare_flat_buffers_from_model_params(
                params=self._global_model_params
            )

            # Step 2: Compute pseudo-gradients and copy global params into worker params
            self._compute_pseudo_gradients(
                pseudo_gradients=pseudo_gradients,
                worker_model_params=[(p._local_tensor if hasattr(p, "_local_tensor") else p).detach() for p in self._worker_model_params],
                global_model_params=self._global_model_params,
            )

            # Step 3: Set pseudo-gradients as gradients of the parameters
            self._set_pseudo_gradients(
                params=self._global_model_params,
                pseudo_gradients=pseudo_gradients,
            )
            
            # Step 4: Perform global optimizer step
            self._global_optimizer.step()

            param_numel = sum(p.numel() for p in self._global_model_params)
            pseudo_gradient_l2_norm = (
                sum(p.detach().pow(2).sum() for p in self._global_model_params) / max(1, param_numel)
            ).sqrt().item()
            weight_delta_l2_norm = (
                sum((g_p - (l_p._local_tensor if hasattr(l_p, "_local_tensor") else l_p).detach()).pow(2).sum() for l_p, g_p in zip(self._worker_model_params, self._global_model_params))
                / max(1, param_numel)
            ).sqrt().item()

            logger.info(f"local_step: {self._local_step_counter} global_step: {self._global_step_counter} {pseudo_gradient_l2_norm=} {weight_delta_l2_norm=} {param_numel=}")

            # Step 5: Update worker model parameters copy
            for local_p, global_p in zip(self._worker_model_params, self._global_model_params):
                if hasattr(local_p, "_local_tensor"):
                    local_p.data.lerp_(
                        DTensor.from_local(
                            global_p,
                            local_p.device_mesh,
                            local_p.placements,
                            shape=local_p.shape,
                            stride=local_p.stride(),
                        ),
                        1.0,
                    )
                else:
                    local_p.data.lerp_(global_p, 1.0)

            self._global_optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        """
        Returns state dict containing both local and global optimizer states,
        as well as step counters and global model parameters.
        """
        state_dict = {
            "local_step_counter": self._local_step_counter,
            "global_step_counter": self._global_step_counter,
            "global_optimizer": self._global_optimizer.state_dict(),
            "global_model_params": [p.clone() for p in self._global_model_params],
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads state dict to restore local and global optimizer states,
        step counters, and global model parameters.
        """
        self._local_step_counter = state_dict["local_step_counter"]
        self._global_step_counter = state_dict["global_step_counter"]
        self._global_optimizer.load_state_dict(state_dict["global_optimizer"])
        
        # Restore global model parameters
        for i, saved_param in enumerate(state_dict["global_model_params"]):
            self._global_model_params[i].data.copy_(saved_param.data)

    def post_state_dict_hook(self, optimizer: Optimizer, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Post state dict hook to include DiLoCo adapter state in the optimizer's state dict.
        This hook is called after the optimizer's state_dict() method.
        """
        state_dict["diloco_adapter"] = self.state_dict()
        return state_dict

    def pre_load_state_dict_hook(self, optimizer: Optimizer, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Pre load state dict hook to extract DiLoCo adapter state from the optimizer's state dict.
        This hook is called before the optimizer's load_state_dict() method.
        """
        if "diloco_adapter" in state_dict:
            diloco_state = state_dict.pop("diloco_adapter")
            self.load_state_dict(diloco_state)
        return state_dict


def build_diloco_optimizers(
    build_local_optimizers_fn: OptimizersBuilder,
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Build DiLoCo optimizer by wrapping the regular optimizer with DiLoCo adapter."""

    # Build regular optimizers first
    optimizers = build_local_optimizers_fn(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )

    # from tools.debugging.pdb import vscode_debug_for_rank

    # vscode_debug_for_rank()

    # Create DiLoCo adapter (without auto-registering hook)
    diloco_adapter = DiLoCoAdapter(
        model_parts, 
        optimizers.optimizers[0], 
        num_local_steps=optimizer_config.num_local_steps, 
        global_lr=optimizer_config.global_lr, 
        momentum=optimizer_config.global_momentum, 
        nesterov=optimizer_config.global_nesterov
    )

    # Register DiLoCo hooks with the optimizers container
    optimizers.register_step_post_hook(diloco_adapter.post_step_hook)
    optimizers.register_state_dict_post_hook(diloco_adapter.post_state_dict_hook)
    optimizers.register_load_state_dict_pre_hook(diloco_adapter.pre_load_state_dict_hook)

    return optimizers
