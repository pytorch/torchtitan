# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor import DTensor

from .muon_utils import zeropower_backends
from .norm_helper import NORM_FUNCTIONS

__all__ = [
    "AbstractDiSCO",
]


class AbstractDiSCO(torch.optim.Optimizer):
    """
    Shared utilities for Spectral Conditioned Optimizer.

    This base class centralizes common functionality not specific to a
    particular distributed layout, including:
      - light-mode grad state save/load hooks
      - zero_grad handling for light mode
      - gradient normalisation helpers
      - optional tracking of norms across steps
    """

    def __init__(self, params, defaults, is_light: bool = False):
        # Initialize as a torch Optimizer and common state
        super().__init__(params, defaults)
        self.is_light: bool = is_light

        # Norm tracking state
        self.need_to_calculate_norm: bool = False
        self.norms_to_log: list[str] = list(NORM_FUNCTIONS.keys())
        self.norms_at_current_step: dict[str, torch.Tensor] = {}

    """
    def step(self, *args, **kwargs):
        momentum = update_moment(gradients)

        full_momentum_matrix = maybe_gather_from_shards(momentum)

        update = lmo(full_momentum_matrix)

        shard_update = maybe_scatter_to_shards(update)

        weights -= lr * shard_update
    """

    # ----- Light mode hooks -----
    def setup_light_state_hooks(self):
        if not self.is_light:
            return
        # Initialize state immediately to capture existing grads
        self._store_grads_in_state()
        # Register hooks so grads persist through state_dict save/load
        self.register_state_dict_pre_hook(type(self)._store_grads_in_state)
        self.register_load_state_dict_post_hook(type(self)._load_grads_from_state)

    def __getstate__(self):
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self, *args, **kwargs):
        # args/kwargs present to allow hook-style invocation
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})["grad_state"] = param.grad

    def _load_grads_from_state(self, *args, **kwargs):
        for param, state in self.state.items():
            if "grad_state" in state:
                param.grad = state["grad_state"]
            elif isinstance(param, torch.Tensor):
                param.grad = None

    # ----- Step norm tracking -----
    def calculate_norm_at_next_step(self, norms_to_log: list[str] = None):
        self.need_to_calculate_norm = True
        if norms_to_log is not None:
            self.norms_to_log = norms_to_log
        self.norms_at_current_step = {}

    def _is_logging_rank(self) -> bool:
        # Subclasses can override this to restrict logging to rank 0
        return True

    def get_norms_at_current_step(self):
        if self._is_logging_rank():
            return self.norms_at_current_step
        else:
            return {}

    def zero_grad(self, *args, **kwargs):
        # Preserve grads for light mode; otherwise use default behavior
        if self.is_light:
            return
        super().zero_grad(*args, **kwargs)

    @staticmethod
    @torch.no_grad()
    def normalise_grad(g: torch.Tensor, norm_factor: str, eps: float):
        """
        Normalises a gradient tensor. Handles both 2D [d_out, d_in] and
        3D [n_experts, d_out, d_in] tensors.
        """
        if norm_factor == "spectral":
            # Use the last two dims so this works for 2-D and batched 3-D
            g = g * (g.size(-2) / g.size(-1)) ** 0.5

        elif norm_factor == "image_spectral":
            ratio = (g.size(-2) / g.size(-1)) ** 0.5
            g = g * (ratio if ratio > 1 else 1)

        elif norm_factor.startswith("embed"):
            # Handle 2-D and batched 3-D consistently
            assert g.ndim == 2
            if g.ndim == 2:
                rms_values = torch.sqrt(g.pow(2).sum(dim=1, keepdim=True))
                dim = g.size(1)
            else:
                raise ValueError("embed* expects 2-D ")
            g = g / (rms_values + eps)
            if norm_factor == "embed_linear":
                g = g * dim
            elif norm_factor == "embed_sqrt":
                g = g * dim**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")

        elif norm_factor.startswith("unembed"):
            assert g.ndim == 2
            if g.ndim == 2:
                rms_values = torch.sqrt(g.pow(2).sum(dim=1, keepdim=True))
                dim = g.size(1)
            else:
                raise ValueError("unembed* expects 2-D ")
            g = g / (rms_values + eps)
            if norm_factor == "unembed_linear":
                g = g / dim
            elif norm_factor == "unembed_sqrt":
                g = g / dim**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")

        elif norm_factor == "sign":
            g = torch.sign(g)

        elif norm_factor == "bias_rms":
            rms_value = torch.sqrt(g.pow(2).mean())
            g = g / (rms_value + eps)

        elif norm_factor == "conv_spectral":
            # Properly handle Conv2D (4-D) and Conv3D (5-D)
            if g.ndim == 4:
                out_ch, in_ch, kh, kw = g.shape
                spatial = kh * kw
            elif g.ndim == 5:
                out_ch, in_ch, kh, kw, kd = g.shape
                spatial = kh * kw * kd
            else:
                raise ValueError("conv_spectral expects 4-D or 5-D conv weights")
            g *= (out_ch / in_ch) ** 0.5 / spatial

        elif norm_factor == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g

    @staticmethod
    @torch.no_grad()
    def lmo(
        g,
        eps,
        norm_factor,
        zeropower_backend,
        backend_steps,
        transpose_experts=False,
    ):
        """Supported Weight Types:
        - 1-D tensors: Bias vectors (Linear/Convolution layers)
        - 2-D tensors: Linear layer weights [D_out, D_in]
        - 3-D tensors: Grouped expert weights [G, D_in, D_out] or [G, D_out, D_in]
        - 4-D tensors: Conv2D weights [D_out, D_in, KH, KW] (forced to "conv_spectral")
        - 5-D tensors: Conv3D weights [D_out, D_in, KH, KW, KD] (forced to "conv_spectral")

        Limitations:
        - Does not support learnable RMS/Layer-norm parameters
        - Does not support shared experts or Fused GLU in format [D_in, D_out * M], where M > 1
        - Does not support Conv1D layers Note:
        - For 3-D expert weights, the layout must be specified during optimizer initialization.


        * 0-D (scalar) weights is supported but should not appear in this function call
        """

        g = g.to_local() if isinstance(g, DTensor) else g

        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        def _orth_and_norm(x):
            x = zeropower_backends[zeropower_backend](x, steps=backend_steps, eps=eps)
            x = AbstractDiSCO.normalise_grad(x, norm_factor=norm_factor, eps=eps)
            return x

        if g.ndim == 2:
            return _orth_and_norm(g)

        # 3-D: batched experts [G, D_out, D_in] (or [G, D_in, D_out] if transposed)
        elif g.ndim == 3:
            if g.shape[0] > 0:
                g = g.transpose(1, 2) if transpose_experts else g
                g = _orth_and_norm(
                    g
                )  # backend is batched; normaliser uses last two dims
                g = g.transpose(1, 2) if transpose_experts else g
            return g  # empty G==0 falls through unchanged

        # 1-D: bias vector
        elif g.ndim == 1:
            if zeropower_backend == "bias_rms" or norm_factor == "bias_rms":
                # cheap bias path
                return AbstractDiSCO.normalise_grad(g, norm_factor="bias_rms", eps=eps)
            # generic: lift to diagonal, apply 2-D logic, project back
            g_diag = torch.diag_embed(g).contiguous()
            g_diag = _orth_and_norm(g_diag)
            return g_diag.diagonal().contiguous()

        # 4-D/5-D: conv weights; flatten spatial dims into 'in' dim
        elif g.ndim == 4 or g.ndim == 5:
            shape = g.shape
            g2d = g.reshape(shape[0], -1)  # [D_out, D_in*prod(K)]
            g2d = zeropower_backends[zeropower_backend](
                g2d, steps=backend_steps, eps=eps
            )
            # force conv-specific scaling
            g2d = AbstractDiSCO.normalise_grad(
                g2d.view(shape), norm_factor="conv_spectral", eps=eps
            )
            return g2d.view(shape)

        else:
            raise ValueError(f"Unknown grad shape: {g.shape}")
