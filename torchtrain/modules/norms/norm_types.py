
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtrain.utils import StrEnum
from enum import auto
from abc import abstractmethod
from typing import Optional, Tuple, Union

class NormType(StrEnum):
    # default classical layernorm without bias
    layernorm = "layernorm"

    # A non-parametric (no affine transform) version of LayerNorm.
    np_layernorm = "np_layernorm"

    # RMSNorm
    rms = "rms"

    # Fused RMSNorm
    fused_rms = "fused_rms"


class NormBase(nn.Module):
    """
    Base class for normalization layers.
    Inspiration from OLMo LLM design
    """
    def __init__(
        self,
        size: int,
        eps: float = 1e-06,
        *,
        elementwise_affine: Optional[bool] = True,

    ):
        super().__init__()

        self.eps = eps
        self.normalized_shape = size,
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape,))
        else:
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, norm_type: NormType, dim: int, **kwargs) -> nn.Module:
        if norm_type == NormType.layernorm:
            return LayerNorm(dim,  **kwargs)
        elif norm_type == NormType.np_layernorm:
            return NPLayerNorm(dim, **kwargs)
        elif norm_type == NormType.rms:
            return RMSNorm(dim, **kwargs)
        elif norm_type == NormType.fused_rms:
            return FusedRMSNorm(dim, **kwargs)
        else:
            raise NotImplementedError(f"Unknown Norm type: '{norm_type}'")

    def init_weights(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore


class LayerNorm(NormBase):
    """ Classical LayerNorm, without bias. """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-06,
        elementwise_affine: Optional[bool] = True,

    ):
        super().__init__(size=dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)


class NPLayerNorm(NormBase):
    """ Non Parametric LayerNorm - no affine transform. """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: Optional[bool] = False,

    ):
        super().__init__(size=dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)


class FusedRMSNorm(NormBase):
    """ Fused RMS Norm """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__(size=dim, elementwise_affine=True, eps=eps)
        try:
            from torchtrain.modules.norms.fused_rms_norm import fused_rms_norm_fn
        except ImportError:
            raise ImportError("Please ensure fused_rms_norm.py is available.")

        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ leverages Triton Fused RMS Norm kernel """
        return self.fused_rms_norm_fn(
            x,
            self.weight,
            eps=self.eps,
        )


class RMSNorm(NormBase):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(size=dim, eps = eps)

    def _norm(self, x: torch.Tensor):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
