from typing import Annotated

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class RMSLayerNorm(nn.Module):
    """RMS normalization class."""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        """
        Initializes a LayerNorm module.

        Args:
            ndim (int): The number of dimensions of the input tensor.
            bias (bool, optional): If True, adds a learnable bias to the normalized tensor. Defaults to True.
            epsilon (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.

        Note:
            Original paper: https://arxiv.org/pdf/1910.07467.pdf
            Source code adopted from https://github.com/facebookresearch/llama/blob/a0a4da8b497c566403941ceec47c2512ecf9dd20/llama/model.py#L34C1-L77C36

        Returns:
            None
        """

        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Applies layer normalization to the input tensor.

        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying layer normalization.

        """
        output = self._norm(x.float()).type_as(x)
        if self.bias is None:
            return output * self.weight
        else:
            return output * self.weight + self.bias

    def reset_parameters(self):
        # This function is called from the ModelFactory, when the model weights are initialized.
        # We recursively iterate through all the modules of a model and call reset_parameters if it exists.
        # Subsequently, we run the weight initialization on top, applying custom weight initialization to
        # a subset of modules based on the configuration.
        # Inpired by torch titan RMS Norm implementation:
        # https://github.com/pytorch/torchtitan/blob/de9fd2b9ea7e763c9182e0df81fc32c2618cc0b6/torchtitan/models/norms.py#L113C1-L114C57
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class LayerNormConfig(BaseModel):
    """
    Configuration class for Layer Normalization.

    Args:
        normalized_shape (int): The expected size of the input shape.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        elementwise_affine (bool, optional): Whether to include learnable affine parameters. Defaults to True.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    normalized_shape: Annotated[int, Field(strict=True, ge=1)]
    eps: Annotated[float, Field(strict=True, gt=0, default=1e-6)]
    elementwise_affine: Annotated[bool, Field(strict=True, default=True)]
    bias: Annotated[bool, Field(strict=True, default=True)]


class RMSLayerNormConfig(BaseModel):
    """
    Configuration class for RMSLayerNorm.

    Args:
        ndim (int): Number of dimensions for the input tensor. Must be greater than or equal to 1.
        epsilon (float, optional): Small value added to the input to avoid division by zero. Defaults to 1e-6.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    ndim: Annotated[int, Field(strict=True, ge=1)]
    epsilon: Annotated[float, Field(gt=0, default=1e-6)]
    bias: Annotated[bool, Field(strict=True, default=True)]
