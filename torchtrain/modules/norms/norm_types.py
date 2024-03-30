
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormBase(nn.Module):
    """
    Base class for normalization layers.
    Inspiration from OLMo LLM design
    """
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
        else:
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size,  **kwargs)
        elif config.layer_norm_type == LayerNormType.nonparametric:
            return LayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.fused_rms:
            return FusedRMSNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown Norm type: '{config.norm_type}'")

    def init_weights(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore


class LayerNorm(NormBase):
    """ Classical LayerNorm, without bias. """

    def __init__(
        self,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__(size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)

class RMSNorm(NormBase):
    """ RMS Norm """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(start_dtype)

        if self.weight is not None:
            return self.weight * x
        else:
            return x

class FusedRMSNorm(NormBase):
    """ Fused RMS Norm """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        try:
            from fused_rms_norm import fused_rms_norm_fn
        except ImportError:
            raise ImportError("Please ensure fused_rms_norm.py is in the same directory as your python file")

        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ leverages Triton Fused RMS Norm kernel """
        return fused_rms_norm_fn(
            x,
            self.weight,
            eps=self.eps,
        )
