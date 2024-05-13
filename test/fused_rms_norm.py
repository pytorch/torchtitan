import importlib
import numbers

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F

global titan_cuda
titan_cuda = None


# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input




class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        global titan_cuda
        if titan_cuda is None:
            titan_cuda = importlib.import_module("titan_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = titan_cuda.rms_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, invvar)
        else:
            ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = titan_cuda.rms_backward_affine(
           grad_output.contiguous(), invvar, input_or_output,
           ctx.normalized_shape, weight_, ctx.eps, ctx.memory_efficient
        )
        return grad_input, grad_weight, None, None, None


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps, memory_efficient=False):
        global titan_cuda
        if titan_cuda is None:
            titan_cuda = importlib.import_module("titan_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        output, invvar = titan_cuda.rms_forward(input_, ctx.normalized_shape, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, invvar)
        else:
            ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = titan_cuda.rms_backward(
            grad_output.contiguous(), invvar, input_or_output,
            ctx.normalized_shape, ctx.eps, ctx.memory_efficient
        )
        return grad_input, None, None, None



class FusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, memory_efficient=False):
        super().__init__()

        global titan_cuda
        titan_cuda = importlib.import_module("titan_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        output = FusedRMSNormAffineFunction.apply(input.float(), self.weight, self.normalized_shape, self.eps, self.memory_efficient)
        return output.type_as(input)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
