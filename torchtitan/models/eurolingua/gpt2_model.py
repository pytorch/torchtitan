import logging
import math
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator, validator

from torchtitan.models.eurolingua.layer_norms import LayerNormConfig, RMSLayerNorm, RMSLayerNormConfig
from torchtitan.models.eurolingua.lookup_enum import LookupEnum
from torchtitan.models.eurolingua.model import ActivationType, NNModel, SwiGLU
from torchtitan.models.eurolingua.util import parse_enum_by_name
from torchtitan.models.eurolingua.utils import convert_base_model_config_to_dict

# try:
#     from flash_attn import flash_attn_func
# except ModuleNotFoundError:
#     flash_attn_func = None

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class LayerNorms(LookupEnum):
    """
    Enum lookup class for LayerNorms.

    Attributes:
        RMSNorm: RMSLayerNorm class.
        LayerNorm: nn.LayerNorm class.
    """

    rms_norm = RMSLayerNorm
    layer_norm = nn.LayerNorm


class LayerNormWrapperConfig(BaseModel):
    norm_type: LayerNorms
    config: LayerNormConfig | RMSLayerNormConfig


class PositionTypes(str, Enum):
    """
    Enum class representing different position types.

    Attributes:
        ABSOLUTE (str): Represents the absolute position type.
        NOPE (str): Represents the nope (no postional emebddigns) position type.
    """

    ABSOLUTE = "ABSOLUTE"
    NOPE = "NOPE"


class QueryKeyValueTransform(nn.Module):
    """Query Key Value Transform base class."""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for transforming queries/keys/values.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the output tensors.
        """
        raise NotImplementedError


class IdentityTransform(QueryKeyValueTransform):
    """IdentityTransform class which does not apply any transform."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the IdentityTransform which does not apply any transform.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The tensors q, k, and v.
        """
        return q, k, v


class RotaryTransform(QueryKeyValueTransform):
    """
    RotaryTransform class which implements rotary positional embeddings.

    Source: https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
            We added the corresponding code here, becauase there is a conflict with "@torch.jit.script" used in the
            XFormers implementation and removed in this implementation.#
    """

    def __init__(self, n_embd: int, n_head: int, seq_length_dim: int = -2, base_freq: int = 10000):
        """
        Initializes the RotaryTransform object.

        Args:
            n_embd (int): The size of the embedding dimension.
            n_head (int): The number of attention heads.
            seq_length_dim (int, optional): The dimension along which the sequence length is defined. Defaults to -2.
            base_freq (int): Base frequency for RoPE. Defaults to 10000.
        """
        super().__init__()
        dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        """
        Rearange tentor elements.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update_cos_sin_tables(self, x):
        # Update the cosine and sine tables.
        seq_len = x.shape[self.seq_length_dim]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_length_dim], device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x, cos, sin):
        """
        Applies rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Cosine values for rotary positional embedding.
            sin (torch.Tensor): Sine values for rotary positional embedding.

        Returns:
            torch.Tensor: Tensor after applying rotary positional embedding.
        """
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[:, :, : x.shape[self.seq_length_dim], :]
        sin = sin[:, :, : x.shape[self.seq_length_dim], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RotaryTransform module.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Tuple containing the modified query tensor, key tensor, and value tensor.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        q = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return q, k, v


class QueryKeyValueTransformType(Enum):
    """
    Enum class representing different types of query-key-value transform.

    Attributes:
        IdentityTransform: Represents the identity transform.
        RotaryTransform: Represents the rotary transform.
    """

    IdentityTransform = IdentityTransform
    RotaryTransform = RotaryTransform


class AttentionImplementation(str, Enum):
    """
    Enum class representing different implementations of attention.

    Attributes:
        MANUAL (str): Manual attention implementation.
        PYTORCH_FLASH (str): PyTorch's flash attention implementation.
        DAO_FLASH (str): DAO's flash attention implementation.
    """

    MANUAL = "manual"
    PYTORCH_FLASH = "pytorch_flash"
    DAO_FLASH = "dao_flash"


class AttentionConfig(BaseModel):
    """
    Configuration class for attention mechanism.

    Attributes:
        qkv_transforms (list[QueryKeyValueTransformConfig]): List of configurations for query-key-value transforms.
    """

    class QueryKeyValueTransformConfig(BaseModel):
        """
        Configuration class for QueryKeyValueTransform.

        Attributes:
            type_hint (QueryKeyValueTransformType): The type hint for the transform.
            config (RotaryTransformConfig | IdentityTransformConfig): The configuration for the transform.
        """

        class IdentityTransformConfig(BaseModel):
            """IdentityTransformConfig class."""

            pass

        class RotaryTransformConfig(BaseModel):
            """
            Configuration class for RotaryTransform.

            Attributes:
                n_embd (int): Number of embeddings.
                n_head (int): Number of attention heads.
                seq_length_dim (int): Dimension of the sequence length.
                base_freq (int): Base frequency for RoPE.

            """

            n_embd: Annotated[int, Field(strict=True, ge=0)]
            n_head: Annotated[int, Field(strict=True, ge=0)]
            seq_length_dim: Annotated[int, Field(strict=True)]
            base_freq: Annotated[int, Field(strict=True, ge=10000)]

        @validator("type_hint", pre=True, always=True)
        def parse_sharding_strategy_by_name(cls, name):
            """
            Parses a QueryKeyValueTransform by its name.

            Args:
                name (str): The name of the sharding strategy.

            Returns:
                QueryKeyValueTransformType: The parsed sharding strategy.

            """
            return parse_enum_by_name(name=name, enum_type=QueryKeyValueTransformType)

        type_hint: QueryKeyValueTransformType
        config: RotaryTransformConfig | IdentityTransformConfig

    qkv_transforms: list[QueryKeyValueTransformConfig]


class GPT2LLMConfig(BaseModel):
    """
    Configuration class for GPT2LLM model.

    Args:
        sample_key (str): The key for the samples.
        prediction_key (str): The key for the predictions.
        use_meta_device (bool, optional): Whether to use meta device. Defaults to False.
        poe_type (PositionTypes): The type of position encoding.
        sequence_length (int): The length of the sequence.
        vocab_size (int): The size of the vocabulary.
        n_layer (int): The number of layers.
        n_head_q (int): The number of attention heads for queries.
        n_head_kv (int): The number of attention heads for keys and values.
        n_embd (int): The embedding size.
        ffn_hidden (int): The hidden size of the feed-forward network.
        dropout (float): The dropout rate.
        bias (bool): Whether to use bias in Linears.
        attention_config (AttentionConfig): The attention configuration.
        attention_implementation (AttentionImplementation): The attention implementation.
        activation_type (ActivationType): The activation type.
        attention_norm_config (LayerNormWrapperConfig): Config for normalization of the attention.
        ffn_norm_config (LayerNormWrapperConfig): Config for normalization of the feed-forward network.
        lm_head_norm_config (LayerNormWrapperConfig): Config for normalization of the language model head.
        use_weight_tying (bool): Whether to use weight tying.

    """

    sample_key: str
    prediction_key: str
    use_meta_device: Optional[bool] = False
    poe_type: PositionTypes
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears like GPT-2. False: a bit better and faster
    attention_config: AttentionConfig
    attention_implementation: AttentionImplementation
    activation_type: ActivationType
    attention_norm_config: LayerNormWrapperConfig
    ffn_norm_config: LayerNormWrapperConfig
    lm_head_norm_config: LayerNormWrapperConfig
    use_weight_tying: bool

    @model_validator(mode="after")
    def check_divisibility(self) -> "GPT2LLMConfig":
        """
        Check if the value of n_head_q is divisible by n_head_kv.

        Raises:
            ValueError: If n_head_q is not divisible by n_head_kv.

        Returns:
            GPT2LLMConfig: The current instance of GPT2LLMConfig.
        """
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError("n_head_q must be divisible by n_head_kv")
        return self

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        """
        Validates the sizes of the GPT2 model parameters.

        Returns:
            GPT2LLMConfig: The current instance of GPT2LLMConfig object.

        Raises:
            ValueError: If any of the parameters (ffn_hidden, vocab_size, n_embd) is not divisible by 128.
        """
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                # See https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


class CausalSelfAttention(nn.Module):
    """Causal Self Attention class."""

    def __init__(
        self,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        attention_config: AttentionConfig,
        attention_impl: AttentionImplementation,
        bias: bool,
        dropout: float,
    ):
        """
        Initializes the CausalSelfAttention object.

        Args:
            n_head_q (int): Number of attention heads for queries.
            n_head_kv (int): Number of attention heads for keys and values.
            n_embd (int): Size of the embedding dimension.
            attention_config (AttentionConfig): The attention configuration.
            attention_impl (AttentionImplementation): The attention implementation.
            bias (bool): Whether to include bias in linear layers.
            dropout (float): Dropout rate.

        Returns:
            None
        """
        super().__init__()
        assert n_embd % n_head_q == 0, "`n_embd needs` to be divisible by `n_head_q`."
        assert n_head_q % n_head_kv == 0, "`n_head_q needs` to be divisible by `n_head_kv`."

        self.n_rep = n_head_q // n_head_kv
        self.attention_impl = attention_impl

        # query, key, value projections (separate)
        self.q_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )
        self.k_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )
        self.v_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )

        # regularization
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv

        self.n_embd = n_embd
        # TODO: we might want different values for attention_dropout and linear_dropout
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(self.dropout)

        # TODO: inject QKVTransforms from outside
        self.qkv_transforms = nn.ModuleList(
            transform_config.type_hint.value(
                **convert_base_model_config_to_dict(transform_config.config)
            )  # TODO refactor, still uses the legacy type_hint
            for transform_config in attention_config.qkv_transforms
        )

    def projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies projections to the input tensor to get queries, keys, and values.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the query, key, and value tensors.
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        return self.q_attn(x), self.k_attn(x), self.v_attn(x)

    @staticmethod
    def execute_qkv_transforms(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qkv_transforms: nn.ModuleList, n_head_q: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a series of transformations to the query, key, and value tensors.

        Args:
            q (torch.Tensor): The query tensors.
            k (torch.Tensor): The key tensors
            v (torch.Tensor): The value tensors.
            qkv_transforms (nn.ModuleList): A list of transformation modules to be applied to q, k, and v.
            n_head_q (int): The number of heads for the query tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the transformed query, key, and value tensors.
        """
        batch_size, sequence_length, embedding_dim = q.size()
        # hidden dimension of single head
        # Note, that number of heads does not change the overall parameters of the networks
        # to scale up the network we either have to increase the embedding_dim or the number of layers
        n_head_dim = embedding_dim // n_head_q

        q = q.view(batch_size, sequence_length, n_head_q, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_q, T, hd)
        k = k.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)
        v = v.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)

        for transform in qkv_transforms:
            q, k, v = transform(q, k, v)

        return q, k, v

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat the key-value tensor along the second dimension.

        Args:
            x (torch.Tensor): The input tensor of shape (B, nh_kv, T, hs).
            n_rep (int): The number of times to repeat the tensor along the second dimension.

        Returns:
            torch.Tensor: The repeated tensor of shape (B, nh_kv * n_rep, T, hs).

        Note:
            Source code adopted from
            https://github.com/facebookresearch/llama/blob/9a001c7a0987afd7b8de94e538916eff8950a73a/llama/model.py#L164
            Adapted ordered dimensions and namings: bs=B, n_kv_heads=nh_kv, slen=T, head_dim=hs
        """
        B, nh_kv, T, hs = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    @classmethod
    def repeat_kv_heads(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Repeats the key-value (k, v) heads if the number of query (q) heads is different.

        Args:
            cls (class): The class object.
            q (torch.Tensor): The query tensor of shape (B, nh_q, T, hs).
            k (torch.Tensor): The key tensor of shape (B, nh_kv, T, hs).
            v (torch.Tensor): The value tensor of shape (B, nh_kv, T, hs).

        Returns:
            tuple: A tuple containing the repeated key tensor (k) and the repeated value tensor (v).
        """
        # repeat k/v heads if self.n_rep > 1
        n_head_q = q.shape[1]
        n_head_kv = k.shape[1]
        if n_head_q != n_head_kv:
            n_rep = n_head_q // n_head_kv
            k = cls._repeat_kv(k, n_rep)  # (B, nh_q, T, hs)
            v = cls._repeat_kv(v, n_rep)  # (B, nh_q, T, hs)
        return k, v

    @classmethod
    def execute_attention(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: float,
        attention_impl: AttentionImplementation,
    ) -> torch.Tensor:
        """
        Executes attention mechanism based on the specified implementation.

        Args:
            cls (object): The class object.
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            dropout (float): The dropout rate.
            attention_impl (AttentionImplementation): The attention implementation to use.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            NotImplementedError: If the specified attention implementation is not supported.
        """
        if attention_impl == AttentionImplementation.MANUAL:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = manual_scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.PYTORCH_FLASH:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.DAO_FLASH:
            raise NotImplementedError
            # # Due to the lack of GPUs in github actions and the requirement of those in the flash-attn library,
            # # we have to check if the library is installed and raise an error if not.
            # # Note, that the library is not required for the CPU-only tests.
            # if flash_attn_func is None:
            #     raise NotImplementedError("ERROR! Dao Flash Attention is not installed.")
            # # the next three lines are only needed for flash-attn from Daio Lab
            # q = q.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
            # k = k.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            # v = v.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            # y = flash_attn_func(
            #     q, k, v, dropout_p=dropout, causal=True, softmax_scale=None, window_size=(-1, -1)
            # )  # (B, T, nh_q, hd)
        else:
            raise NotImplementedError(f"Attention implementation {attention_impl} not supported")
        return y  # (B, T, nh_q, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CausalSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embd), representing the output projection.
        """
        B, T, _ = x.size()  # batch size (B), sequence length (T), embedding dimensionality (self.n_embd)
        q, k, v = self.projection(x)  # q: (B, T, n_embd), k: (B, T, n_embd // n_rep), v: (B, T, n_embd // n_rep)

        # q: (B, nh_q, T, hd), k: (B, nh_kv, T, hd), v: (B, nh_kv, T, hd)
        q, k, v = CausalSelfAttention.execute_qkv_transforms(q, k, v, self.qkv_transforms, self.n_head_q)
        y = CausalSelfAttention.execute_attention(q, k, v, self.dropout, self.attention_impl)  # (B, T, nh_q, hd)
        # (B, T, n_embd), re-assemble all head outputs side by side
        # Note that, we set n_embd to -1, as we shard on that dimension when using tensor parallelism
        # in which case the size is n_embd // tp degree
        y = y.reshape(B, T, -1)
        return self.resid_dropout(self.c_proj(y))  # (B, T, n_embd), output projection


class TransformerMLP(nn.Module):
    """TransformerMLP class."""

    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        """
        Initializes the TransformerMLP class.

        Args:
            n_embd (int): The size of the input embedding.
            ffn_hidden (int): The size of the hidden layer in the feed-forward network.
            bias (bool): Whether to include bias terms in the linear layers.
            dropout (float): The dropout probability.

        Returns:
            None
        """
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=n_embd,
            out_features=ffn_hidden,  # best practice: 4 * n_embd,
            bias=bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=ffn_hidden,
            out_features=n_embd,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerMLP module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """GPT2Block class."""

    def __init__(
        self,
        n_embd: int,
        bias: bool,
        n_head_q: int,
        n_head_kv: int,
        activation_type: ActivationType,
        attention_impl: AttentionImplementation,
        attention_config: AttentionConfig,
        dropout: float,
        ffn_hidden: int,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
    ):
        """
        Initializes the GPT2Block.

        Args:
            n_embd (int): The embedding dimension.
            bias (bool): Whether to include bias in the model.
            n_head_q (int): The number of attention heads for queries.
            n_head_kv (int): The number of attention heads for keys and values.
            activation_type (ActivationType): The type of activation function to use.
            attention_impl (AttentionImplementation): The implementation of attention mechanism.
            attention_config (AttentionConfig): The configuration for attention mechanism.
            dropout (float): The dropout rate.
            ffn_hidden (int): The size of the hidden layer in the feed-forward network.
            attention_norm (nn.Module): The normalization layer for attention.
            ffn_norm (nn.Module): The normalization layer for feed-forward network.
        """
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self._check_ffn_hidden_dim(n_embd=n_embd, ffn_hidden=ffn_hidden)
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_config=attention_config,
            attention_impl=attention_impl,
            bias=bias,
            dropout=dropout,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias)
        else:
            raise NotImplementedError("unimplemented activation")

    def _check_ffn_hidden_dim(self, n_embd: int, ffn_hidden: int) -> None:
        expected_hidden_dim = 4 * n_embd

        if ffn_hidden != expected_hidden_dim:
            logger.warning(
                f"Expected `ffn_hidden` to be 4 * `n_embd` ({expected_hidden_dim}), "
                f"but got `n_embd = {n_embd}` and `ffn_hidden = {ffn_hidden}`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT2Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x + self.attn(self.attention_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT2LLM(NNModel):
    """GPT2LLM class."""

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        sequence_length: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        attention_implementation: AttentionImplementation,
        attention_config: AttentionConfig,
        attention_norm_config: LayerNormWrapperConfig,
        ffn_norm_config: LayerNormWrapperConfig,
        lm_head_norm_config: LayerNormWrapperConfig,
        use_weight_tying: bool,
        seed: int = None,
    ):
        """
        Initializes the GPT2LLM object.

        Args:
            sample_key (str): The sample key.
            prediction_key (str): The prediction key.
            poe_type (PositionTypes): The position type.
            sequence_length (int): The sequence length.
            vocab_size (int): The vocabulary size.
            n_layer (int): The number of layers.
            n_head_q (int): The number of query heads.
            n_head_kv (int): The number of key-value heads.
            n_embd (int): The embedding dimension.
            ffn_hidden (int): The hidden dimension of the feed-forward network.
            dropout (float): The dropout rate.
            bias (bool): Whether to include bias in linear layers.
            activation_type (ActivationType): The activation type.
            attention_implementation (AttentionImplementation): The attention implementation.
            attention_config (AttentionConfig): The attention configuration.
            attention_norm_config (LayerNormWrapperConfig): Config for the attention normalization module.
            ffn_norm_config (LayerNormWrapperConfig): Config for the feed-forward network normalization module.
            lm_head_norm_config (LayerNormWrapperConfig): Config for the language model head normalization module.
            seed (int, optional): The random seed. Defaults to None.
            use_weight_tying (bool): Whether to use weight tying.
        """
        weight_decay_groups = {
            "linear": [".attn", ".mlp", ".lm_head.weight"],
            "embedding": [".wte", ".wpe"],
            "layernorm": [".attention_norm", ".ffn_norm", ".lm_head_norm"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.poe_type = poe_type

        assert vocab_size is not None
        assert sequence_length is not None

        # TODO: dependency injection
        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=sequence_length, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            # Using a pre-trained layer, requires to define a separate FSDP unit for the frozen layer c.f.
            # https://github.com/huggingface/accelerate/issues/807
            # wpe = nn.Embedding.from_pretrained(torch.zeros(sequence_length, n_embd))
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention_config.qkv_transforms
        ]:
            raise ValueError('It is expected to use "RotaryTransform" together with "NOPE".')

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=wpe,
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        GPT2Block(
                            n_embd=n_embd,
                            bias=bias,
                            n_head_q=n_head_q,
                            n_head_kv=n_head_kv,
                            activation_type=activation_type,
                            attention_impl=attention_implementation,
                            attention_config=attention_config,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                            # deepcopy did not work here! The weights were then automatically
                            # moved to a cuda device even when the deepcopied weights were on
                            # a meta device!
                            attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                            ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                        )
                        for _ in range(n_layer)
                    ]
                ),
                lm_head_norm=lm_head_norm_config.norm_type.value(**dict(lm_head_norm_config.config)),
                # NOTE: If we make the bias configurable, we must update the number of parameters calculation
                # in the test_initialization_fsdp1.py, accordingly.
                lm_head=nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False),
            )
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if use_weight_tying:
            self.transformer.wte.weight = (
                self.transformer.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying


    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        # TODO verify correctness of initialization
        def _init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        if buffer_device is not None:
            self.to(buffer_device)

        self.apply(_init_fn)


    def forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation of the GPT2LLM module.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.
                - sample_key (str): Key for the input tensor containing token ids.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing output tensors.
                - prediction_key (str): Key for the output tensor containing logits.
        """
        input_ids = inputs # = inputs[self.sample_key]
        device = input_ids.device
        _, t = input_ids.size()  # batch size, sequence length
        assert t <= self.sequence_length, f"Cannot forward sequence of length {t}, the model's maximum "
        f"input sequence length is only {self.sequence_length}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        if self.poe_type is PositionTypes.ABSOLUTE:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            tok_emb = tok_emb + pos_emb

        # TODO: use drop out also without absolute position embedding?
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.lm_head_norm(x)
        logits = self.transformer.lm_head(x)
        return logits # {self.prediction_key: logits}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_tensor = self.forward_impl(inputs)
        return output_tensor


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """
    Compute scaled dot product attention.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, num_queries, query_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_keys, key_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_values, value_dim).
        attn_mask (torch.Tensor, optional): The attention mask tensor of shape (num_queries, num_keys).
            Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether the attention is causal or not. Defaults to False.
        scale (float, optional): The scaling factor. Defaults to None.

    Returns:
        torch.Tensor: The attention weights tensor of shape (batch_size, num_queries, num_keys).

    Note:
        Taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(
        L, S, dtype=query.dtype, device=query.device
    )  # device added (not part of the original code)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)  # device added
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
