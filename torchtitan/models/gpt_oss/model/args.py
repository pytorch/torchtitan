# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Literal

from torch import nn

from torchtitan.config.job_config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


@dataclass
class GptOssModelArgs(BaseModelArgs):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        norm_eps (float): Epsilon used for RMSNorm.
        moe_args (MoEArgs): Arguments for Mixture of Experts (MoE) layers.
        swiglu_limit (float): SwiGLU activation limit.
        head_dim (int): Dimension of each attention head.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key-value heads.
        sliding_window_size (int): Size of the sliding attention window.
        attn_mask_type (str): Type of basic attention mask.
        attn_type (bool): Attention type, only supports Flex.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        ntk_alpha (float): NTK-by-parts alpha (slow correction, used for high freq).
        ntk_beta (float): NTK-by-parts beta (fast correction, used for low freq).
    """

    max_batch_size: int = 8
    max_seq_len: int = 131072
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 201088
    dim: int = 2880
    moe_inter_dim: int = 2880
    n_layers: int = 24
    norm_eps: float = 1e-5  # eps used for RMSNorm
    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    swiglu_limit: float = 7.0
    # Multi-Head Latent Attention (MLA)
    head_dim: int = 64
    n_heads: int = 64
    n_kv_heads: int = 8
    sliding_window_size: int = 128
    attn_mask_type: str = "causal"
    attn_type: str = "flex"  # NOTE: gpt-oss only support FlexAttention
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 150000.0
    rope_factor: float = 32
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0
    # Expert parallel communication backend: "standard" or "deepep"
    moe_impl: str = "standard"

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.moe_args.use_grouped_mm = False

        if job_config.parallelism.context_parallel_degree > 1:
            raise NotImplementedError(
                "CP support for gpt-oss model is still in progress."
            )

        # Set MoE implementation based on expert parallel comm backend
        self.moe_impl = job_config.parallelism.expert_parallel_comm_backend

        # Validate RoPE parameters against HF checkpoint config if loading from HF
        if (
            job_config.checkpoint.initial_load_in_hf
            and job_config.checkpoint.initial_load_path
        ):
            self._validate_rope_params_from_hf(job_config.checkpoint.initial_load_path)
        elif (
            job_config.checkpoint.initial_load_in_hf and job_config.model.hf_assets_path
        ):
            self._validate_rope_params_from_hf(job_config.model.hf_assets_path)

    def _validate_rope_params_from_hf(self, hf_path: str) -> None:
        """Validate RoPE parameters match HF checkpoint config.json.

        When loading from HF checkpoint, RoPE parameters (ntk_alpha, ntk_beta, rope_factor)
        are NOT loaded from config.json - they use model_args defaults. This function
        validates that the defaults match the checkpoint to avoid silent mismatches.
        """
        import json
        import os

        config_path = os.path.join(hf_path, "config.json")
        if not os.path.exists(config_path):
            logger.warning(
                f"HF config.json not found at {config_path}, skipping RoPE validation"
            )
            return

        with open(config_path, "r") as f:
            hf_config = json.load(f)

        # Check rope_scaling parameters (YaRN)
        rope_scaling = hf_config.get("rope_scaling", {})
        if rope_scaling:
            # Map HF parameter names to new names
            # beta_fast (old) -> ntk_beta (new)
            # beta_slow (old) -> ntk_alpha (new)
            hf_beta_fast = rope_scaling.get("beta_fast")
            hf_beta_slow = rope_scaling.get("beta_slow")
            hf_factor = rope_scaling.get("factor")
            hf_orig_len = rope_scaling.get("original_max_position_embeddings")

            mismatches = []
            if (
                hf_beta_fast is not None
                and abs(float(hf_beta_fast) - self.ntk_beta) > 1e-6
            ):
                mismatches.append(
                    f"beta_fast: HF={hf_beta_fast} vs model_args.ntk_beta={self.ntk_beta}"
                )
            if (
                hf_beta_slow is not None
                and abs(float(hf_beta_slow) - self.ntk_alpha) > 1e-6
            ):
                mismatches.append(
                    f"beta_slow: HF={hf_beta_slow} vs model_args.ntk_alpha={self.ntk_alpha}"
                )
            if (
                hf_factor is not None
                and abs(float(hf_factor) - self.rope_factor) > 1e-6
            ):
                mismatches.append(
                    f"factor: HF={hf_factor} vs model_args.rope_factor={self.rope_factor}"
                )
            if hf_orig_len is not None and hf_orig_len != self.original_seq_len:
                mismatches.append(
                    f"original_max_position_embeddings: HF={hf_orig_len} vs model_args.original_seq_len={self.original_seq_len}"
                )

            if mismatches:
                raise ValueError(
                    f"RoPE parameter mismatch between HF checkpoint and model_args:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                    + "\n\nRoPE parameters are NOT auto-loaded from HF config.json. "
                    + "You must set them in your TOML config or model flavor defaults to match the checkpoint."
                )
            else:
                logger.info(
                    f"[OK] RoPE parameters validated against HF checkpoint: ntk_alpha={self.ntk_alpha}, ntk_beta={self.ntk_beta}, rope_factor={self.rope_factor}"
                )

        # Check rope_theta
        hf_rope_theta = hf_config.get("rope_theta")
        if (
            hf_rope_theta is not None
            and abs(float(hf_rope_theta) - self.rope_theta) > 1e-6
        ):
            raise ValueError(
                f"rope_theta mismatch: HF checkpoint has {hf_rope_theta}, but model_args has {self.rope_theta}. "
                "Set model_args.rope_theta to match the checkpoint."
            )

    # pyrefly: ignore [bad-override]
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)
