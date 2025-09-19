"""
Qwen3 registration for TorchTitan training.
"""

from torchtitan.components.loss import build_cross_entropy_loss, build_sft_with_moe_aux_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from .infra.parallelize import parallelize_qwen3
from .model.args import Qwen3TransformerModelArgs, SupportedLossFunctionIdentifiers
from .model.model import Transformer
from .model.state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3TransformerModelArgs",
    "Transformer",
    "qwen3_configs",
]



# Preset flavors
qwen3_configs = {
    "debugmodel": Qwen3TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=151671, # original Qwen has 151936 tokens, but our surgery changed it.
        max_seq_len=2048,
        norm_eps=1e-6,
        intermediate_size=1024,
        rope_theta=1_000_000.0,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
    ),
    "235B": Qwen3TransformerModelArgs(
        dim=4096,
        n_layers=94,
        n_heads=64,
        n_kv_heads=4,
        vocab_size=151671, # original Qwen has 151936 tokens, but our surgery changed it.
        max_seq_len=40960,
        norm_eps=1e-6,
        intermediate_size=12288,
        rope_theta=1_000_000.0,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
    ),
    "30B": Qwen3TransformerModelArgs(
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,
        vocab_size=151671, # original Qwen has 151936 tokens, but our surgery changed it.
        max_seq_len=40960,
        norm_eps=1e-6,
        intermediate_size=6144,
        rope_theta=1_000_000.0,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
    ),
}

def build_qwen_loss_function(job_config: JobConfig):
    if job_config.model.loss_function_identifier == SupportedLossFunctionIdentifiers.SFT_WITH_MOE_AUX_LOSS.value:
        num_experts = qwen3_configs[job_config.model.flavor].num_experts
        num_experts_per_tok = qwen3_configs[job_config.model.flavor].num_experts_per_tok
        return build_sft_with_moe_aux_loss(job_config, top_k=num_experts_per_tok, num_experts=num_experts)
    elif job_config.model.loss_function_identifier == SupportedLossFunctionIdentifiers.SFT.value:
        return build_cross_entropy_loss(job_config)
    else:
        raise ValueError(f"Unknown loss function identifier: {job_config.model.loss_function_identifier}")

register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=Transformer,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_qwen_loss_function,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
)
