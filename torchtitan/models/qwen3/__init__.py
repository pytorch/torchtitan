from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.dataloader import build_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from ..llama3 import (
    Llama3StateDictAdapter,
    parallelize_llama,
    pipeline_llama,
    Transformer,
    TransformerModelArgs,
)

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "qwen3_configs",
]


qwen3_configs = {
    "0.6B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        norm_eps=1e-6,
        use_qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "1.7B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        use_qk_norm=True,
        hidden_dim=6144,
        norm_eps=1e-6,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "4B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-6,
        use_qk_norm=True,
        hidden_dim=9728,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "8B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-6,
        use_qk_norm=True,
        hidden_dim=12288,
        rope_theta=1000000,
    ),
    "14B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        norm_eps=1e-6,
        use_qk_norm=True,
        hidden_dim=17408,
        rope_theta=1000000,
    ),
    "32B": TransformerModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,
        norm_eps=1e-6,
        use_qk_norm=True,
        hidden_dim=25600,
        rope_theta=1000000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="qwen3",
        model_cls=Transformer,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
