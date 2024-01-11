from torchtrain.models.llama.model import ModelArgs, Transformer

llama_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=1, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "70B": ModelArgs(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=4096),
}
