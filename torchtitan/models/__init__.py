from torchtitan.models.llama import llama_configs, Transformer

models_config = {
    "llama": llama_configs,
}

model_name_to_cls = {
    "llama": Transformer,
}

model_name_to_tokenizer = {
    "llama": "sentencepiece",
}
