from torchtitan.config.job_config import PEFT


def update_state_dict_adapter(from_hf_map: dict, peft_config: PEFT):
    if (not peft_config.enable_peft) or (not peft_config.use_lora):
        return from_hf_map
    peft_from_hf_map = dict()
    for key, value in from_hf_map.items():
        if "embed_tokens" in key:
            if not peft_config.train_embeddings:
                peft_from_hf_map[key] = value
            else:
                peft_from_hf_map["base_model.model." + key] = value
        if "norm" in key:
            if not peft_config.lora_train_norm:
                peft_from_hf_map[key] = value
            else:
                peft_from_hf_map["base_model.model." + key] = value
        if "lm_head" in key:
            if not peft_config.train_output_layer:
                peft_from_hf_map[key] = value
            else:
                peft_from_hf_map["base_model.model." + key] = value
        else:
            if ".weight" in key:
                # linear, or expert weight
                key = "base_model.model." + key.split(".weight")[0]
                value = value.split(".weight")[0]
                peft_from_hf_map[key + ".lora_A.weight"] = value + ".lora_a"
                peft_from_hf_map[key + ".lora_B.weight"] = value + ".lora_b"

    return peft_from_hf_map


def prune_state_dict_for_peft(state_dict: dict, peft_config: PEFT):
    if not peft_config.enable_peft:
        return state_dict
    if peft_config.layers_to_train is not None:
        for key in list(state_dict.keys()):
            if "layers" in key:
                layer_num = int(key.split(".")[1])
                if layer_num not in peft_config.layers_to_train:
                    del state_dict[key]
    if not peft_config.train_embeddings:
        key = "model.embed_tokens.weight"
        del state_dict[key]
    if not peft_config.train_output_layer:
        key = "lm_head.weight"
        del state_dict[key]
    if (not peft_config.lora_train_norm) and (peft_config.use_lora):
        for key in list(state_dict.keys()):
            if "norm" in key:
                del state_dict[key]
    if peft_config.use_lora:
        for key in list(state_dict.keys()):
            if ("layers" in key) and ("lora" not in key):
                if ("norm" in key) and (peft_config.lora_train_norm):
                    continue
                else:
                    del state_dict[key]
    return state_dict
