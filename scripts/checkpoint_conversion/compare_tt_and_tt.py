import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import excluded_parameters_for_model_only
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

device_type = "cuda" if torch.cuda.is_available() else "cpu"


def loss_fn(logits1, logits2):
    # Convert logits to probabilities
    probs1 = F.log_softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)

    # Calculate KL Divergence
    kl_loss = F.kl_div(probs1, probs2, "mean")
    return kl_loss


@torch.no_grad
def forward_tt(config_path, checkpoint_path, test_set):

    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    train_spec = get_train_spec(config.model.name)

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)

    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    model = train_spec.model_cls(model_args)

    # materalize model
    device = torch.device(device_type)
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights()
    model.eval()

    state_dict = model.state_dict()
    for k in excluded_parameters_for_model_only:
        state_dict.pop(k, None)

    # Checkpoint Loading
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    load_from_hf = False
    for filename in os.listdir(checkpoint_path):
        if filename == "model.safetensors.index.json":
            load_from_hf = True
    if load_from_hf:
        sd_adapter = train_spec.state_dict_adapter
        hf_state_dict = sd_adapter.to_hf(state_dict)
        dcp.load(hf_state_dict, HuggingFaceStorageReader(path=checkpoint_path))
        state_dict = sd_adapter.from_hf(hf_state_dict)
    else:
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

    output_list = []
    for prompt in test_set:
        input_ids = prompt.to(device_type)
        # ensure batch dimension (T,) --> (B, T)
        # print(input_ids.shape)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # print(input_ids.shape)

        # obtains the logits of only the last token in the predictions
        predictions = model(input_ids)[:, -1, :].unsqueeze(1)

        # print("tt logits")
        # print(predictions.shape)
        # print(predictions)
        output_list.append(predictions)

    del model
    torch.cuda.empty_cache()

    return output_list


if __name__ == "__main__":
    config_path = "torchtitan/models/llama3/train_configs/llama3_8b.toml"

    # TODO change this to corect path
    checkpoint_path_baseline = "outputs/checkpoint-test/step-0-fromhf"
    checkpoint_path_test = "outputs/checkpoint-test/step-1"

    # test params
    prompt_len = 8
    test_size = 100

    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])
    train_spec = get_train_spec(config.model.name)
    tokenizer = train_spec.build_tokenizer_fn(config)

    test_set = [
        torch.randint(
            0,
            tokenizer.get_vocab_size(),
            (
                1,  # batch size
                prompt_len,
            ),
        )
        for _ in range(test_size)
    ]

    torch.manual_seed(42)
    tt_baseline = forward_tt(config_path, checkpoint_path_baseline, test_set)
    tt_test = forward_tt(config_path, checkpoint_path_test, test_set)

    total_loss = 0
    for baseline, test in zip(tt_baseline, tt_test):
        total_loss += loss_fn(baseline, test)
    avg_loss = total_loss / len(test_set)

    print("Average loss", avg_loss)
