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
def forward_hf(model_name, model_path: Optional[str], input_ids):
    # Load the tokenizer and model
    model_path = model_path if model_path else model_name
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device(device_type)
    model.to(device)

    # List to store outputs
    outputs_list = []

    for inputs in input_ids:
        inputs = inputs.to(device)
        outputs = model.generate(
            inputs=inputs,
            max_length=prompt_len + 1,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )

        print("hf inputs")
        print(inputs)
        print("hf outputs")
        outputs = torch.stack(outputs.logits)
        print(outputs.shape)
        print(outputs)
        outputs_list.append(outputs)

    del model
    torch.cuda.empty_cache()

    return outputs_list


if __name__ == "__main__":
    # hf params
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    hf_model_path = "outputs/checkpoint/step-0-tohf"
    hf_model_path_no_perm = "outputs/checkpoint/step-0-tohfnoperm"

    config_path = "torchtitan/models/llama3/train_configs/llama3_8b.toml"

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

    # baseline hf generation
    torch.manual_seed(42)
    hf_outputs = forward_hf(hf_model_name, None, test_set)

    # testing to hf script by running tt model on hf
    torch.manual_seed(42)
    pp_hf_outputs = forward_hf(hf_model_name, hf_model_path, test_set)

    total_loss = 0
    for hf, pp in zip(hf_outputs, pp_hf_outputs):
        total_loss += loss_fn(hf, pp)
    avg_loss = total_loss / len(test_set)

    print("Average loss", avg_loss)
