# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import excluded_parameters_for_model_only
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import logger
from transformers import AutoModelForCausalLM

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

        outputs = torch.stack(outputs.logits)
        outputs_list.append(outputs)

    del model
    torch.cuda.empty_cache()

    return outputs_list


@torch.no_grad
def forward_tt(config_path, checkpoint_path, test_set):

    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    train_spec = get_train_spec(config.model.name)

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
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # obtains the logits of only the last token in the predictions
        predictions = model(input_ids)[:, -1, :].unsqueeze(1)
        output_list.append(predictions)

    del model
    torch.cuda.empty_cache()

    return output_list


if __name__ == "__main__":
    # hf params
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    hf_model_path = "outputs/checkpoint/step-0-tohf"
    hf_model_path_no_perm = "outputs/checkpoint/step-0-tohfnoperm"

    # tt params
    config_path = "torchtitan/models/llama3/train_configs/llama3_8b.toml"
    baseline_checkpoint_path = "outputs/checkpoint/step-0-fromllama"
    checkpoint_path = "outputs/checkpoint/step-0-fromhf"
    checkpoint_path_no_perm = "outputs/checkpoint/step-0-fromhfnoperm"

    # test params
    prompt_len = 8
    test_size = 100

    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])
    train_spec = get_train_spec(config.model.name)
    tokenizer = train_spec.build_tokenizer_fn(config)

    # Build test set of randomly generated token ids
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

    # baseline logits
    baseline_hf_outputs = forward_hf(hf_model_name, None, test_set)
    baseline_tt_outputs = forward_tt(config_path, baseline_checkpoint_path, test_set)

    # testing from hf script
    from_hf_outputs = forward_tt(config_path, checkpoint_path, test_set)
    from_hf_outputs_no_perm = forward_tt(config_path, checkpoint_path_no_perm, test_set)

    # testing to hf script
    to_hf_outputs = forward_hf(hf_model_name, hf_model_path, test_set)
    to_hf_outputs_no_perm = forward_hf(hf_model_name, hf_model_path_no_perm, test_set)

    # Define the set of outputs to test loss for
    test_configs = {
        "from_hf": [baseline_hf_outputs, from_hf_outputs],
        "to_hf": [to_hf_outputs, baseline_tt_outputs],
        "from_hf_no_perm": [baseline_hf_outputs, from_hf_outputs_no_perm],
        "to_hf_no_perm": [to_hf_outputs_no_perm, baseline_tt_outputs],
    }
    avg_losses = {}

    for test_name, (hf_outputs, tt_outputs) in test_configs.items():
        total_loss = 0
        for hf, tt in zip(hf_outputs, tt_outputs):
            total_loss += loss_fn(hf, tt)
        avg_loss = total_loss / len(test_set)
        avg_losses[test_name] = avg_loss.item()

    for test_name, avg_loss in avg_losses.items():
        print(f"Average loss of test {test_name} is {avg_loss}")
