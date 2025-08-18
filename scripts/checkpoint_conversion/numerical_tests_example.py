# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torchtitan.components.checkpoint import ModelWrapper
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
    model.init_weights(buffer_device=device)
    model.eval()

    modelWrapper = ModelWrapper(model)
    state_dict = modelWrapper._get_state_dict()

    # Checkpoint Loading
    logger.info(f"Loading checkpoint at: {checkpoint_path}")
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

    # tt params
    config_path = "torchtitan/models/llama3/train_configs/llama3_8b.toml"
    checkpoint_path = "outputs/test_checkpoint/step-0-fromhf"  # dcp checkpoint from convert_from_hf.py
    # dcp checkpoint from convert_from_hf.py without using sd_adapter's permute
    checkpoint_path_no_perm = "outputs/test_checkpoint/step-0-fromhfnoperm"

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

    # testing from hf conversion
    from_hf_outputs = forward_tt(config_path, checkpoint_path, test_set)
    from_hf_outputs_no_perm = forward_tt(config_path, checkpoint_path_no_perm, test_set)

    # Define the set of outputs to test loss for
    test_configs = {
        "from_hf": [baseline_hf_outputs, from_hf_outputs],
        "from_hf_no_perm": [baseline_hf_outputs, from_hf_outputs_no_perm],
    }
    avg_losses = {}

    for test_name, (baseline_outputs, conversion_outputs) in test_configs.items():
        total_loss = 0
        for baseline, outputs in zip(baseline_outputs, conversion_outputs):
            total_loss += loss_fn(baseline, outputs)
        avg_loss = total_loss / len(test_set)
        avg_losses[test_name] = avg_loss.item()

    for test_name, avg_loss in avg_losses.items():
        print(f"Average loss for test {test_name} is {avg_loss}")
