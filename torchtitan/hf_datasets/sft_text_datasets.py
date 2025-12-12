# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# code are heavily borrowed from
# [1] https://github.com/volcengine/verl/blob/main/verl/utils/dataset/multiturn_sft_dataset.py
# [2] https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L35
# [3] https://github.com/volcengine/verl/blob/main/verl/utils/dataset/sft_dataset.py#L33
from dataclasses import field
from functools import partial
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config.job_config import SFTDataConfig
from torchtitan.tools.logging import logger

IGNORE_INDEX = -100


def zero_pad_sequences(
    sequences: List[torch.Tensor],
    side: str = "left",
    value: int = 0,
    stack: bool = False,
) -> torch.Tensor:
    """Pad each tensor in `sequences` to the same length on the requested side."""
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def extract_system_prompt_and_generation(tokenizer):
    """Derive system and generation prompt token chunks from the tokenizer's chat template."""
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def preprocess_data(
    data,
    input_template=None,
    input_key="input",
    output_key=None,
    apply_chat_template=None,
    multiturn=False,
):
    """Normalize dataset rows into prompt/response pairs that respect chat templates."""
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(
                prompt_message, tokenize=False, add_generation_prompt=True
            )
            response = apply_chat_template(
                prompt_message + response_message, tokenize=False
            )[len(prompt) :]
        else:
            prompt = apply_chat_template(
                data[input_key][:-1], tokenize=False, add_generation_prompt=True
            )
            response = apply_chat_template(data[input_key], tokenize=False)[
                len(prompt) :
            ]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(IterableDataset, Stateful):
    """
    Iterable dataset that tokenizes a conversational stream for SFT training.
    """

    def __init__(
        self,
        dataset,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        sft_data_config: SFTDataConfig = field(default_factory=SFTDataConfig()),
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self.infinite = infinite

        self.sft_data_config = sft_data_config

        self.messages_key = sft_data_config.messages_key
        self.prompt_key = sft_data_config.prompt_key
        self.response_key = sft_data_config.response_key
        self.tools_key = sft_data_config.tools_key
        self.thinking_key = sft_data_config.thinking_key
        self.enable_tools = sft_data_config.enable_tools
        self.enable_thinking = sft_data_config.enable_thinking
        self.is_multiturn = sft_data_config.is_multiturn
        self.pad_mode = sft_data_config.pad_mode
        self.greedy_packing = sft_data_config.greedy_packing

        if self.pad_mode == "no_padding":
            self.collate_fn = partial(
                collate_sft_batch, pad_token_id=self.tokenizer.pad_id
            )
        else:
            self.collate_fn = None

        self.truncation = sft_data_config.truncation
        self.max_length = seq_len
        self.apply_chat_template_kwargs = {}

        self.apply_chat_template = sft_data_config.apply_chat_template
        if self.apply_chat_template:
            assert self.tokenizer.backup_hf_tokenizer.chat_template is not None, (
                f"Chat template is not set for the tokenizer {self.tokenizer.tokenizer_path}, "
                f"please set it in the tokenizer config file"
            )
            (
                self.system_prompt,
                self.generation_prompt,
            ) = extract_system_prompt_and_generation(self.tokenizer)
        else:
            self.system_prompt = torch.tensor([], dtype=torch.long)
            self.generation_prompt = torch.tensor([], dtype=torch.long)

        self.buffer_max_length = self.max_length if self.greedy_packing else 1
        # Stateful variables
        self._sample_idx = 0
        self._buffer = self._reset_buffer()

    def _reset_buffer(self):
        """Reset the greedy packing buffer to empty tensors."""
        return {
            "input_ids": [],
            "attention_masks": [],
            "position_ids": [],
            "labels": [],
            "current_len": 0,
        }

    def _get_data_iter(self):
        """Return an iterator over the local partition, resuming for map-style datasets."""
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def load_state_dict(self, state_dict):
        self._buffer = state_dict["buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "dataset" in state_dict
            self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        _state_dict = {"buffer": self._buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["dataset"] = self._data.state_dict()

        return _state_dict

    def _build_messages(self, example: dict):
        """Normalize dataset example into list of turn dictionaries with text content."""
        if self.is_multiturn:
            messages: list = example[self.messages_key]
            for message in messages:
                content = message["content"]
                if not isinstance(content, str):
                    continue
                content_list = []
                segments = [item for item in content if item != ""]
                for segment in segments:
                    content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
            return messages
        else:
            messages = [
                {"role": "user", "content": example[self.prompt_key]},
                {"role": "assistant", "content": example[self.response_key]},
            ]
            return messages

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Tokenize one conversation turn while applying template overrides."""
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        if self.apply_chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [message],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
            inputs = dict(inputs)
            input_ids = inputs.pop("input_ids")[0]
            attention_mask = inputs.pop("attention_mask")[0]
        else:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    [item["text"] for item in content if item["type"] == "text"]
                )

            enc = self.tokenizer.backup_hf_tokenizer(
                content,
                add_special_tokens=False,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask

    def sanity_check(
        self,
        input_ids: torch.Tensor,
        messages: list[dict],
        tools: list[dict],
        enable_thinking: bool,
    ):
        """Ensure concatenated per-turn templates match a single-shot template invocation."""
        if not self.apply_chat_template:
            return
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            ignore_extra_tokens=True,
            enable_thinking=False,
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)

    def _process_one_row(self, row_dict: dict):
        """Convert a dataset row into model-ready tensors with causal labels."""
        messages = self._build_messages(row_dict)
        tools = row_dict[self.tools_key] if self.enable_tools else None
        enable_thinking = row_dict[self.thinking_key] if self.enable_thinking else None

        # 1. tokenize each message
        input_ids, loss_mask, attention_mask = [], [], []
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask = self._process_single_message(
                index=i,
                message=message,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        assert (
            input_ids.shape == loss_mask.shape == attention_mask.shape
        ), f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        target_length = self.max_length + 1
        # Handle sequence length
        if self.pad_mode == "right":
            if sequence_length < target_length:
                # Pad sequences
                pad_token_id = (
                    self.tokenizer.pad_id if self.tokenizer.pad_id is not None else 0
                )
                padded_input_ids = torch.full(
                    (target_length - sequence_length,),
                    pad_token_id,
                    dtype=input_ids.dtype,
                )
                padded_attention_mask = torch.zeros(
                    (target_length - sequence_length,), dtype=attention_mask.dtype
                )
                padded_loss_mask = torch.zeros(
                    (target_length - sequence_length,), dtype=loss_mask.dtype
                )

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(
                    position_ids, (0, target_length - sequence_length), value=0
                )
            elif sequence_length > target_length:
                if self.truncation == "left":
                    input_ids = input_ids[-target_length:]
                    attention_mask = attention_mask[-target_length:]
                    loss_mask = loss_mask[-target_length:]
                    position_ids = position_ids[-target_length:]
                elif self.truncation == "right":
                    input_ids = input_ids[:target_length]
                    attention_mask = attention_mask[:target_length]
                    loss_mask = loss_mask[:target_length]
                    position_ids = position_ids[:target_length]
                elif self.truncation == "error":
                    raise ValueError(
                        f"{sequence_length=} is larger than {target_length=}"
                    )
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

        elif self.pad_mode == "no_padding":
            # truncate if longer than max_length (respect truncation setting)
            if len(input_ids) > target_length:
                input_ids = input_ids[:target_length]
                loss_mask = loss_mask[:target_length]
                position_ids = position_ids[:target_length]
            # In NO_PADDING mode, keep a real attention mask (all ones).
            # Collate will pad it later in `collate_sft_batch`.
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        labels = input_ids[1:].clone()
        labels[loss_mask[1:] == 0] = IGNORE_INDEX
        input_dict = {
            "input": input_ids[:-1],
            "positions": position_ids[:-1],
            "attention_masks": attention_mask[:-1],
        }

        return input_dict, labels

    def _yield_buffer(self):
        """Concatenates the current buffer and returns a batch-ready dict."""
        if not self._buffer["input_ids"]:
            return None, None

        ret_input = {
            "input": torch.cat(self._buffer["input_ids"], dim=0),
            "positions": torch.cat(self._buffer["position_ids"], dim=0),
            "attention_masks": torch.cat(self._buffer["attention_masks"], dim=0),
        }
        ret_labels = torch.cat(self._buffer["labels"], dim=0)

        if self.pad_mode == "no_padding" and self.greedy_packing:
            L = int(ret_input["input"].numel())
            T = int(self.buffer_max_length)  # == seq_len when greedy_packing=True
            if L < T:
                pad_len = T - L
                pad_token_id = (
                    self.tokenizer.pad_id
                    if self.tokenizer.pad_id is not None
                    else self.tokenizer.eos_id
                )

                ret_input["input"] = F.pad(
                    ret_input["input"], (0, pad_len), value=pad_token_id
                )
                ret_input["positions"] = F.pad(
                    ret_input["positions"], (0, pad_len), value=0
                )
                ret_input["attention_masks"] = F.pad(
                    ret_input["attention_masks"], (0, pad_len), value=0
                )
                ret_labels = F.pad(ret_labels, (0, pad_len), value=IGNORE_INDEX)

        return ret_input, ret_labels

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                input_dict, labels = self._process_one_row(sample)
                new_len = input_dict["input"].shape[0]

                if self._buffer["current_len"] + new_len > self.buffer_max_length:
                    # A. Yield current buffer (if it has data)
                    if self._buffer["current_len"] > 0:
                        yield_input, yield_labels = self._yield_buffer()
                        yield yield_input, yield_labels
                    # B. Reset buffer and add the NEW sample
                    self._buffer = self._reset_buffer()
                    self._buffer["input_ids"].append(input_dict["input"])
                    self._buffer["attention_masks"].append(
                        input_dict["attention_masks"]
                    )
                    self._buffer["position_ids"].append(input_dict["positions"])
                    self._buffer["labels"].append(labels)
                    self._buffer["current_len"] = new_len
                else:
                    # C. Append to existing buffer
                    self._buffer["input_ids"].append(input_dict["input"])
                    self._buffer["attention_masks"].append(
                        input_dict["attention_masks"]
                    )
                    self._buffer["position_ids"].append(input_dict["positions"])
                    self._buffer["labels"].append(labels)
                    self._buffer["current_len"] += new_len
                self._sample_idx += 1

            if not self.infinite:
                logger.warning("Dataset has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning("Dataset is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)


def _pad_1d_right(seqs: list[torch.Tensor], value: int) -> torch.Tensor:
    """Right-pad a list of 1D tensors to the same length using `value`."""
    max_len = max(int(x.numel()) for x in seqs)
    bsz = len(seqs)
    out = torch.full((bsz, max_len), value, dtype=seqs[0].dtype, device=seqs[0].device)
    for i, x in enumerate(seqs):
        L = int(x.numel())
        out[i, :L] = x
    return out


def collate_sft_batch(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    *,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Pad a list of samples so all inputs, positions, and attention masks can stack.
    Works for already-fixed-length data or NO_PADDING mode samples.
    """

    inputs_list = [b[0] for b in batch]
    labels_list = [b[1] for b in batch]

    input_seqs = [x["input"] for x in inputs_list]
    pos_seqs = [x["positions"] for x in inputs_list]

    attn_seqs = [x["attention_masks"] for x in inputs_list]

    # Fast path: same length => stack
    same_len = all(int(t.numel()) == int(input_seqs[0].numel()) for t in input_seqs)
    if same_len:
        batch_inputs = {
            "input": torch.stack(input_seqs, dim=0),
            "positions": torch.stack(pos_seqs, dim=0),
            "attention_masks": torch.stack(attn_seqs, dim=0),
        }
        batch_labels = torch.stack(labels_list, dim=0)
        return batch_inputs, batch_labels

    # Variable length => RIGHT pad
    batch_inputs = {
        "input": _pad_1d_right(input_seqs, value=pad_token_id),
        "positions": _pad_1d_right(pos_seqs, value=0),
        "attention_masks": _pad_1d_right(attn_seqs, value=0),
    }
    batch_labels = _pad_1d_right(labels_list, value=IGNORE_INDEX)
    return batch_inputs, batch_labels
