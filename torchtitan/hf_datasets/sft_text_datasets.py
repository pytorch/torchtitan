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
from typing import Any, Optional

import torch
import torch.nn.functional as F

from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.loss import IGNORE_INDEX

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config.job_config import SFTDataConfig
from torchtitan.tools.logging import logger


def _pad_1d_right(seqs: list[torch.Tensor], value: int) -> torch.Tensor:
    """Right-pad a list of 1D tensors to the same length using `value`."""
    max_len = max(int(x.numel()) for x in seqs)
    bsz = len(seqs)
    out = torch.full((bsz, max_len), value, dtype=seqs[0].dtype, device=seqs[0].device)
    for i, x in enumerate(seqs):
        L = int(x.numel())
        out[i, :L] = x
    return out


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


def _build_messages(
    row_dict: dict,
    is_multiturn: bool = False,
    messages_key: str = "messages",
    prompt_key: str = "prompt",
    response_key: str = "response",
):
    """Normalize dataset example into list of turn dictionaries with text content."""
    if is_multiturn:
        # data should already be in the format of "messages"
        messages: list = row_dict[messages_key]
        return messages
    else:
        # we manually assemble the messages by explicitly
        # setting the "user" and "assistant" roles.
        messages = [
            {"role": "user", "content": row_dict[prompt_key]},
            {"role": "assistant", "content": row_dict[response_key]},
        ]
        return messages


"""
From DataFrame rows to a "sequence"

1) [Get row data]
   Iterate over the DataFrame to obtain a per-row dictionary (row_dict).

2) [Build messages]
   Given row_dict, call `_build_messages` to convert the original data into `messages`.

   Here, `messages` is a list of dicts with at least the keys "role" and "content", e.g.
   [
       {"role": "user", "content": "Hello, how are you?"},
       {"role": "assistant", "content": "I am good, thank you!"}
   ]

   We currently support two ways to parse the original data into `messages`:

   2.1) [QA mode]
        If this is NOT multi-turn mode, we build messages by explicitly assigning "user"
        and "assistant" roles, reading from `prompt_key` and `response_key`:
        [
            {"role": "user", "content": row_dict[prompt_key]},
            {"role": "assistant", "content": row_dict[response_key]},
        ]

   2.2) [Multi-turn mode]
        If this IS multi-turn mode (even if there is only one turn), we read the full
        message list from the column `messages_key`. The expected input format is:
        {
            messages_key: [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I am good, thank you!"},
                {"role": "user", "content": "What is your name?"},
            ]
        }

   2.3) [Not landed yet]
        If your data format is completely different, you can implement your own
        `_build_messages` function. It must return a `messages` list in the same format.

3) [Convert messages to tokens]
   After we have `messages`, we tokenize them by calling `_process_one_row`.

   3.1) If there is only ONE message, we tokenize it via `_process_single_message`:
        - Either tokenize the message["content"] directly, OR
        - Apply the tokenizer's chat template (requires the chat template to be configured).

   3.2) If there are MULTIPLE messages, we call `_process_single_message` for each message.
        There is one special case:
        When `apply_chat_template=True`, each call to `_process_single_message` may add a
        [system prompt]. That can produce:
        [System] [User] [System] [Assistant] [System] [User] [System] [Assistant] ...
        which is not what we want for a single conversation. Therefore, we include special
        handling to remove the unexpected [system prompt] and [generation prompt] added
        from the second message onward.
"""

"""
From one or more "sequences" to a "batch"

We offer two modes to handle padding and truncation when forming a batch.
Padding is needed because sequences may have different lengths, and PyTorch
cannot stack variable-length sequences into a single tensor directly.

Example:
    batch = [
        sequence_1,
        sequence_2,
        sequence_3,
    ]

### Right padding
The simplest approach is to right-pad each sequence to a target length using
a special `<pad_token>`. The batch then looks like:
    batch = [
        [sequence_1 + several <pad_token>],
        [sequence_2 + several <pad_token>],
        [sequence_3 + several <pad_token>],
    ]

After right padding, all sequences have the same length. We set the
`attention_mask` for `<pad_token>` positions to 0 so that attention
implementations such as `varlen_attn` or `flex_attn` can ignore them.

### Greedy packing
Alternatively, we can use greedy packing to reduce padding waste by packing
multiple sequences into a single fixed-length row:
    batch = [
        [sequence_1, sequence_2, sequence_3],
        [sequence_4, sequence_5, <pad_token>],
    ]
    **We need strictly compact padding without no gap between sequences.**

We still right-pad each packed row to the target length with `<pad_token>`.
To prevent cross-document attention, we must also infer (or track) the
boundaries between sequences within each packed row.
"""


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
        self.always_thinking = sft_data_config.always_thinking

        self.is_multiturn = sft_data_config.is_multiturn
        self.pad_mode = sft_data_config.pad_mode
        self.ignore_input_ids_mismatch = sft_data_config.ignore_input_ids_mismatch

        self.truncation = sft_data_config.truncation
        self.max_length = seq_len
        self.apply_chat_template_kwargs = sft_data_config.chat_template_kwargs

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

        if not hasattr(self.tokenizer, "pad_id") or self.tokenizer.pad_id is None:
            self.pad_id = self.tokenizer.eos_id
            self.pad_token = self.tokenizer.eos_token
            logger.warning(
                f"pad_id is not set for the tokenizer {self.tokenizer.tokenizer_path}, "
                f"using eos_id as pad_id and eos_token as pad_token"
            )
        else:
            self.pad_id = self.tokenizer.pad_id
            self.pad_token = self.tokenizer.pad_token
        self.eos_id = self.tokenizer.eos_id

        self.buffer_max_length = self.max_length
        # Stateful variables
        self._sample_idx = 0
        self._buffer = self._reset_buffer()

    def _reset_buffer(self):
        """Reset the greedy packing buffer to empty tensors."""
        return {
            "input_ids": [],
            "position_ids": [],
            "labels": [],
            "segment_lens": [],
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
                return_attention_mask=False,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
            inputs = dict(inputs)
            input_ids = inputs.pop("input_ids")[0]
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
                return_attention_mask=False,
            )
            input_ids = enc["input_ids"][0]

            if message["role"] == "assistant":
                input_ids = torch.cat([input_ids, input_ids.new_tensor([self.eos_id])])

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(input_ids)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(input_ids)

        return input_ids, loss_mask

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
        messages = _build_messages(
            row_dict,
            self.is_multiturn,
            self.messages_key,
            self.prompt_key,
            self.response_key,
        )
        tools = row_dict.get(self.tools_key, None)
        enable_thinking = row_dict.get(self.thinking_key, None) or self.always_thinking
        if isinstance(enable_thinking, str):
            enable_thinking = enable_thinking == "on"
        # row_dict[self.thinking_key] = can be True/False <-> or [on/off]

        # 1. tokenize each message
        input_ids, loss_mask = [], []
        for i, message in enumerate(messages):
            _input_ids, _loss_mask = self._process_single_message(
                index=i,
                message=message,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)

        self.sanity_check(input_ids, messages, tools, enable_thinking)

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        target_length = self.max_length + 1

        # Calculate valid length (unpadded) of the sequence for the model
        # Note: We slice input_ids[:-1] later, so the valid length for training is len - 1
        # If truncated, it is target_length - 1
        valid_seq_len = min(sequence_length, target_length) - 1
        if self.pad_mode == "right_padding":
            if sequence_length < target_length:
                # Pad sequences
                pad_token_id = self.pad_id
                padded_input_ids = torch.full(
                    (target_length - sequence_length,),
                    pad_token_id,
                    dtype=input_ids.dtype,
                )
                padded_loss_mask = torch.zeros(
                    (target_length - sequence_length,), dtype=loss_mask.dtype
                )

                input_ids = torch.cat((input_ids, padded_input_ids))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(
                    position_ids, (0, target_length - sequence_length), value=0
                )
            elif sequence_length > target_length:
                if self.truncation == "left_trunc":
                    input_ids = input_ids[-target_length:]
                    loss_mask = loss_mask[-target_length:]
                    position_ids = position_ids[-target_length:]
                elif self.truncation == "right_trunc":
                    input_ids = input_ids[:target_length]
                    loss_mask = loss_mask[:target_length]
                    position_ids = position_ids[:target_length]
                elif self.truncation == "error":
                    raise ValueError(
                        f"{sequence_length=} is larger than {target_length=}"
                    )
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

        elif self.pad_mode == "greedy_packing":
            # notice the actual packing logic happens in the `_yield_buffer` function.
            # truncate if longer than max_length (respect truncation setting)
            if len(input_ids) > target_length:
                input_ids = input_ids[:target_length]
                loss_mask = loss_mask[:target_length]
                position_ids = position_ids[:target_length]
            # In GREEDY_PACKING mode, keep a real attention mask (all ones).
            # Collate will pad it later in `collate_sft_batch`.
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        labels = input_ids[1:].clone()
        labels[loss_mask[1:] == 0] = IGNORE_INDEX
        input_ids = input_ids[:-1]
        position_ids = position_ids[:-1]

        return input_ids, labels, position_ids, valid_seq_len

    def _greedy_pack_buffer(self):
        if not self._buffer["input_ids"]:
            return None

        # Concatenate buffer
        input_ids = torch.cat(self._buffer["input_ids"], dim=0)
        labels = torch.cat(self._buffer["labels"], dim=0)
        positions = torch.cat(self._buffer["position_ids"], dim=0)

        # Calculate seq_lens and start_indices
        lens = torch.as_tensor(self._buffer["segment_lens"], dtype=torch.int32)
        if lens.numel() == 0:
            start_indices = torch.zeros((0,), dtype=torch.int32)
        else:
            start_indices = torch.cat([lens.new_zeros(1), lens.cumsum(0)[:-1]])

        L = int(input_ids.numel())
        T = int(self.buffer_max_length)
        if L < T:
            pad_len = T - L
            input_ids = F.pad(input_ids, (0, pad_len), value=self.pad_id)
            positions = F.pad(positions, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)

        start_indices = F.pad(start_indices, (0, T - start_indices.numel()), value=-1)
        lens = F.pad(lens, (0, T - lens.numel()), value=-1)

        return {
            "input": input_ids,
            "positions": positions,
            "seq_lens": lens,
            "seq_start_indices": start_indices,
        }, labels

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                input_ids, labels, positions, valid_len = self._process_one_row(sample)
                new_len = input_ids.shape[0]

                if self.pad_mode == "right_padding":
                    # Yield consistent dict structure immediately
                    return_dict = {
                        "input": input_ids,
                        "positions": positions,
                        "seq_lens": torch.tensor([valid_len], dtype=torch.int32),
                        "seq_start_indices": torch.tensor([0], dtype=torch.int32),
                    }
                    yield return_dict, labels
                    self._sample_idx += 1
                    continue

                if self._buffer["current_len"] + new_len > self.buffer_max_length:
                    if self._buffer["current_len"] > 0:
                        yield self._greedy_pack_buffer()

                    self._buffer = self._reset_buffer()
                    self._buffer["input_ids"].append(input_ids)
                    self._buffer["position_ids"].append(positions)
                    self._buffer["labels"].append(labels)
                    self._buffer["segment_lens"].append(valid_len)
                    self._buffer["current_len"] = new_len
                else:
                    self._buffer["input_ids"].append(input_ids)
                    self._buffer["position_ids"].append(positions)
                    self._buffer["labels"].append(labels)
                    self._buffer["segment_lens"].append(valid_len)
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
