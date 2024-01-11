# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# copied and adjusted from torchtune

from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtrain.datasets.tokenizer import TokenizerIf


_CROSS_ENTROPY_IGNORE_IDX = -100


class AlpacaDataset(Dataset):
    """PyTorch Representation of the Alpaca Dataset from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.

    Data input format:
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    Example:
    >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(self, tokenizer: TokenizerIf, **kwargs) -> None:
        self._data = load_dataset("tatsu-lab/alpaca", split="train")
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self._transform(self._data[index]["text"])

    def _transform(self, sample: str) -> Tuple[List[int], List[int]]:
        """Split a sample on 'response' tag to create input and labels.

        Args:
            sample (str): Sample text.

        Returns:
            Tuple of encoded inputs and labels.
        """
        response_tag = "\n\n### Response:\n"
        inst_inp_response_tag = sample[: sample.index(response_tag) + len(response_tag)]
        response = sample[sample.index(response_tag) + len(response_tag) :]
        inst_inp_response_tag = self._tokenizer.encode(
            inst_inp_response_tag, bos=True, eos=False
        )
        response = self._tokenizer.encode(response, bos=False, eos=True)
        input = inst_inp_response_tag + response
        label = [
            _CROSS_ENTROPY_IGNORE_IDX for _ in range(len(inst_inp_response_tag))
        ] + response
        assert len(input) == len(label)
        return input, label
