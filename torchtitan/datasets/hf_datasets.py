from typing import List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from torchtitan.datasets.tokenizer import TokenizerIf
from torchtitan.logging_utils import logger

from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node

_supported_datasets = {
    "alpaca": "tatsu-lab/alpaca",
    "minipile": "JeanKaddour/minipile",
    "c4": "allenai/c4",
    "openwebtext": "Skylion007/openwebtext",
}


class HuggingFaceDataset(IterableDataset):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]): Path to the dataset in the file system. If provided, data will be loaded from this path instead of downloaded.
        tokenizer (TokenizerIf): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support four datasets:
    alpaca (52K training entries)
    minipile (1M training entries, amalgamated from other datasets)
    openwebtext (1M training entries, same type of data for entire dataset)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> Alpaca <<:
    Data input format (alpaca):
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    >> MiniPile <<:
    MiniPile dataset is detailed in the paper: https://arxiv.org/abs/2304.08442
    Data input format (minipile):
    {
        "text": "Open-end spinning devices with such rotor bearing arrangements are known in
                various different embodiments, and have been extensively described,
                for example in German Patent Publications"
    }

     >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    >> OpenWebText <<:
    OpenWeb crawl, English
    Example:
    {
        'text': "Amazon has launched a new cheaper version of its Echo Dot voice-controlled device today.
    The launch comes six months after Amazon first introduced two new Echo devices —
    one of which was the $90 Echo Dot,..."
    }

    Example use (alpaca):
    >>> alpaca_ds = HuggingFaceDataset(dataset_name="alpaca", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8


    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: TokenizerIf,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        if dataset_name not in _supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. Supported datasets are: {_supported_datasets.keys()}."
            )

        # TODO: This is a temporary solution for small datasets like Alpaca.
        #       For larger datasets we need to use a more scalable approach.
        if dataset_path:
            logger.info(f"Loading {dataset_name} dataset locally from {dataset_path}")
            ds = load_from_disk(dataset_path)
        else:
            logger.info(f"Preparing {dataset_name} dataset from HuggingFace")
            # Setting `streaming=True` works for large dataset, but is slightly slower and unstable.
            # c4 is huge, and requires both streaming and language selection (we default to en)
            if dataset_name == "c4":
                ds = load_dataset(
                    _supported_datasets[dataset_name],
                    "en",
                    split="train",
                    streaming=True,
                )
            else:
                ds = load_dataset(_supported_datasets[dataset_name], split="train")

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        while True:
            for sample in iter(self._data):
                sample_text = sample["text"]
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                all_tokens.extend(sample_tokens)

                while len(all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    all_tokens = all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label
            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: TokenizerIf,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )

    return DataLoader(hf_ds, batch_size=batch_size)
