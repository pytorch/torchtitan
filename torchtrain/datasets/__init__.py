from torchtrain.datasets.alpaca import AlpacaDataset
from torchtrain.datasets.tokenizer import create_tokenizer
from torchtrain.datasets.pad_batch_sequence import pad_batch_to_longest_seq


dataset_cls_map = {
    "alpaca": AlpacaDataset,
}
