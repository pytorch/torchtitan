from torchtrain.datasets.alpaca import build_alpaca_data_loader
from torchtrain.datasets.tokenizer import create_tokenizer
from torchtrain.datasets.pad_batch_sequence import pad_batch_to_longest_seq


dataloader_fn = {
    "alpaca": build_alpaca_data_loader,
}
