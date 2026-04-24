
# Currently better use torch.utils.data.ConcatDataset

from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset
from .utils import rank_zero_print

class MultiDataset(Dataset):
    """
    A wrapper datatset, that enables the fusion of multiple dataset sources.
    """
    def __init__(self, datasets, keys=None):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        rank_zero_print(f"Combining datasets {datasets} of lengths: {self.lengths}")
        self.lengths_cum = np.cumsum(self.lengths)
        self.lengths_cum_prev = np.concatenate([[0], self.lengths_cum])
        self.total_len = self.lengths_cum[-1] if len(self.lengths_cum) > 0 else 0
        self.keys = keys
        if keys is not None:
            rank_zero_print(f"MultiDataset keys: {keys} ({type(keys)})")
        self.warnings = defaultdict(lambda: defaultdict(lambda: False))

    def __getitem__(self, idx):
        # Determine which dataset this index belongs to
        key_index = np.searchsorted(self.lengths_cum, idx, side='right')
        sub_idx = idx - self.lengths_cum_prev[key_index]
        # Fetch the item from the appropriate dataset
        item = self.datasets[key_index][sub_idx]
        if self.keys is not None:
            sampled_item = {}
            try:
                for key in self.keys:
                    if key in item:
                        sampled_item[key] = item[key]
                    else:
                        if not self.warnings[key_index][key]:
                            rank_zero_print(f"Dataset {key_index} doesn't have key {key}, "
                                            f"available are only: {list(item.keys())}")
                            self.warnings[key_index][key] = True
            except Exception as e:
                print(f"Exception when querrying sampel {sub_idx} of dataset "+
                      f"{key_index} (returned item of type {type(item)}, should be a Mapping):")
                raise e
            return sampled_item
        #print(f"returning sample with keys: {item.keys()}, {[(k,type(v)) for k,v in item.items()]}", flush=True)
        return item

    def __len__(self):
        return self.total_len

# Can be used e.g. as
# dataset = MultiDataset([ParquetDatasetArrow(parquet_file = download_mons(dataset="tuxemons")),
#                         ParquetDatasetArrow(parquet_file = download_mons(dataset="pokemons"))])