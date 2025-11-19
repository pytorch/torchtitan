import importlib

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config import JobConfig
from .packed_memmap import build_memmap_text_dataloader
from .preprocessed import build_preprocessed_dataloader

from .text_datasets import build_text_dataloader

DATALOADERS = {
    "huggingface": build_text_dataloader,
    "preprocessed": build_preprocessed_dataloader,
    "packed_memmap": build_memmap_text_dataloader,
}


def build_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
):
    """Build the appropriate dataloader function"""

    if job_config.training.dataset_type == "nanoset":
        if importlib.util.find_spec("datatrove") is not None:
            from .nanoset import build_nanoset_dataloader

            dataloader = build_nanoset_dataloader
        else:
            raise RuntimeError("Install `datatrove` package for nanoset support")
    elif job_config.training.dataset_type not in DATALOADERS:
        raise ValueError(f"Unknown dataset type `{job_config.training.dataset_type}`")
    else:
        dataloader = DATALOADERS[job_config.training.dataset_type]

    return dataloader(dp_world_size, dp_rank, tokenizer, job_config, infinite)
