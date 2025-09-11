# torchtitan_ext/datasets/blendcorpus_builder.py
import math
from types import SimpleNamespace
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader

# BlendCorpus public API (from README)
# from blendcorpus import (
#     # get_config as bc_get_config,
#     # set_config as bc_set_config,
#     mpu as bc_mpu,
#     build_gpt_datasets,
#     build_pretraining_data_loader,
# )
#
from blendcorpus import parallel_state as bc_mpu
from blendcorpus.data.gpt_dataset import build_gpt_datasets
from blendcorpus.data.data_samplers import build_pretraining_data_loader
# from blendcorpus.data.gpt_dataset import build_gpt_datasets, build_pretraining_data_loader
#
# from .data.data_samplers import build_pretraining_data_loader
# from blendcorpus.data.gpt_dataset import build_gpt_datasets

# from blendcorpus import parallel_state as mpu
# from blendcorpus.data.gpt_dataset import build_gpt_datasets
# from blendcorpus.data.data_samplers import build_pretraining_data_loader
# from blendcorpus.data.config import get_config, set_config
# from blendcorpus.tokenizer import build_tokenizer
from blendcorpus.data.config import get_config as bc_get_config
from blendcorpus.data.config import set_config as bc_set_config
from blendcorpus.utils import get_ltor_masks_and_position_ids as bc_get_masks
def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def _shift_tokens_to_labels(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # tokens: [B, T]
    input_ids = tokens[:, :-1].contiguous()
    labels    = tokens[:, 1:].contiguous()
    return input_ids, labels


def _maybe_attention_mask(
    input_ids: torch.Tensor,
    eod_token_id: Optional[int],
    reset_pos_ids: bool,
    reset_attn_mask: bool,
    eod_mask_loss: bool,
) -> torch.Tensor:
    # You can return None if TorchTitan model builds its own causal mask internally.
    # If you prefer to supply one, BlendCorpus utility builds LTR masks for Megatron-style flows.
    attn_mask, _, _ = bc_get_masks(
        input_ids,
        eod_token_id if eod_token_id is not None else -1,
        reset_pos_ids,
        reset_attn_mask,
        eod_mask_loss,
    )
    return attn_mask  # [B, 1, T, T] causal mask

# --- put this at module scope (e.g., above build_blendcorpus_dataloader) ---
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader

class AdapterDL:
    """
    Lightweight wrapper that:
      - exposes __len__/__iter__ for TorchTitan,
      - can be re-pointed after restore via set_consumed_by_global_step,
      - is picklable (top-level class),
      - can optionally checkpoint only a small state dict.
    """
    def __init__(self, dl: DataLoader, *, ds, bc_cfg):
        self.dl = dl
        self._ds = ds
        self._bc_cfg = bc_cfg
        self._consumed_samples: int = 0  # best-effort, updated when we "retarget"
        try:
            self._len = len(dl)  # type: ignore[arg-type]
        except TypeError:
            self._len = int(1e12)

    def __len__(self):
        return self._len

    @staticmethod
    def _adapt_iter(it):
        for batch in it:
            # BC batch: {"dataset_idx": [B], "text": Long[B, T]}
            tokens = batch["text"].long()
            input_ids = tokens[:, :-1].contiguous()
            labels    = tokens[:,  1:].contiguous()
            # Many TorchTitan paths expect dict-like batches; keep both fields together.
            yield {"input": input_ids}, labels

    def __iter__(self):
        return self._adapt_iter(iter(self.dl))

    def set_consumed_by_global_step(self, global_step: int, global_batch_size: int):
        """Rebuild underlying BC loader to reflect samples already consumed."""
        consumed = int(global_step) * int(global_batch_size)
        self._consumed_samples = consumed
        self.dl = build_pretraining_data_loader(self._ds, consumed, self._bc_cfg)

    # --- Optional: minimal state checkpointing ---
    def state_dict(self) -> Dict[str, int]:
        return {"consumed_samples": int(self._consumed_samples)}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        cs = int(state.get("consumed_samples", 0))
        if cs != self._consumed_samples:
            self._consumed_samples = cs
            self.dl = build_pretraining_data_loader(self._ds, cs, self._bc_cfg)
# --- end top-level class ---
def build_blendcorpus_dataloader(cfg, global_batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Map TorchTitan config → BlendCorpus config
    cfg.blendcorpus.seq_length = getattr(cfg.training, "seq_len")
    cfg.blendcorpus.data_file_list = getattr(cfg.training, "dataset_path")
    cfg.blendcorpus.train_iters = int(getattr(cfg.training, "steps"))
    cfg.blendcorpus.micro_batch_size = int(getattr(cfg.training, "local_batch_size"))
    cfg.blendcorpus.global_batch_size = int(global_batch_size)

    cfg.blendcorpus.tensor_model_parallel_size   = cfg.parallelism.tensor_parallel_degree
    cfg.blendcorpus.pipeline_model_parallel_size = cfg.parallelism.pipeline_parallel_degree
    cfg.blendcorpus.sequence_parallel_size       = cfg.parallelism.context_parallel_degree

    bc_mpu.initialize_model_parallel(
        tensor_model_parallel_size=cfg.blendcorpus.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.blendcorpus.pipeline_model_parallel_size,
        sequence_parallel_size=cfg.blendcorpus.sequence_parallel_size,
    )

    bc_set_config(cfg.blendcorpus)
    bc_cfg = bc_get_config()

    # Build datasets and loaders
    train_ds, valid_ds, test_ds = build_gpt_datasets(bc_cfg)

    train_loader = build_pretraining_data_loader(train_ds, 0, bc_cfg)
    valid_loader = build_pretraining_data_loader(valid_ds, 0, bc_cfg) if valid_ds is not None else None
    test_loader  = build_pretraining_data_loader(test_ds,  0, bc_cfg) if test_ds  is not None else None

    return (
        AdapterDL(train_loader, ds=train_ds,  bc_cfg=bc_cfg) if train_loader is not None else None,
        AdapterDL(valid_loader, ds=valid_ds,  bc_cfg=bc_cfg) if valid_loader is not None else None,
        AdapterDL(test_loader,  ds=test_ds,   bc_cfg=bc_cfg) if test_loader  is not None else None,
    )
