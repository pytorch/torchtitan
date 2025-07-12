from dataclasses import dataclass
from pathlib import Path
from torch import nn
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs

@dataclass
class GPT2LLMModelArgs(BaseModelArgs):
    model_config_path: Path = None


    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        # self.vocab_size = tokenizer.n_words
        # self.max_seq_len = job_config.training.seq_len
        # self.eos_id = tokenizer.eos_id

        # if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
        #     raise ValueError(
        #         "FlexAttention is not compatible with selective AC yet. "
        #         "See https://github.com/pytorch/pytorch/issues/147879"
        #     )

        # if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
        #     raise ValueError(
        #         "FlexAttention is not compatible with CP yet. "
        #         "We are still working on this."
        #     )
        pass

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # nparams = sum(p.numel() for p in model.parameters())
        # nparams_embedding = sum(
        #     sum(p.numel() for p in m.parameters())
        #     for m in model.children()
        #     if isinstance(m, nn.Embedding)
        # )

        # l, h, q, t = (
        #     self.n_layers,
        #     self.n_heads,
        #     self.dim // self.n_heads,
        #     seq_len,
        # )
        # # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # # 2. the flash attention does 1 more matmul recomputation in the backward
        # #    but recomputation should not be counted in calculating MFU           (+0)
        # # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # # 4. we follow the convention and do not account for sparsity in causal attention
        # num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        # return nparams, num_flops_per_token
        return 1, 1
