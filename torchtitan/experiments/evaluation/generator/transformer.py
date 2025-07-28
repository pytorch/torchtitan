# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import time
import argparse
from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, _mask_mod_signature

from torchtitan.experiments.evaluation.llama3.attention import build_attention
from torchtitan.experiments.evaluation.llama3.model import Attention
from torchtitan.experiments.evaluation.generator.cache import KVCache


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True) + 1e-9)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.view(shape[:-1])


def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 512  # Maximum number of tokens to generate
    max_tokens: int = 1024  # Maximum number of tokens that can go through the model
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = True
    reduce_generation_overhead: bool = True
    show_progress: bool = False
    model_type_size: str = None
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    dp_degree: int = 1  # Data parallelism degree, but not used in Generator class.


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model,
        tokenizer,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device
        self.model_type_size = cfg.model_type_size

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,  # packed generation
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.sdpa = build_attention(True, "block_causal")

        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        
        self.prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
                module.sdpa = build_attention(False, "causal")

        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    # From here on some methods for generation
    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=mask,
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(self, prompts, stop_on_eos=True, stop_on_until=True):
        # Add BOS token for Llama models
        prompts = [
            self.tokenizer.encode(p, bos=True, eos=False) for p in prompts
        ]
        # Truncate
        max_seqlen = (
            self.max_tokens
            if not hasattr(self.model, "max_seqlen")
            else self.model.max_seqlen
        )
        max_prompt_len = self.max_prompt_len or min(
            max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths)
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):
                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        # If not done or not stopping on EOS, append the token
                        generated_tokens[seq_id].append(tok)

                        stops_on_eos_cond = stop_on_eos and (tok == self.tokenizer.eos_id)
                        stops_on_until_cond = False
                        if stop_on_until and self.until:
                            current_end_str = self.tokenizer.decode(
                                generated_tokens[seq_id][-self.max_until_size :]
                            )
                            stops_on_until_cond = any(
                                [e in current_end_str for e in self.until]
                            )
                        is_done[seq_id] = stops_on_eos_cond or stops_on_until_cond
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        return generation, loglikelihood, greedy


def main(args):
    # to avoid iterative import errors
    from torchtitan.experiments.evaluation.generator.utils import load_transformer_generator

    generator = load_transformer_generator(
        model_type_size=args.model,
        exp_name=args.exp_name,
        max_seq_len=args.max_tokens,
        args={
            "max_gen_len": args.max_gen_len,
            "max_tokens": args.max_tokens,
            "compile_prefilling": args.compile_prefilling,
            "reduce_generation_overhead": args.reduce_generation_overhead,
            "dtype": args.dtype,
            "model_type_size": args.model,
            "dp_degree": args.dp_degree,
        },
    )

    # Allow multiple prompts
    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(generator.tokenizer.encode(gen, bos=False, eos=False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator wrapper for a TorchTitan Transformer model.")
    parser.add_argument("--model", type=str, required=True, help="Model type and size.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--dtype", type=str, default="bf16", help="Model dtype. Use bf16 in multi-gpu settings.")

    parser.add_argument("--compile_prefilling", action="store_false", help="Enable torch.compile for prefilling")
    parser.add_argument("--reduce_generation_overhead", action="store_false", help="Enable reduce-overhead mode for generation")
    parser.add_argument("--max_gen_len", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens that can go through the model")
    parser.add_argument("--dp_degree", type=int, default=1, help="FSDP degree.")
    
    cli_args = parser.parse_args()
    main(cli_args)