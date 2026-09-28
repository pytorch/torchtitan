# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""lm-evaluation-harness backend that scores requests with a torchtitan model.

Requests are packed into fixed-size 1D token buffers in the layout the training
collator produces (documents concatenated, positions reset per document, a
padding tail). Scoring therefore goes through the model's own
``preprocess_inputs`` and attention masking, and the fixed buffer size keeps
flex attention from recompiling for every batch shape.

lm-eval splits requests across data-parallel ranks and gathers the metrics.
Within a data-parallel rank the model may be tensor parallel; its TP ranks
see the same requests, so they issue identical forward passes.
"""

from typing import Any

import torch
import torch.distributed as dist
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model import BaseModel


class TrainstationLM(TemplateLM):
    """Log-likelihood scoring of lm-eval requests with a torchtitan model.

    Args:
        model: Model in eval mode on ``device``, possibly tensor parallel.
        tokenizer: The tokenizer the model was trained with.
        parallel_dims: Parallel dims of the eval job: data-parallel replicas,
            each optionally tensor parallel.
        parallelism: Parallelism config passed through to ``preprocess_inputs``.
        max_context_length: Maximum length of one scored sequence.
        num_tokens_per_batch: Size of each packed token buffer.
        max_num_documents: Cap on sequences per buffer, required by varlen
            attention configs that set ``dataloader.max_num_documents``.
        device: Device the model lives on.
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_context_length: int,
        num_tokens_per_batch: int,
        max_num_documents: int | None = None,
        device: torch.device,
    ) -> None:
        super().__init__()
        if tokenizer.eos_id is None:
            raise ValueError("lm-eval scoring requires a tokenizer with an EOS token.")
        if num_tokens_per_batch < max_context_length:
            raise ValueError(
                f"num_tokens_per_batch ({num_tokens_per_batch}) must be at least "
                f"max_context_length ({max_context_length})."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.parallelism = parallelism
        self.max_context_length = max_context_length
        self.num_tokens_per_batch = num_tokens_per_batch
        self.max_num_documents = max_num_documents
        self._device = device
        self._forward_context = dist_utils.get_spmd_context(parallel_dims=parallel_dims)
        self._dp_group = None
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            self._rank = batch_mesh.get_local_rank()
            self._world_size = batch_mesh.size()
            self._dp_group = batch_mesh.get_group()

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def prefix_token_id(self) -> int:
        # Training documents start with BOS (see hf_datasets/text_datasets.py),
        # so condition empty contexts and rolling windows on BOS as well.
        bos_id = getattr(self.tokenizer, "bos_id", None)
        return bos_id if bos_id is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        return self.max_context_length

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        add_bos = True if add_special_tokens is None else add_special_tokens
        return self.tokenizer.encode(string, add_bos=add_bos, add_eos=False)

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        windows: list[tuple[None, list[int], list[int]]] = []
        owners: list[int] = []
        for index, (string,) in enumerate(req.args for req in requests):
            tokens = self.tok_encode(string, add_special_tokens=False)
            for context, continuation in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=tokens,
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            ):
                windows.append((None, context, continuation))
                owners.append(index)

        totals = [0.0] * len(requests)
        scores = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        for index, (logprob, _) in zip(owners, scores, strict=True):
            totals[index] += logprob
        return totals

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False):
        raise NotImplementedError(
            "TrainstationLM only supports log-likelihood tasks; generative "
            "tasks are not supported yet."
        )

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[Any, list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        # Each request becomes (input tokens, target tokens); the targets are
        # the last len(targets) positions of the input shifted by one.
        sequences: list[tuple[list[int], list[int]]] = []
        for _, context_enc, continuation_enc in requests:
            if len(continuation_enc) > self.max_length:
                raise ValueError(
                    f"Continuation of {len(continuation_enc)} tokens exceeds "
                    f"max_context_length ({self.max_length})."
                )
            # Left-truncate; the last token is only ever a target.
            tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :]
            sequences.append((tokens[:-1], continuation_enc))

        # Longest first so buffers fill tightly.
        order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i][0]))
        results: list[tuple[float, bool] | None] = [None] * len(sequences)
        progress = tqdm(
            total=len(sequences),
            disable=disable_tqdm or self.rank != 0,
            desc="Scoring requests",
        )
        for batch in self._pack(order, sequences):
            for index, result in zip(
                batch, self._score_buffer([sequences[i] for i in batch]), strict=True
            ):
                results[index] = result
            progress.update(len(batch))
        progress.close()
        return results  # pyrefly: ignore [bad-return]

    def _pack(
        self, order: list[int], sequences: list[tuple[list[int], list[int]]]
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        used = 0
        for index in order:
            length = len(sequences[index][0])
            full = used + length > self.num_tokens_per_batch or (
                self.max_num_documents is not None
                and len(current) == self.max_num_documents
            )
            if current and full:
                batches.append(current)
                current, used = [], 0
            current.append(index)
            used += length
        if current:
            batches.append(current)
        return batches

    @torch.no_grad()
    def _score_buffer(
        self, sequences: list[tuple[list[int], list[int]]]
    ) -> list[tuple[float, bool]]:
        size = self.num_tokens_per_batch
        input_ids = torch.zeros(size, dtype=torch.int64)
        positions = torch.zeros(size, dtype=torch.int64)
        padding_mask = torch.ones(size, dtype=torch.bool)

        # Flat indices of the logits that predict each target token.
        target_rows: list[int] = []
        target_ids: list[int] = []
        start = 0
        for inputs, targets in sequences:
            end = start + len(inputs)
            input_ids[start:end] = torch.tensor(inputs)
            positions[start:end] = torch.arange(len(inputs))
            padding_mask[start:end] = False
            target_rows.extend(range(end - len(targets), end))
            target_ids.extend(targets)
            start = end
        if start < size:
            # Same padding layout as TextCollator.
            positions[start:] = torch.arange(size - start) % self.max_context_length

        batch = {
            "input": input_ids,
            "labels": torch.full((size,), IGNORE_INDEX, dtype=torch.int64),
            "positions": positions,
            "padding_mask": padding_mask,
        }
        batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
        with self._forward_context():
            inputs, _, extra_kwargs = self.model.preprocess_inputs(
                batch,
                parallel_dims=self.parallel_dims,
                parallelism=self.parallelism,
                max_num_documents=self.max_num_documents,
                max_context_length=self.max_context_length,
            )
            logits = self.model(inputs, **extra_kwargs)
        if isinstance(logits, DTensor):
            # Under TP the logits may be sharded over the vocabulary.
            logits = logits.full_tensor()

        rows = torch.tensor(target_rows, device=self._device)
        ids = torch.tensor(target_ids, device=self._device)
        logprobs = torch.log_softmax(logits[rows].float(), dim=-1)
        token_logprobs = logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        is_greedy = logprobs.argmax(dim=-1) == ids

        results = []
        offset = 0
        token_logprobs, is_greedy = token_logprobs.tolist(), is_greedy.tolist()
        for _, targets in sequences:
            span = slice(offset, offset + len(targets))
            results.append((sum(token_logprobs[span]), all(is_greedy[span])))
            offset += len(targets)
        return results

    # Distributed primitives used by lm_eval.evaluator.evaluate(), over the
    # data-parallel group (lm-eval's ranks).

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self._dp_group)
        return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered)

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        gathered = [None] * self.world_size if self.rank == dst else None
        dist.gather_object(obj, gathered, group=self._dp_group, group_dst=dst)
        return gathered

    def barrier(self) -> None:
        dist.barrier(group=self._dp_group)
