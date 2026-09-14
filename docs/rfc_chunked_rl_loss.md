
TLDR: We want to enable chunked CE for RL losses, which need full logprobs. The current `ChunkedCELoss` returns a scalar (already reduced). This RFC discusses the path to support RL losses with chunked CE.

## 1. Problem

Hidden states in a transformer LM have shape `[B, L, H]` (batch, sequence length, hidden dim). The final `lm_head` projects `H` to vocabulary size `V`, producing logits `[B, L, V]`. For modern models `V` is large: Qwen3-0.6B has `V=151936`.

At `L=131072`, `B=1`, fp32:
```python
B, L, V = 1, 131072, 151936
bytes_for_fp32_logits = B * L * V * 4 # ~74 GiB
```

Backward through cross-entropy grows the peak further.

A common mitigation is to chunk along a sequence **before** `lm_head`, compute fwd/bwd and release the memory, so the full memory is never materialized. Another option is to chunk along the vocab dimension and do online softmax `[B, L, V/N]`.

### What does it mean for RL?

TorchTitan's `ChunkedCELoss` does chunking on sequence dim. It returns a scalar loss and the accumulated hidden gradient. It does **not** expose token logprobs.

**This is the motivation for this RFC**: RL losses need the non-reduced logprobs, so they can compare train logprobs vs generator logprobs, i.e. `ratio = exp(train_logprobs - generator_logprobs)`.

One can ask: "Can we also chunk the RL losses?". Yes! However, not all of them. Some RL losses are sequence-wise, i.e. **NOT** token-wise. They take the mean over sequences, and not over tokens, and chunking would break this logic.

## 2. TLDR of the proposed solution

There are two natural ways to expose token logprobs while being memory efficient:

**(a) Execute RL loss on chunk.** Token-wise losses (SFT, GRPO, DAPO, CISPO, SAPO) compute the scalar chunk contribution from per-chunk logprobs, backprop it inline, and discard the chunk graph:

```python
for h_chunk, label_chunk, generator_logprobs_chunk in chunks:
    # In SFT, rl_loss_on_chunk is just `cross_entropy(logits, labels, reduction="sum")`
    chunk_loss = rl_loss_on_chunk(h_chunk, label_chunk, generator_logprobs_chunk)

    chunk_loss.backward()
```

**(b) Checkpoint on chunks (recompute on backward), return full tensor.** Sequence-wise losses (GSPO) need the full `[B, L]` token logprobs before they can compute the objective. Here comes checkpointing:
1. We operate the chunks `[B, L/chunk, V]` normally for CE, but with `requires_grad=False`. We will recompute them later in the backward.
2. Concatenate the `[B, L/chunk]` logprobs slices. Notice that moving forward we don't have the "V" dimension, which is much cheaper.
3. Then compute the RL loss on `[B, L]`.

The implementation must avoid retaining each chunk's `[B, chunk, V]` autograd graph — manual checkpointing replays each chunk's forward during backward instead:

```python
class _CheckpointedChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(...):
        ...
        train_logprobs_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for h_chunk, label_chunk in zip(h_chunks, label_chunks):
                logits = lm_head(h_chunk)
                train_logprobs_chunks.append(
                    -cross_entropy(
                        logits, label_chunk, ignore_index=ignore_index, reduction="none",
                    )
                )
        ...
        # returns full tensor
        return torch.cat(train_logprobs_chunks, dim=1)

    @staticmethod
    def backward(
        ctx, grad_train_logprobs: torch.Tensor,                           # [B, L]
    ) -> tuple:
        """Backward pass — replays each chunk under ``enable_grad``."""
        # ...

def my_rl_loss(...):
    train_logprobs = _CheckpointedChunkedCE.apply(...)
    # rest of the rl loss...
```

```text
Forward, per chunk (under torch.no_grad):
    h_chunk [B, chunk, H]
        |
        v
    lm_head ----> logits [B, chunk, V]          <-- alive briefly
        |             |
        v             v
    token logprobs    [B, chunk]                <-- kept
                      |
                      v
              torch.cat -> [B, L]              <-- returned

Backward, per chunk (under torch.enable_grad):
    grad_train_logprobs[:, slice]    h_chunk requires_grad
            \                /
             \              /
              v            v
              lm_head + token_logprobs       <-- re-materialized
                       |
                       v
              h_chunk.grad accumulated into [B, L, H]
```


## 3. Current `ChunkedCELoss` on main

The loss is composed of multiple parts, mostly handling distributed tensors and gradients. We only need to touch one of them -- the chunk loop:

```python
...
last_idx = len(h_chunks) - 1
for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
    if fsdp_enabled and i == last_idx:
        lm_head.set_requires_gradient_sync(True, recurse=False)

    logits = lm_head(h_chunk)

    # This is the part that needs to change (mostly)
    # becomes chunk_loss = self._from_chunk_loss(logit_chunk, **kwargs)
    chunk_loss = self.fn(logits, label_chunk)
    if global_valid_tokens is not None:
        chunk_loss = chunk_loss / global_valid_tokens

    total_loss = total_loss + chunk_loss.detach()

    if requires_grad:
        chunk_loss.backward()
        assert h_chunk.grad is not None
        grad_accumulator.add(h_chunk.grad)
        h_chunk.grad = None
...
```

Notice that all we have to change is the `self.fn` and some minor portions regarding passing some extra inputs to this loss. In SFT, this `self.fn` is just the cross_entropy.

**INSIGHT**: We can generalize this workflow to not just CE, but also all token-wise RL losses. We propose renaming it to `ChunkAwareLossWrapper`.

## 4. Trainer view

In this proposal, the trainer never knows which kind of loss is configured:

```python
hidden = model(input_ids, labels=None)                  # _skip_lm_head=True
loss, metrics = self.loss_fn(hidden, **loss_inputs)
loss.backward()
```

That is the entire trainer contract. SFT, GRPO, GSPO, DAPO, CISPO, SAPO all expose the same `(hidden, **loss_inputs) -> (loss, metrics)` shape. The loss class decides how to handle the chunking.

## 5. GRPO point of view

```python
class GRPOLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        clip_eps: float = 0.2

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(
        self,
        hidden: torch.Tensor,                            # [B, L, H]
        *,
        labels: torch.Tensor,                            # [B, L]
        generator_logprobs: torch.Tensor,               # [B, L]
        advantages: torch.Tensor,                        # [B, L]
        loss_mask: torch.Tensor,                         # [B, L]
        global_valid_tokens: torch.Tensor,              # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # TODO: probably needs a better name?
        loss, metrics = ChunkAwareLossWrapper(
            hidden,
            from_chunk_loss_fn=partial(
                self._from_chunk_loss,
                labels=labels,
                generator_logprobs=generator_logprobs,
                advantages=advantages,
                loss_mask=loss_mask,
                global_valid_tokens=global_valid_tokens,
            ),
        )
        return loss, metrics

    # TODO: maybe better name?
    def _from_chunk_loss(
        self,
        logit_chunk: torch.Tensor,                           # [B, chunk, V]
        token_slice: slice,
        *,
        labels: torch.Tensor,                                # [B, L]
        generator_logprobs: torch.Tensor,                    # [B, L]
        advantages: torch.Tensor,                            # [B, L]
        loss_mask: torch.Tensor,                             # [B, L]
        global_valid_tokens: torch.Tensor,                   # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # Slice batch inputs to chunks.
        # Wrapper provides only logit_chunk and token_slice.
        labels_chunk = labels[:, token_slice]                              # [B, chunk]
        generator_logprobs_chunk = generator_logprobs[:, token_slice]      # [B, chunk]
        advantages_chunk = advantages[:, token_slice]                      # [B, chunk]
        loss_mask_chunk = loss_mask[:, token_slice]                        # [B, chunk]

        # compute the actual loss
        return self._token_wise_loss(
            logit_chunk,
            labels_chunk,
            generator_logprobs_chunk,
            advantages_chunk,
            loss_mask_chunk,
            global_valid_tokens,
        )

    # TODO: maybe better name?
    def _token_wise_loss(
        self,
        logits,                                           # [B, chunk, V]
        labels,                                           # [B, chunk]
        generator_logprobs,                               # [B, chunk]
        advantages,                                       # [B, chunk]
        loss_mask,                                        # [B, chunk]
        global_valid_tokens,                              # scalar
    ):
        """Loss unaware if inputs are chunked or not."""
        # cross_entropy returns NLL (negative logprob), so negate to get logprobs.
        train_logprobs = -cross_entropy(logits, labels, reduction="none")  # [B, chunk]
        ratio = torch.exp(train_logprobs - generator_logprobs)            # [B, chunk]
        clipped_ratio = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        token_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
        scalar_loss = (token_loss * loss_mask).sum() / global_valid_tokens

        # metrics computation
        metrics = {}

        return scalar_loss, metrics


```


## 6. SFT point of view

```python
class CrossEntropyLoss(BaseLoss):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(
        self,
        hidden: torch.Tensor,                            # [B, L, H]
        *,
        labels: torch.Tensor,                            # [B, L]
        global_valid_tokens: torch.Tensor,              # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # TODO: probably needs a better name?
        loss, metrics = ChunkAwareLossWrapper(
            hidden,
            from_chunk_loss_fn=partial(
                self._from_chunk_loss,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            ),
        )
        return loss, metrics

    def _from_chunk_loss(
        self,
        logit_chunk: torch.Tensor,                          # [B, chunk, V]
        token_slice: slice,
        *,
        labels: torch.Tensor,                               # [B, L]
        global_valid_tokens: torch.Tensor,                  # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # Slice batch inputs to chunks.
        # Wrapper provides only logit_chunk and token_slice.
        labels_chunk = labels[:, token_slice]               # [B, chunk]

        # compute the actual loss
        return self._token_wise_loss(
            logit_chunk,
            labels_chunk,
            global_valid_tokens,
        )

    def _token_wise_loss(
        self,
        logits,                                             # [B, chunk, V]
        labels,                                             # [B, chunk]
        global_valid_tokens,                                # scalar
    ):
        """Loss unaware if inputs are chunked or not."""
        scalar_loss = cross_entropy(logits, labels, reduction="sum")
        scalar_loss = scalar_loss / global_valid_tokens

        # metrics computation
        metrics = {}

        return scalar_loss, metrics
```

## 7. Solving the sequence-wise problem

Sequence-wise losses (GSPO) need the full `[B, L]` token logprobs before they can compute the objective. Per-chunk inline backward does not produce that tensor.

**Naive approach:**

```python
train_logprobs_chunks = []
for h_chunk, label_chunk in chunks:
    logits = lm_head(h_chunk)                                          # [B, chunk, V]
    train_logprobs_chunks.append(
        -cross_entropy(logits, label_chunk, ignore_index=IGNORE_INDEX, reduction="none")
    )
train_logprobs = torch.cat(train_logprobs_chunks, dim=1)             # [B, L]
loss = sequence_loss(train_logprobs, batch)
loss.backward()
```

This is **not memory-good**. Even though each appended slice is only `[B, chunk]`, autograd keeps each chunk's `[B, chunk, V]` graph alive (LM-head outputs + softmax state) until the final `loss.backward()` runs. Peak vocab memory grows with the number of chunks, defeating the purpose of chunking.

**Proposal:** a `torch.autograd.Function` that performs manual checkpointing at the LM-head boundary — forward runs the chunk loop under `torch.no_grad()` (no chunk graph retained), backward replays each chunk under `torch.enable_grad()`. Same memory shape as main; same FSDP residency policy; same DTensor local chunking.

DISCLAIMER: This was vibecoded.

```python
class _CheckpointedChunkedCE(torch.autograd.Function):
    """Manual checkpointing for token logprobs at the LM-head boundary.

    Returns ``train_logprobs [B, L]`` from forward, replaying each chunk's
    ``lm_head(h_chunk) -> token logprobs`` under ``torch.enable_grad()``
    during backward. See §8 for discussion of ``torch.utils.checkpoint``.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,                            # [B, L, H]
        labels: torch.Tensor,                            # [B, L]
        lm_head: nn.Module,
        num_chunks: int,
        ignore_index: int,
        *lm_params: torch.Tensor,                        # positional so autograd tracks dependency
    ) -> torch.Tensor:
        """Forward pass — keeps only `[B, chunk]` train logprobs per chunk.

        Args:
            hidden: pre-LM-head hidden states `[B, L, H]`.
            labels: target labels `[B, L]`.
            lm_head: ``nn.Module`` projecting `H -> V`.
            num_chunks: number of slices along the sequence axis.
            ignore_index: forwarded to ``cross_entropy``.
            *lm_params: ``lm_head.parameters()`` passed positionally so autograd
                tracks the dependency for the backward.

        Returns:
            ``train_logprobs`` tensor of shape `[B, L]`.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        hidden = _prepare_hidden_for_lm_head(hidden) # Helper that can be shared with ChunkAwareLossWrapper
        ctx.save_for_backward(hidden.detach(), labels)
        ctx.lm_head = lm_head
        ctx.num_chunks = num_chunks
        ctx.ignore_index = ignore_index
        ctx.num_lm_params = len(lm_params)
        ctx.hidden_requires_grad = hidden.requires_grad

        h_chunks, label_chunks, _ = _chunk_hidden_and_labels(
            hidden, labels, num_chunks, requires_grad=False,
        )
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        if fsdp_enabled:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)

        train_logprobs_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for h_chunk, label_chunk in zip(h_chunks, label_chunks):
                logits = lm_head(h_chunk)                                  # [B, chunk, V]
                train_logprobs_chunks.append(
                    -cross_entropy(
                        logits, label_chunk, ignore_index=ignore_index, reduction="none",
                    )                                                       # [B, chunk]
                )

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.reshard()

        return torch.cat(train_logprobs_chunks, dim=1)                    # [B, L]

    @staticmethod
    def backward(
        ctx, grad_train_logprobs: torch.Tensor,                           # [B, L]
    ) -> tuple:
        """Backward pass — replays each chunk under ``enable_grad``.

        Args:
            grad_train_logprobs: upstream gradient `[B, L]` from the outer
                ``loss.backward()`` walk.

        Returns:
            tuple matching the forward positional args: ``(grad_hidden,
            None, None, None, None, *param_grads)``.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        hidden, labels = ctx.saved_tensors
        lm_head = ctx.lm_head
        num_chunks = ctx.num_chunks
        ignore_index = ctx.ignore_index

        h_chunks, label_chunks, slices = _chunk_hidden_and_labels(
            hidden, labels, num_chunks, requires_grad=ctx.hidden_requires_grad,
        )
        grad_accumulator = GradAccumulator(hidden, num_chunks=num_chunks, dtype=torch.float32)
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        last_idx = num_chunks - 1

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)

        with torch.enable_grad():
            for i, (token_slice, h_chunk, label_chunk) in enumerate(
                zip(slices, h_chunks, label_chunks)
            ):
                if fsdp_enabled and i == last_idx:
                    lm_head.set_requires_gradient_sync(True, recurse=False)

                logits = lm_head(h_chunk)                                  # [B, chunk, V]
                train_logprobs_chunk = -cross_entropy(
                    logits, label_chunk, ignore_index=ignore_index, reduction="none",
                )                                                          # [B, chunk]
                torch.autograd.backward(
                    train_logprobs_chunk,
                    grad_tensors=grad_train_logprobs[:, token_slice].contiguous(),
                )
                if ctx.hidden_requires_grad and h_chunk.grad is not None:
                    grad_accumulator.add(h_chunk.grad)
                    h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()

        grad_hidden = (
            grad_accumulator.result().to(hidden.dtype)
            if ctx.hidden_requires_grad
            else None
        )
        # Candidate implementation: nested autograd accumulates lm_head parameter
        # grads directly on lm_head.parameters().
        return grad_hidden, None, None, None, None, *(None,) * ctx.num_lm_params
```

Memory lifetime, illustrative:

```text
Forward, per chunk (under torch.no_grad):
    h_chunk [B, chunk, H]
        |
        v
    lm_head ----> logits [B, chunk, V]          <-- alive briefly
        |             |
        v             v
    token logprobs    [B, chunk]                <-- kept
                      |
                      v
              torch.cat -> [B, L]              <-- returned

Backward, per chunk (under torch.enable_grad):
    grad_train_logprobs[:, slice]    h_chunk requires_grad
            \                /
             \              /
              v            v
              lm_head + token_logprobs       <-- re-materialized
                       |
                       v
              h_chunk.grad accumulated into [B, L, H]
```

**Pros:**

- Peak vocab memory: one chunk's `[B, chunk, V]` during forward; one chunk's `[B, chunk, V]` during backward.
- Caller does normal `loss.backward()` — autograd schedules the replay.

**Cons:**

- One extra LM-head forward pass per chunk during backward (the "checkpointing" cost).

## 8. Why not `torch.utils.checkpoint`

TODO: could we use the `torch.utils.checkpoint.checkpoint` API instead of a hand-written
`torch.autograd.Function`? It seems that we cannot reuse it directly — managing the FSDP
reshard / gradient-sync policy and accumulating the hidden grads into the `[B, L, H]`
buffer across chunks is awkward to express through it — but it might be worth investigating.

```python
train_logprobs_chunks = []
for h_chunk, label_chunk in chunks:
    train_logprobs_chunk = torch.utils.checkpoint.checkpoint(
        _chunk_to_logprobs, h_chunk, label_chunk, lm_head, ignore_index,
        use_reentrant=False,
    )
    train_logprobs_chunks.append(train_logprobs_chunk)
train_logprobs = torch.cat(train_logprobs_chunks, dim=1)
```

## 9. GSPO point of view

GSPO is a sequence-wise RL loss. It needs the full train_logprobs and cannot be calculated token-wise.

```python
class GSPOLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        clip_eps: float = 0.2

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(
        self,
        hidden: torch.Tensor,                            # [B, L, H]
        *,
        labels: torch.Tensor,                            # [B, L]
        generator_logprobs: torch.Tensor,               # [B, L]
        advantages: torch.Tensor,                        # [B, L]
        loss_mask: torch.Tensor,                         # [B, L]
        global_valid_tokens: torch.Tensor,              # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # full [B, L] logprobs are required before the sequence-wise objective
        train_logprobs = _CheckpointedChunkedCE.apply(hidden, labels, ...)

        # rest of the RL loss, computed normally
        return loss, metrics
```

## 10. Reducing the cost of maintaining `_CheckpointedChunkedCE` and `ChunkAwareLossWrapper`

Multiple parts of the function can be reused.

TODO: find common parts and create utilities for them, to decrease redundant code

## 11. Simplification

We could opt to not support `_CheckpointedChunkedCE` and, therefore, sequence-wise losses.
This would simplify RL enablement for now. In the future, if sequence-wise losses become more popular,
a path forward is described in this RFC.
