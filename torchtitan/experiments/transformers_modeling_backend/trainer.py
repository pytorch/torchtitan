# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.trainer import Trainer


class HFTransformerTrainer(Trainer):
    """Trainer for the HF transformers backend.

    Adds two behaviors over the core ``Trainer``:

    1. Under context parallelism it builds the flex ``BlockMask`` from the
       unsharded positions *before* the base class shards inputs. The base
       ``post_dataloading_process`` then routes the mask through
       ``prepare_context_parallel_input``, which shards the mask's Q axis (and
       the inputs/positions) with the configured load balancer -- yielding a
       local-Q x full-KV mask that matches the k/v all-gathered inside the flex
       kernel (see ``_wrap_flex_kernel_cp`` in parallelize.py). The core
       ``Trainer`` only builds masks for ``Decoder.Config`` models; the HF
       backend is not one, so without this the mask would be built inside
       ``model.forward`` from already-sharded positions (a square local x local
       mask, wrong once k/v span the full sequence). Outside CP, mask building
       stays in the model -- this hook is a no-op there. This also covers SFT:
       an SFT block-causal mask is the same same-document causal mask the packed
       pretraining path builds, so an SFT flavor only needs the flex impl plus
       attn_mask_type="block_causal".

    2. Fails loud when a model runs under CP with the "headtail" load balancer,
       which cannot shard a flex ``BlockMask`` (see
       ``_validate_cp_load_balancer``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: "HFTransformerTrainer.Config"):
        super().__init__(config)
        self._validate_cp_load_balancer()
        # Under spmd_types, loss-parallel cross entropy needs the global vocab
        # size explicitly: the default backend infers it from the DTensor pred's
        # shape, but under spmd_types pred is a plain local (vocab-sharded) shard
        # so its last dim is the local size. Native models set this at config
        # time via decoder_vocab_size(model_spec); the HF backend's vocab comes
        # from AutoConfig and is only known after build, so fill it in here when
        # a CrossEntropyLoss left it unset. Harmless under other backends.
        loss_fn = getattr(self, "loss_fn", None)
        if loss_fn is not None and getattr(loss_fn, "global_vocab_size", None) is None:
            vocab_size = getattr(self.model_parts[0].model.config, "vocab_size", None)
            if vocab_size is not None and hasattr(loss_fn, "global_vocab_size"):
                loss_fn.global_vocab_size = vocab_size

    def _validate_cp_load_balancer(self) -> None:
        """Reject headtail load balancing under CP.

        A flex ``BlockMask`` can only be sharded by the "ptrr" balancer (or with
        balancing disabled via None); the
        default "headtail" cannot shard it. Raise rather than silently
        overriding a user-set value.
        """
        if not self.parallel_dims.cp_enabled:
            return
        if self.config.parallelism.context_parallel_load_balancer == "headtail":
            raise ValueError(
                "context_parallel_load_balancer='headtail' cannot shard a "
                "flex-attention BlockMask under context parallelism. Set "
                "--parallelism.context_parallel_load_balancer to 'ptrr' (or "
                "None to disable balancing)."
            )

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.parallel_dims.cp_enabled and "attention_masks" not in input_dict:
            positions = input_dict.get("positions")
            if positions is not None:
                masks = self.model_parts[0].get_attention_masks(positions=positions)
                if masks is not None:
                    input_dict = {**input_dict, "attention_masks": masks}
        return super().post_dataloading_process(input_dict, labels)
