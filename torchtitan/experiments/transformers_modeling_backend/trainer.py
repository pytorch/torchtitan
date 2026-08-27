# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.trainer import Trainer


class HFTransformerTrainer(Trainer):
    """Trainer for the HF transformers backend.

    The flex ``BlockMask`` used under context parallelism is now built inside
    ``HFTransformerModel.preprocess_inputs`` (the model), so this trainer no
    longer overrides the dataloading hook to inject it.

    Its only remaining behavior over the core ``Trainer`` is to fail loud when a
    model runs under CP with the "headtail" load balancer, which cannot shard a
    flex ``BlockMask`` (see ``_validate_cp_load_balancer``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: "HFTransformerTrainer.Config"):
        super().__init__(config)
        self._validate_cp_load_balancer()

    def _validate_cp_load_balancer(self) -> None:
        """Reject headtail load balancing under CP.

        A flex ``BlockMask`` can only be sharded by the "ptrr" balancer (or with
        balancing disabled via None); the
        "headtail" cannot shard it. Raise rather than silently overriding a
        user-set value.
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
