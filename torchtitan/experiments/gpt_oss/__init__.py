# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from .infra.optimizer import build_gptoss_optimizers

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_gptoss
from .model.args import GptOssModelArgs
from .model.model import GptOssModel

__all__ = [
    "parallelize_gptoss",
    "GptOssModelArgs",
    "GptOssModel",
    "gptoss_configs",
]


gptoss_configs = {
    "debugmodel": GptOssModelArgs(
        hidden_size=256,
        num_hidden_layers=4,
        use_flex_attn=False,
        use_grouped_mm=False,
    ),
    "20b": GptOssModelArgs(
        num_hidden_layers=24,
        num_local_experts=32,
    ),
    "120b": GptOssModelArgs(
        num_hidden_layers=36,
        num_local_experts=128,
    ),
}


register_train_spec(
    TrainSpec(
        name="gpt_oss",
        cls=GptOssModel,
        config=gptoss_configs,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        build_optimizers_fn=build_gptoss_optimizers,  # use optimizer hooks to update expert weights
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
