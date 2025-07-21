# Copyright (c) 2025, Anthropic Research Labs
# All rights reserved.

from torchtitan.datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer import TikTokenizer
from torchtitan.models.mistral3.model import (
    ModelArgs,
    Mistral3RMSNorm,
    Mistral3PatchMerger,
    Mistral3MultiModalProjector,
    VisionEncoder,
    MultimodalDecoder,
    Mistral3ForConditionalGeneration,
)
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.train_spec import register_train_spec, TrainSpec

from .parallelize_mistral3 import parallelize_mistral3
from .pipeline_mistral3 import pipeline_mistral3


from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_mistral3
from .infra.pipeline import pipeline_mistral3
#from .model.args import ModelArgs
from .model.model import Mistral3ForConditionalGeneration


__all__ = [
    "parallelize_mistral3",
    "pipeline_mistral3",
    "ModelArgs",
    "Mistral3RMSNorm",
    "Mistral3PatchMerger",
    "Mistral3MultiModalProjector",
    "VisionEncoder",
    "MultimodalDecoder",
    "Mistral3ForConditionalGeneration",
    "mistral3_configs",
]

# Define model configurations
mistral3_configs = {
    "24B": ModelArgs(
        # vision encoder part
        vision_embed_dim=1024,
        vision_num_layers=24,
        vision_num_heads=16,
        vision_feature_layer=-2,
        patch_size=14,
        image_size=1540,
        in_channels=3,
        spatial_merge_size=2,
        
        # projection part
        num_layers_projection=8,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,

        # decoder part
        decoder_embed_dim=5120,
        decoder_num_layers=40,
        decoder_num_heads=32,
        decoder_num_kv_heads=8,
        fusion_interval=8,
        image_token_index=10,
        
        # common part
        vocab_size=131072,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=1000000000.0,
        max_seq_len=131072,
    ),
}



# Register the model
register_train_spec(
    TrainSpec(
        name="mistral3",
        cls=Mistral3ForConditionalGeneration,
        config=mistral3_configs,
        parallelize_fn=parallelize_mistral3,
        pipelining_fn=pipeline_mistral3,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        tokenizer_cls=TikTokenizer,
    )
)