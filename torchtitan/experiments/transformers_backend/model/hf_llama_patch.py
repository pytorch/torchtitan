

import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaDecoderLayer
from transformers.modeling_utils import PreTrainedModel

_original_llama_decoder_layer_init = LlamaDecoderLayer.__init__

def _llama_decoder_layer_init_patched(self, config: LlamaConfig, layer_idx: int):
    _original_llama_decoder_layer_init(self, config, layer_idx)
    self.mlp.layer_idx = layer_idx

def _initialize_weights_patched(self, module):
    # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
    # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
    # which prevents subsequent proper initialization.
    if getattr(module, "_is_hf_initialized", False):
        return

    for param in module.parameters(recurse=True):
        if param.device.type == "meta":
            return
    
    # If not on a meta device, call the original weight initialization
    self._init_weights(module)
    module._is_hf_initialized = True

def _init_weights_patched(self, module):
    """
    Patched version of _init_weights to match TorchTitan's initialization for Llama.
    `self` is a LlamaPreTrainedModel instance.
    """
    config = self.config

    if isinstance(module, (LlamaAttention, LlamaMLP)):
        layer_idx = module.layer_idx

        if config.depth_init:
            init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
        else:
            init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

    if isinstance(module, LlamaAttention):
        nn.init.trunc_normal_(module.q_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.k_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.v_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.o_proj.weight, mean=0.0, std=init_std)
    
    elif isinstance(module, LlamaMLP):
        nn.init.trunc_normal_(module.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.up_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(module.down_proj.weight, mean=0.0, std=init_std)

    elif module is getattr(self, "lm_head", None): #TODO(3outeille): find a better way to detect lm_head
        final_out_std = config.hidden_size**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        std = config.initializer_range
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    elif (
        isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        or "LayerNorm" in module.__class__.__name__
        or "RMSNorm" in module.__class__.__name__
    ):
        # Norms can exist without weights (in which case they are None from torch primitives)
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()


def patch_hf_llama():
    LlamaDecoderLayer.__init__ = _llama_decoder_layer_init_patched
    PreTrainedModel._init_weights = _init_weights_patched
    PreTrainedModel._initialize_weights = _initialize_weights_patched