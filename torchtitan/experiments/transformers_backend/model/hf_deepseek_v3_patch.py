import os
import torch.nn as nn
from torchtitan.utils.test_utils import seeded_init_decorator_for_test

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3MLP, DeepseekV3MoE, DeepseekV3DecoderLayer
from transformers.modeling_utils import PreTrainedModel


_original_deepseek_v3_decoder_layer_init = DeepseekV3DecoderLayer.__init__

def _deepseek_v3_decoder_layer_init_patched(self, config: DeepseekV3Config, layer_idx: int):
    _original_deepseek_v3_decoder_layer_init(self, config, layer_idx)
    
    self.layer_idx = layer_idx
    self.mlp.layer_idx = layer_idx
    
    if hasattr(self.mlp, 'experts'):
        for expert in self.mlp.experts:
            expert.layer_idx = layer_idx
        self.mlp.shared_experts.layer_idx = layer_idx
    
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

@seeded_init_decorator_for_test(seed=os.environ.get("SEED"))
def _init_weights_patched(self, module):
    """
    Patched version of _init_weights to match TorchTitan's initialization for Llama.
    `self` is a LlamaPreTrainedModel instance.
    """
    config = self.config
    
    #TODO(3outeille): only out_proj/down_proj needs std=init_std. so we can refactor to loop over module and only init last layer with std=init_std
    if isinstance(module, (DeepseekV3Attention, DeepseekV3MLP, DeepseekV3MoE)):
        layer_idx = module.layer_idx
        init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5

    if isinstance(module, DeepseekV3Attention):
        if hasattr(module, 'q_proj'):
            nn.init.trunc_normal_(module.q_proj.weight, mean=0.0, std=0.02)
        else:
            nn.init.trunc_normal_(module.q_a_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(module.q_b_proj.weight, mean=0.0, std=0.02)
        
        nn.init.trunc_normal_(module.kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.kv_b_proj.weight, mean=0.0, std=0.02)
        
        nn.init.trunc_normal_(module.o_proj.weight, mean=0.0, std=init_std)
    
    elif isinstance(module, DeepseekV3MLP):
        nn.init.trunc_normal_(module.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.up_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.down_proj.weight, mean=0.0, std=init_std)

    elif isinstance(module, DeepseekV3MoE):
        nn.init.trunc_normal_(module.gate.weight, mean=0.0, std=init_std)
        for expert in module.experts:
            nn.init.trunc_normal_(expert.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(expert.up_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(expert.down_proj.weight, mean=0.0, std=init_std)
        
        nn.init.trunc_normal_(module.shared_experts.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.shared_experts.up_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.shared_experts.down_proj.weight, mean=0.0, std=init_std)

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


def patch_hf_deepseek_v3():
    DeepseekV3DecoderLayer.__init__ = _deepseek_v3_decoder_layer_init_patched
    PreTrainedModel._init_weights = _init_weights_patched
    PreTrainedModel._initialize_weights = _initialize_weights_patched
