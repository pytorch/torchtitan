import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, LlamaMLP, LlamaDecoderLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional


_original_llama_decoder_layer_init = LlamaDecoderLayer.__init__

def _llama_decoder_layer_init_patched(self, config: LlamaConfig, layer_idx: int):
    _original_llama_decoder_layer_init(self, config, layer_idx)
    self.layer_idx = layer_idx
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

def _patched_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """
    A patched version of LlamaModel.forward that disables the causal mask.
    This is a direct copy of the original method with one line changed.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # --- START OF PATCH ---
    # NOTE(3outeille): When TP enabled, the causal_mask will be created based on input_embeds which has sharded seq_len.
    # We set it to False so that SDPA is creating the causal mask based on query & key seq_len.
    causal_mask = None
    # --- END OF PATCH ---

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

def patch_hf_llama():
    LlamaModel.forward = _patched_forward
    LlamaDecoderLayer.__init__ = _llama_decoder_layer_init_patched
    PreTrainedModel._init_weights = _init_weights_patched
    PreTrainedModel._initialize_weights = _initialize_weights_patched