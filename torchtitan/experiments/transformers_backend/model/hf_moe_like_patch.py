import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


def patch_hf_moe_like(decoder_layer_cls, attention_cls, mlp_cls, moe_cls):
    """
    This patch modifies a Hugging Face MoE (Mixture-of-Experts) model's weight
    initialization to match the initialization scheme used in TorchTitan,
    drawing from patterns in models like DeepseekV3.

    The patch targets:
    - `PreTrainedModel._initialize_weights`: For correct meta device initialization.
    - `PreTrainedModel._init_weights`: To implement TorchTitan's specific initialization
      for attention, MLP, MoE, embedding, and layer norm layers.
    - `DecoderLayer.__init__`: Adds `layer_idx` to attention, MLP, and MoE expert
      modules, required for depth-dependent initialization.
    """

    _original_decoder_layer_init = decoder_layer_cls.__init__

    def _decoder_layer_init_patched(self, config: PretrainedConfig, layer_idx: int):
        _original_decoder_layer_init(self, config, layer_idx)
        self.layer_idx = layer_idx

        if hasattr(self, "self_attn"):
            self.self_attn.layer_idx = layer_idx

        if hasattr(self, "mlp"):
            self.mlp.layer_idx = layer_idx
            if hasattr(self.mlp, "experts"):
                for expert in self.mlp.experts:
                    expert.layer_idx = layer_idx
            if hasattr(self.mlp, "shared_experts"):
                # Not all MoE models have shared experts
                if self.mlp.shared_experts is not None:
                    self.mlp.shared_experts.layer_idx = layer_idx

    def _initialize_weights_patched(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        for param in module.parameters(recurse=True):
            if param.device.type == "meta":
                return
        self._init_weights(module)
        module._is_hf_initialized = True

    def _init_weights_patched(self, module):
        """
        Patched version of _init_weights for MoE models.
        """
        config = self.config
        init_std = None

        if isinstance(module, (attention_cls, mlp_cls, moe_cls)):
            if hasattr(module, "layer_idx"):
                layer_idx = module.layer_idx
                if hasattr(config, "depth_init") and config.depth_init:
                    init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
                else:
                    # Fallback for models without depth_init
                    init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

        if isinstance(module, attention_cls):
            # Handle different attention projection layer names by initializing if they exist
            if hasattr(module, "q_proj"):
                nn.init.trunc_normal_(module.q_proj.weight, mean=0.0, std=0.02)
            if hasattr(module, "k_proj"):
                nn.init.trunc_normal_(module.k_proj.weight, mean=0.0, std=0.02)
            if hasattr(module, "v_proj"):
                nn.init.trunc_normal_(module.v_proj.weight, mean=0.0, std=0.02)

            if hasattr(module, "q_a_proj"):
                nn.init.trunc_normal_(module.q_a_proj.weight, mean=0.0, std=0.02)
            if hasattr(module, "q_b_proj"):
                nn.init.trunc_normal_(module.q_b_proj.weight, mean=0.0, std=0.02)
            
            if hasattr(module, "kv_a_proj_with_mqa"):
                nn.init.trunc_normal_(module.kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
            if hasattr(module, "kv_b_proj"):
                nn.init.trunc_normal_(module.kv_b_proj.weight, mean=0.0, std=0.02)
            
            if hasattr(module, "o_proj") and init_std is not None:
                nn.init.trunc_normal_(module.o_proj.weight, mean=0.0, std=init_std)

        elif isinstance(module, mlp_cls):
            nn.init.trunc_normal_(module.gate_proj.weight, mean=0.0, std=0.02)
            # DeepseekV3 uses std=0.02 for up_proj, unlike Llama
            nn.init.trunc_normal_(module.up_proj.weight, mean=0.0, std=0.02)
            if init_std is not None:
                nn.init.trunc_normal_(module.down_proj.weight, mean=0.0, std=init_std)

        elif isinstance(module, moe_cls):
            if hasattr(module, "gate") and init_std is not None:
                nn.init.trunc_normal_(module.gate.weight, mean=0.0, std=init_std)
            if hasattr(module, "experts"):
                for expert in module.experts:
                    nn.init.trunc_normal_(expert.gate_proj.weight, mean=0.0, std=0.02)
                    nn.init.trunc_normal_(expert.up_proj.weight, mean=0.0, std=0.02)
                    if init_std is not None:
                        nn.init.trunc_normal_(expert.down_proj.weight, mean=0.0, std=init_std)
            if hasattr(module, "shared_experts") and module.shared_experts is not None:
                nn.init.trunc_normal_(module.shared_experts.gate_proj.weight, mean=0.0, std=0.02)
                nn.init.trunc_normal_(module.shared_experts.up_proj.weight, mean=0.0, std=0.02)
                if init_std is not None:
                    nn.init.trunc_normal_(module.shared_experts.down_proj.weight, mean=0.0, std=init_std)

        elif module is getattr(self, "lm_head", None):
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

        elif "LayerNorm" in module.__class__.__name__ or "RMSNorm" in module.__class__.__name__:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    decoder_layer_cls.__init__ = _decoder_layer_init_patched
    PreTrainedModel._init_weights = _init_weights_patched
    PreTrainedModel._initialize_weights = _initialize_weights_patched
