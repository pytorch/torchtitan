import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import math
from torch.nn import init


def patch_hf_llama_like(decoder_layer_cls, attention_cls, mlp_cls=None):
    """
    This patch modifies a Hugging Face Llama-like model's weight initialization to match
    the initialization scheme used in TorchTitan. This is crucial for ensuring
    bit-for-bit reproducibility when converting checkpoints between the native
    TorchTitan format and the Hugging Face format.

    The patch targets the following aspects of the model:
    - `PreTrainedModel._initialize_weights`: Handles meta device initialization correctly.
    - `PreTrainedModel._init_weights`: Implements TorchTitan's specific initialization
      for attention, MLP, embedding, and layer norm layers. This includes depth-dependent
      initialization for attention and MLP layers.
    - `DecoderLayer.__init__`: Adds `layer_idx` to attention and MLP modules within
      each decoder layer, which is required for the depth-dependent initialization.
    """

    _original_decoder_layer_init = decoder_layer_cls.__init__

    def _decoder_layer_init_patched(self, config: PretrainedConfig, layer_idx: int):
        _original_decoder_layer_init(self, config, layer_idx)
        self.layer_idx = layer_idx
        # Ensure both attention and mlp modules have layer_idx for depth-based init
        if hasattr(self, "self_attn"):
            self.self_attn.layer_idx = layer_idx
        # some models might not have mlp in each layer
        if hasattr(self, "mlp") and self.mlp is not None:
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
        Patched version of _init_weights to match TorchTitan's initialization for Llama-like models.
        `self` is a PreTrainedModel instance.
        """
        config = self.config

        # check if layer is  (resid_dropout): Dropout(p=0.1, inplace=False)
        if hasattr(module, "resid_dropout"):
            print()

        # Build tuple of classes to check for layer_idx-based init_std calculation
        layer_idx_classes = [attention_cls]
        if mlp_cls:
            layer_idx_classes.append(mlp_cls)
        layer_idx_classes = tuple(layer_idx_classes)

        if isinstance(module, layer_idx_classes):
            if not hasattr(module, "layer_idx"):
                return
            layer_idx = module.layer_idx

            if hasattr(config, "depth_init") and config.depth_init:
                init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
            else:
                init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

        if isinstance(module, attention_cls):
            # Initialize weights and biases for q, k, v projections
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(module, proj_name)
                nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
                if proj.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(proj.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(proj.bias, -bound, bound)

            # Handle different names for the output projection layer
            o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
            if o_proj is not None:
                nn.init.trunc_normal_(o_proj.weight, mean=0.0, std=init_std)
                if o_proj.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(o_proj.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(o_proj.bias, -bound, bound)

        elif mlp_cls and isinstance(module, mlp_cls):
            # Handle different names for MLP layers
            gate_proj = getattr(module, "gate_proj", getattr(module, "fc1", None))
            up_proj = getattr(module, "up_proj", None)
            down_proj = getattr(module, "down_proj", getattr(module, "fc2", None))

            # gate_proj (or fc1) should always use std=0.02 for numerical stability.
            if gate_proj is not None:
                nn.init.trunc_normal_(gate_proj.weight, mean=0.0, std=0.02)
                if gate_proj.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(gate_proj.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(gate_proj.bias, -bound, bound)
            # up_proj and down_proj (or fc2) use the depth-dependent init_std.
            if up_proj is not None:
                nn.init.trunc_normal_(up_proj.weight, mean=0.0, std=init_std)
                if up_proj.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(up_proj.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(up_proj.bias, -bound, bound)
            if down_proj is not None:
                nn.init.trunc_normal_(down_proj.weight, mean=0.0, std=init_std)
                if down_proj.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(down_proj.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(down_proj.bias, -bound, bound)

        elif module is getattr(
            self, "lm_head", None
        ):  # TODO(3outeille): find a better way to detect lm_head
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
            isinstance(
                module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            )
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    decoder_layer_cls.__init__ = _decoder_layer_init_patched
    PreTrainedModel._init_weights = _init_weights_patched
    PreTrainedModel._initialize_weights = _initialize_weights_patched
