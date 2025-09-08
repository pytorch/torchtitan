import sys
import types
import torch
import random
from einops import rearrange

# Temporary hack to fix version issues within huggingface transformers.
def _install_flex_attention_stubs():
    """
    Some environments ship a torchao int8 SDPA lowering that references
    names that may not be importable. Provide lightweight stubs so tests
    don't require editing site-packages.
    """
    # As a fallback, create the module torch._inductor.kernel.flex_attention
    # with the expected symbols so imports elsewhere would succeed.
    mod = types.ModuleType("torch._inductor.kernel.flex_attention")
    def construct_strides(size, order):
        size = list(size)
        if not size:
            return []
        strides = [0] * len(size)
        strides[-1] = 1
        for i in range(len(size) - 2, -1, -1):
            strides[i] = strides[i + 1] * int(size[i + 1])
        return strides
    def maybe_realize(nodes):
        return nodes
    mod.construct_strides = construct_strides  # type: ignore[attr-defined]
    mod.maybe_realize = maybe_realize  # type: ignore[attr-defined]
    sys.modules["torch._inductor.kernel.flex_attention"] = mod


_install_flex_attention_stubs()



LOAD_REAL_30b_MODEL = False
LOAD_REAL_235b_MODEL = False
ONLY_FIRST_LAYER = True
MOE = True

assert (LOAD_REAL_30b_MODEL + LOAD_REAL_235b_MODEL) <= 1, "Only one of LOAD_REAL_30b_MODEL and LOAD_REAL_235b_MODEL should be True"

def build_hf_small_config():
    # Tiny config for CPU test
    print("Importing transformers...")
    from transformers.models.qwen3_moe import Qwen3MoeConfig
    from transformers import AutoConfig
    print("Transformers imported successfully")

    global LOAD_REAL_30b_MODEL, MOE, LOAD_REAL_235b_MODEL
    if LOAD_REAL_30b_MODEL:
        return Qwen3MoeConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    elif LOAD_REAL_235b_MODEL:
        return Qwen3MoeConfig.from_pretrained("Qwen/Qwen3-235B-A22B")

    hidden_size = 84
    num_attention_heads = 6

    if MOE:
        return Qwen3MoeConfig(
            hidden_size=hidden_size,
            num_hidden_layers=20,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=2,
            head_dim=hidden_size//num_attention_heads,
            intermediate_size=128,
            moe_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=256,
            vocab_size=151936,
            rope_theta=1_000_000.0,
            decoder_sparse_step=1,
            norm_topk_prob=True,
            mlp_only_layers=[],
            sliding_window=None,
            output_router_logits=False,
        )
    else:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        # cfg = AutoConfig.from_pretrained("Qwen/Qwen3-32B")
        return cfg

def build_hf_model(cfg):
    from transformers import AutoModelForCausalLM
    print("Building HF model...")
    global LOAD_REAL_30b_MODEL, LOAD_REAL_235b_MODEL
    if LOAD_REAL_30b_MODEL:
        assert MOE
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-30B-A3B", 
            torch_dtype=dtype, 
            attn_implementation=None, # 32 bit precision
        )
    elif LOAD_REAL_235b_MODEL:
        assert MOE
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-235B-A22B", 
            torch_dtype=dtype, 
            attn_implementation=None,
        )
    elif not MOE:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            # "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-32B",
            torch_dtype=dtype,
            attn_implementation=None,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            cfg, 
            attn_implementation=None,
            torch_dtype=torch.float32
        )
        print("HF model created successfully")

    model.eval()
    return model


def build_tt_args_from_hf(cfg):
    from torchtitan.models.qwen3.model.args import (
        Qwen3TransformerModelArgs,
    )

    global MOE
    if MOE:
        return Qwen3TransformerModelArgs(
            dim=cfg.hidden_size,
            n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_position_embeddings,
            attention_bias=cfg.attention_bias,
            norm_eps=cfg.rms_norm_eps,
            intermediate_size=cfg.intermediate_size,
            rope_theta=cfg.rope_theta,
            head_dim=cfg.head_dim,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router_aux_loss_coef=cfg.router_aux_loss_coef if hasattr(cfg, "router_aux_loss_coef") else 0.001,
            norm_topk_prob=cfg.norm_topk_prob if hasattr(cfg, "norm_topk_prob") else True,
            decoder_sparse_step=cfg.decoder_sparse_step if hasattr(cfg, "decoder_sparse_step") else 1,
            mlp_only_layers=list(cfg.mlp_only_layers) if hasattr(cfg, "mlp_only_layers") else [],
            use_grouped_mm=False, # necessary for 32 bit precision
        )
    else:
        return Qwen3TransformerModelArgs(
            dim=cfg.hidden_size,
            n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_position_embeddings,
            attention_bias=cfg.attention_bias,
            norm_eps=cfg.rms_norm_eps,
            intermediate_size=cfg.intermediate_size,
            rope_theta=cfg.rope_theta,
            head_dim=cfg.head_dim,
            tie_word_embeddings=cfg.tie_word_embeddings,
            num_experts=0,
        )


def build_tt_model(tt_args):
    from torchtitan.models.qwen3.model.model import Transformer
    print("Building TT model...")
    
    model = Transformer(tt_args)
    print("TT model created, setting to eval mode...")
    model.eval()
    # for safety set all parameters to NaN
    for param in model.parameters():
        param.data = torch.nan * param.data
    print("TT model ready")
    return model

def check_equal(hf_sd, hf_sd_again):
    # all keys should be the same
    assert set(hf_sd.keys()) == set(hf_sd_again.keys()), f"HF and HF roundtrip state dicts have different keys: {hf_sd.keys()} != {hf_sd_again.keys()}"
    # all values should be the same
    for key in hf_sd.keys():
        assert hf_sd[key].shape == hf_sd_again[key].shape, f"HF and HF roundtrip state dicts have different shapes for key {key}: {hf_sd[key].shape} != {hf_sd_again[key].shape}"
        assert torch.allclose(hf_sd[key], hf_sd_again[key]), f"HF and HF roundtrip state dicts have different values for key {key}"
    return True

def convert_hf_to_tt(hf_model, tt_model):
    from torchtitan.models.qwen3.model.state_dict_adapter import (
        Qwen3StateDictAdapter,
    )

    adapter = Qwen3StateDictAdapter(tt_model.model_args, None)
    hf_sd = hf_model.state_dict()
    tie_word_embeddings = hf_model.config.tie_word_embeddings
    tt_sd = adapter.from_hf(hf_sd, tie_word_embeddings=tie_word_embeddings)
    hf_sd_again = adapter.to_hf(tt_sd, tie_word_embeddings=tie_word_embeddings)
    assert check_equal(hf_sd, hf_sd_again), "HF and TT state dicts are not equal"
    print("Roundtripping state dicts works.")
    random_tt_sd = tt_model.state_dict()
    for key in random_tt_sd.keys():
        if key != 'freqs_cis':
            assert key in tt_sd.keys(), f"Key {key} is not in TT state dict"
            assert random_tt_sd[key].shape == tt_sd[key].shape, f"Key {key} has different shape in TT state dict: {random_tt_sd[key].shape} != {tt_sd[key].shape}"
    print("All weights in TT state dict are present in HF state dict.")
    if not MOE:
        hf_parameter_names = [name for name, _ in hf_model.named_parameters()]
        tt_parameter_names = [name for name, _ in tt_model.named_parameters()]
        assert 'freqs_cis' not in tt_parameter_names, f"freqs_cis is in TT state dict, {tt_parameter_names}"
        assert len(hf_parameter_names) == len(tt_parameter_names), f"HF and TT state dicts have different number of keys: {len(hf_parameter_names)}, {len(tt_parameter_names)}"

    missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
    missing = [m for m in missing if m!='freqs_cis']
    if len(missing) > 0:
        print("Missing weights:", missing)
    if len(unexpected) > 0:
        print("Unexpected weights:", unexpected)
    print(tt_model.state_dict().keys())
    return missing, unexpected



def prepare_input_ids(cfg) -> tuple[torch.Tensor, torch.Tensor | None]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    text = "Once upon a time, in a land of floating islands, lived a brave explorer named Elara. One day, she found a map that led to a lost city, but the journey was filled with ancient riddles. Elara began to read the first riddle: 'I have cities, but no houses, forests, but no trees, and water, but no fish. What am I?'"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    assert len(input_ids.shape) == 2, f"Input ids shape {input_ids.shape} is not 2"
    position_ids = None
    return input_ids, position_ids

def compare_models_on_gpu():
    print("Starting model comparison...")
    torch.set_grad_enabled(False)
    device_tt = torch.device("cuda:0")
    device_hf = torch.device("cuda:1")
    print(f"Using devices: TT on {device_tt}, HF on {device_hf}")
    
    # fix all seeds
    torch.manual_seed(0)
    random.seed(0)

    print("Building config...")
    cfg = build_hf_small_config()
    hf_model = build_hf_model(cfg)
    print("Building TT args...")
    tt_args = build_tt_args_from_hf(cfg)
    tt_model = build_tt_model(tt_args).to(torch.float32)
    print("Models built successfully")

    # Align TT weights to HF
    convert_hf_to_tt(hf_model, tt_model)
    hf_model.to(device=device_hf)
    tt_model.to(device=device_tt, dtype=torch.float32)

    input_ids, position_ids = prepare_input_ids(cfg)

    # HF forward
    with torch.no_grad():
        pos_ids_ = position_ids.to(device_hf) if position_ids is not None else None
        hf_outputs = hf_model(input_ids=input_ids.to(device_hf), position_ids=pos_ids_)
        hf_logits = hf_outputs.logits.cpu()

    print("TT forward.")

    # TT forward
    with torch.no_grad():
        pos_ids_ = position_ids.to(device_tt) if position_ids is not None else None
        tt_dict = tt_model(tokens=input_ids.to(device_tt), position_ids=pos_ids_)
        tt_logits = tt_dict['logits'].cpu()

    assert hf_logits.shape == tt_logits.shape, f"HF logits shape {hf_logits.shape} != TT logits shape {tt_logits.shape}"

    max_abs_diff = (hf_logits - tt_logits).abs().max().item()
    print("Max absolute diff:", max_abs_diff)
    argmax_hf = hf_logits.argmax(dim=-1)
    argmax_tt = tt_logits.argmax(dim=-1)
    print("Argmax mismatch:", (argmax_hf != argmax_tt).sum().item(), "mistmatch rate:", (argmax_hf != argmax_tt).sum().item()/argmax_hf.numel())
    assert max_abs_diff < 1e-3, f"Mismatch too large: {max_abs_diff}, max value: {hf_logits.max()}, min value: {hf_logits.min()}"


def compare_first_layer_on_gpu():
    torch.set_grad_enabled(False)
    device_tt = torch.device("cpu")
    device_hf = torch.device("cpu")
    # fix all seeds
    torch.manual_seed(0)
    random.seed(0)

    cfg = build_hf_small_config()
    hf_model = build_hf_model(cfg)
    tt_args = build_tt_args_from_hf(cfg)
    tt_model = build_tt_model(tt_args).to(torch.float32)

    # Align TT weights to HF
    convert_hf_to_tt(hf_model, tt_model)
    hf_model.to(device=device_hf)
    tt_model.to(device=device_tt, dtype=torch.float32)

    input_ids, position_ids = prepare_input_ids(cfg)

    print("First layer forward.")

    def hf_forward(hf_model, input_ids, position_ids):
        states_dict = {}
        with torch.no_grad():
            input_ids = input_ids.to(device_hf)
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1)
            position_ids = position_ids.to(device_hf)
            hf_first_layer = hf_model.model.layers[0]
            inputs_embeds = hf_model.model.embed_tokens(input_ids)
            states_dict['inputs_embeds'] = inputs_embeds.cpu()
            hidden_states = inputs_embeds
            # create position embeddings to be shared across the decoder layers
            position_embeddings = hf_model.model.rotary_emb(hidden_states, position_ids)

            # Start of first layer
            residual = hidden_states
            hidden_states = hf_first_layer.input_layernorm(hidden_states)
            states_dict['normed_embeddings'] = hidden_states.cpu()
            # Self Attention
            def self_attn_forward(attn_block, hidden_states, position_embeddings, position_ids):
                from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
                from typing import Callable

                def reverse_permute(x, nheads):
                    return rearrange(x, 'b s nheads (two ropedim) -> b s nheads (ropedim two)', two=2, nheads=nheads)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_block.head_dim)

                query_states_projected = attn_block.q_proj(hidden_states).view(hidden_shape)
                key_states_projected = attn_block.k_proj(hidden_states).view(hidden_shape)

                states_dict['query_states_projected'] = reverse_permute(query_states_projected.cpu(), attn_block.config.num_attention_heads)
                states_dict['key_states_projected'] = reverse_permute(key_states_projected.cpu(), attn_block.config.num_key_value_heads)

                query_states = attn_block.q_norm(query_states_projected).transpose(1, 2)
                key_states = attn_block.k_norm(key_states_projected).transpose(1, 2)
                value_states = attn_block.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                states_dict['query_states'] = reverse_permute(query_states.transpose(1, 2), attn_block.config.num_attention_heads).cpu()
                states_dict['key_states'] = reverse_permute(key_states.transpose(1, 2), attn_block.config.num_key_value_heads).cpu()
                states_dict['value_states'] = value_states.transpose(1, 2).cpu()

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                states_dict['query_states_after_rope'] = reverse_permute(query_states.transpose(1, 2), attn_block.config.num_attention_heads).cpu()
                states_dict['key_states_after_rope'] = reverse_permute(key_states.transpose(1, 2), attn_block.config.num_key_value_heads).cpu()

                attention_interface: Callable = eager_attention_forward
                if attn_block.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[attn_block.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    attn_block,
                    query_states,
                    key_states,
                    value_states,
                    None,
                    dropout=0.0,
                    scaling=attn_block.scaling,
                    sliding_window=attn_block.sliding_window,  # diff with Llama
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                states_dict['attn_output'] = attn_output.cpu()
                attn_output = attn_block.o_proj(attn_output)
                return attn_output, attn_weights

            hidden_states, _ = self_attn_forward(hf_first_layer.self_attn, hidden_states, position_embeddings, position_ids)
            states_dict['attention_output'] = hidden_states.cpu()
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = hf_first_layer.post_attention_layernorm(hidden_states)
            states_dict['normed_embeddings_pre_mlp'] = hidden_states.cpu()
            hidden_states = hf_first_layer.mlp(hidden_states)
            # For the MoE layers, we need to unpack
            if isinstance(hidden_states, tuple):
                states_dict['mlp_output'] = hidden_states[0].cpu()
                hidden_states, _ = hidden_states
            else:
                states_dict['mlp_output'] = hidden_states.cpu()
            hidden_states = residual + hidden_states

            return hidden_states.cpu(), states_dict

    def tt_forward(tt_model, input_ids, position_ids):
        with torch.no_grad():
            states_dict = {}
            device_tt = torch.device('cuda:0') # TT needs the model to be on the GPU, so we only put the first layer on the GPU.
            input_ids = input_ids.to(device_tt)
            position_ids = position_ids.to(device_tt) if position_ids is not None else None
            tt_first_layer = list(tt_model.layers.values())[0].to(device_tt)
            tt_model.tok_embeddings = tt_model.tok_embeddings.to(device_tt)
            tt_model.freqs_cis = tt_model.freqs_cis.to(device_tt)

            h = tt_model.tok_embeddings(input_ids)
            states_dict['inputs_embeds'] = h.cpu()
            h = h.to(device_tt)

            if position_ids is not None:
                freqs_cis = tt_model.freqs_cis[position_ids]
            else:
                freqs_cis = tt_model.freqs_cis

            # Start of first layer
            # Attention sub-block with pre-normalization
            normed_embeddings = tt_first_layer.attention_norm(h)
            states_dict['normed_embeddings'] = normed_embeddings.cpu()

            def self_attn_forward(attn_block, x: torch.Tensor, freqs_cis: torch.Tensor, position_ids: torch.Tensor | None):
                """Forward pass using TT attention backends (RoPE cis like Llama)."""
                from torchtitan.models.qwen3.model.model import repeat_kv, apply_rotary_emb
                def reshape_to_tt(x):
                    return rearrange(x, 'b s h (two d) -> b s h (d two)', two=2)
                    
                bs, seqlen, _ = x.shape
                # Project and reshape Q, K, V
                xq = attn_block.wq(x).view(bs, seqlen, -1, attn_block.head_dim)
                xk = attn_block.wk(x).view(bs, seqlen, -1, attn_block.head_dim)
                xv = attn_block.wv(x).view(bs, seqlen, -1, attn_block.head_dim)

                # Apply QK Norm (Qwen3 specific) before RoPE
                states_dict['query_states_projected'] = xq.cpu()
                states_dict['key_states_projected'] = xk.cpu()
                xq = attn_block.q_norm(xq)
                xk = attn_block.k_norm(xk)

                states_dict['query_states'] = xq.cpu()
                states_dict['key_states'] = xk.cpu()
                states_dict['value_states'] = xv.cpu()

                # Apply Rotary Positional Embeddings (cis path like Llama)
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, packing_mode=position_ids is not None)
                states_dict['query_states_after_rope'] = xq.cpu()
                states_dict['key_states_after_rope'] = xk.cpu()

                # Grouped Query Attention: repeat K, V heads
                keys = repeat_kv(xk, attn_block.n_rep)
                values = repeat_kv(xv, attn_block.n_rep)

                # Transpose for attention calculation
                xq = xq.transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)

                # SDPA/Flex attention
                output = attn_block.sdpa(xq, keys, values, position_ids=position_ids, scale=attn_block.scaling)

                # Reshape and project output
                output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
                states_dict['attn_output'] = output.cpu()
                return attn_block.wo(output)
            attention_output = self_attn_forward(tt_first_layer.attention, normed_embeddings, freqs_cis, position_ids)
            states_dict['attention_output'] = attention_output.cpu()
            h = h + attention_output

            # FFN/MoE sub-block with pre-normalization
            mlp_input = tt_first_layer.ffn_norm(h)
            states_dict['normed_embeddings_pre_mlp'] = mlp_input.cpu()
            if tt_first_layer.moe_enabled:
                mlp_output, new_router_logits = tt_first_layer.mlp(mlp_input)
            else:
                mlp_output = tt_first_layer.mlp(mlp_input)
            states_dict['mlp_output'] = mlp_output.cpu()
            out = h + mlp_output
            return out.cpu(), states_dict

    hf_hidden_states, hf_states_dict = hf_forward(hf_model, input_ids, position_ids)
    tt_hidden_states, tt_states_dict = tt_forward(tt_model, input_ids, position_ids)
    assert hf_hidden_states.shape == tt_hidden_states.shape, f"HF hidden states shape {hf_hidden_states.shape} != TT hidden states shape {tt_hidden_states.shape}"
    max_abs_diff = (hf_hidden_states - tt_hidden_states).abs().max().item()
    print("Max absolute diff:", max_abs_diff)
    print("Checking every step in the first layer.")
    for key in hf_states_dict.keys():
        assert hf_states_dict[key].shape == tt_states_dict[key].shape, f"HF and TT states dicts have different shapes for key {key}: {hf_states_dict[key].shape} != {tt_states_dict[key].shape}"
        max_abs_diff = (hf_states_dict[key] - tt_states_dict[key]).abs().max().item()
        mean_abs_diff = (hf_states_dict[key] - tt_states_dict[key]).abs().mean().item()
        hf_abs_max = hf_states_dict[key].abs().max().item()
        hf_abs_mean = hf_states_dict[key].abs().mean().item()
        tt_abs_max = tt_states_dict[key].abs().max().item()
        tt_abs_mean = tt_states_dict[key].abs().mean().item()
        print(f"Max absolute diff for {key}: {max_abs_diff}, mean absolute diff: {mean_abs_diff}, hf abs max: {hf_abs_max}, hf abs mean: {hf_abs_mean}, tt abs max: {tt_abs_max}, tt abs mean: {tt_abs_mean}")

if __name__ == "__main__":
    try:
        print("Starting test...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
        
        if ONLY_FIRST_LAYER:
            print("Running first layer comparison...")
            compare_first_layer_on_gpu()
        else:
            print("Running full model comparison...")
            compare_models_on_gpu()
            
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

