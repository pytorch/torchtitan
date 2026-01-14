from torchtitan.models import qwen3
import torchtitan
import pytest
import torch

@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gqa_varlen_vs_flash_forward(batch_size, seq_len, dtype):

    sdpa_gqa_attention = qwen3.model.model.Attention(
        sdpa_args := qwen3.model.args.Qwen3ModelArgs(
            dim = 64,
            n_heads = 8,
            n_kv_heads = 4,
            hidden_dim = 512,
            attn_type = "sdpa"
        ),
    ).to(device="cuda:0", dtype=dtype)


    varlen_gqa_attention = qwen3.model.model.Attention(
        varlen_args := qwen3.model.args.Qwen3ModelArgs(
            dim = 64,
            n_heads = 8,
            n_kv_heads = 4,
            hidden_dim = 512,
            attn_type = "varlen"
        ),
    ).to(device="cuda:0", dtype=dtype)

    varlen_gqa_attention.load_state_dict(sdpa_gqa_attention.state_dict())

    x = torch.randn(batch_size, seq_len, varlen_args.dim, device="cuda:0", dtype=dtype)
    mask = torchtitan.models.attention.VarlenMetadata(
        cu_seq_q=torch.tensor([0, *([seq_len]*batch_size)], device="cuda:0", dtype=torch.int32).cumsum(dim=0, dtype=torch.int32),
        cu_seq_k=torch.tensor([0, *([seq_len]*batch_size)], device="cuda:0", dtype=torch.int32).cumsum(dim=0, dtype=torch.int32),
        max_q=seq_len,
        max_k=seq_len,
    )
    rope_cache = qwen3.model.model.precompute_rope_cache(
        varlen_args.head_dim,
        varlen_args.max_seq_len,
        varlen_args.rope_theta
    ).to(device="cuda:0", dtype=dtype)

    sdpa_out   =   sdpa_gqa_attention(x.detach().clone(), rope_cache=rope_cache, attention_masks=None)
    varlen_out = varlen_gqa_attention(x.detach().clone(), rope_cache=rope_cache, attention_masks=mask)
    
    assert torch.allclose(varlen_out, sdpa_out, atol=1e-2, rtol=1e-2), "VarLen GQA and Flash GQA outputs do not match!"

