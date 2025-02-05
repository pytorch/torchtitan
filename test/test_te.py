import copy

import torch
import torch.nn as nn

# path hack, TODO remove
import sys
sys.path.insert(0, '/home/vasiliy/local/torchtitan/')
import torchtitan.te_utils as te_utils
from torchtitan.models.norms import build_norm
from torchtitan.models.llama.model import FeedForward, Attention, ModelArgs, precompute_freqs_cis

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
maybe_te_float8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

def test_linear_module_swap():
    x = torch.randn(32, 32, device='cuda')

    m = nn.Sequential(nn.Linear(32, 32)).cuda()
    te_utils.swap_linear_to_te_linear(m)
    print(m)
    m = torch.compile(m)

    with maybe_te_float8_ctx:
        y = m(x)
    y.sum().backward()

    print('done')

# Subsection of TransformerBlock with only the ffn norm and the ffn
class NormFFNBlock(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        self.ffn_norm = build_norm("rmsnorm", dim, eps=1e-12)
        self.feed_forward = FeedForward(dim, hidden_dim, multiple_of, None)

    def forward(self, h):
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class NormAttnBlock(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.attention_norm = build_norm("rmsnorm", model_args.dim, eps=1e-12)
        self.attention = Attention(model_args)
        self.model_args = model_args
        self.freqs_cis = precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        ).cuda()

    def forward(self, x):
        x = self.attention_norm(x)
        x = self.attention(x, self.freqs_cis)
        return x

def SQNR(x, y):
    return 20 * torch.log10(
        torch.linalg.norm(x) / torch.linalg.norm(x - y)
    )

def test_norm_attn_rewrite():
    dim = 256
    model_args = ModelArgs()
    m = NormAttnBlock(model_args).cuda().bfloat16()
    m_copy = copy.deepcopy(m)
    te_utils.swap_norm_attn_to_te_friendly_norm_attn(m_copy)
    print(m)

    x = torch.randn(1, 128, model_args.dim).cuda().bfloat16()
    x_copy = copy.deepcopy(x)

    y = m(x)

    y_copy = m_copy(x_copy)

    print(torch.allclose(y, y_copy))
    print(SQNR(y, y_copy))

    te_utils.swap_te_friendly_norm_ffn_to_te_layernorm_linear(m_copy)
    print(m)
    y_copy2 = m_copy(x_copy)
    print(torch.allclose(y_copy, y_copy2))
    print(SQNR(y_copy, y_copy2))



def test_norm_ln_ffn_rewrite():
    dim = 256
    hidden_dim = 512
    multiple_of = 1

    x = torch.randn(1, 128, 256).cuda().bfloat16()
    x_copy = copy.deepcopy(x)

    m = NormFFNBlock(dim, hidden_dim, multiple_of).cuda().bfloat16()
    m_copy = copy.deepcopy(m)
    print(m)

    y = m(x)
    y.sum().backward()

    te_utils.swap_norm_ffn_to_te_friendly_norm_ffn(m_copy)
    print(m_copy)

    y_copy = m_copy(x_copy)
    y_copy.sum().backward()

    # TODO: debug why not an exact match
    print(torch.allclose(y, y_copy))
    print(SQNR(y, y_copy))

    # TODO test w13
    # assert torch.allclose(m.ffn.w2.grad, m_copy.ffn.w2.grad, atol=0, rtol=0)

    te_utils.swap_te_friendly_norm_ffn_to_te_layernorm_linear(m_copy)
    print(m_copy)

    y_copy2 = m_copy(x_copy)
    print(torch.allclose(y_copy, y_copy2))
    print(SQNR(y_copy, y_copy2))

def test_norm_mlp_ffn_rewrite():
    dim = 256
    hidden_dim = 512
    multiple_of = 1

    x = torch.randn(1, 128, 256).cuda().bfloat16()
    x_copy = copy.deepcopy(x)

    m = NormFFNBlock(dim, hidden_dim, multiple_of).cuda().bfloat16()
    m_copy = copy.deepcopy(m)
    print(m)

    y = m(x)
    y.sum().backward()

    te_utils.swap_norm_ffn_to_te_friendly_norm_ffn(m_copy)
    print(m_copy)

    y_copy = m_copy(x_copy)
    y_copy.sum().backward()

    # TODO: debug why not an exact match
    print(torch.allclose(y, y_copy))
    print(SQNR(y, y_copy))

    # TODO test w13
    # assert torch.allclose(m.ffn.w2.grad, m_copy.ffn.w2.grad, atol=0, rtol=0)

    te_utils.swap_te_friendly_norm_ffn_to_te_layernorm_mlp(m_copy)
    print(123)
    print(m_copy)

    y_copy2 = m_copy(x_copy)
    print(torch.allclose(y_copy, y_copy2))
    print(SQNR(y_copy, y_copy2))

# works, so a bug in the swap above?
def test_split_linear():
    M, K, N = 32, 64, 128
    # M, K, N = 4, 6, 8

    x = torch.randn(M, K)

    fc1 = nn.Linear(K, N, bias=False)
    fc2 = nn.Linear(K, N, bias=False)

    fc3 = nn.Linear(K, N * 2, bias=False)
    fc3.weight = torch.nn.Parameter(
        torch.cat([copy.deepcopy(fc1.weight), copy.deepcopy(fc2.weight)], dim=0)
    )

    y1 = fc1(x)
    y2 = fc2(x)
    y3 = fc3(x)
    y3_1, y3_2 = torch.split(y3, fc3.out_features // 2, dim=-1)

    assert torch.allclose(y1, y3_1)
    assert torch.allclose(y2, y3_2)


if __name__ == '__main__':
    test()
