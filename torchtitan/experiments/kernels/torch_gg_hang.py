# https://github.com/pytorch/pytorch/pull/150374
# NOTE: torch._gouped_mm requires bf16 dtypes
#       and shapes to be multiple of 8


import torch

num_experts = 4
M, K, N = 48, 8, 16

# to repro hang, make a given expert have 0 tokens ala (0, 8, 16, 32, 40) or (8,8,16,32,40)
m_offsets_hang = (8, 8, 32, 40)
m_offsets = (8, 16, 32, 40)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
print(f"{x.shape=}")
w = torch.randn(
    num_experts, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True
)
print(f"{w.shape=}")
offs = torch.tensor(m_offsets, dtype=torch.int32, device="cuda")


print(f"Running simple forward...")
o = torch._grouped_mm(x, w, offs)
print(f"forward completed!")
print(f"Running backward...")
o.backward(torch.randn_like(o))
print(f"backward completed!")
torch.cuda.synchronize()
print(f"Completed! {o.shape=}")
