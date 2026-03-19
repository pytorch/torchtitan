"""Compare loss computation between torchtitan and HF on identical data.

Isolates the loss function: same model output, same labels.
If the outputs differ, the loss path is the source of divergence.

Usage:
    python scripts/compare_loss_fn.py
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
VOCAB_SIZE = 1000
SEQ_LEN = 200

torch.manual_seed(42)

# Simulate a token sequence: 50 prompt tokens + 150 response tokens + EOS
full_tokens = torch.randint(0, VOCAB_SIZE, (SEQ_LEN + 1,))

# === torchtitan: shift in dataset, model sees pre-shifted data ===
tt_input = full_tokens[:-1]       # [t0, t1, ..., t199], length 200
tt_labels = full_tokens[1:].clone()  # [t1, t2, ..., t200], length 200
tt_labels[:50] = IGNORE_INDEX     # mask prompt

# Fake model output (same logits for both)
torch.manual_seed(123)
logits_200 = torch.randn(1, SEQ_LEN, VOCAB_SIZE, dtype=torch.bfloat16)

tt_loss_sum = F.cross_entropy(
    logits_200.flatten(0, 1).float(),
    tt_labels.flatten(),
    reduction="sum",
    ignore_index=IGNORE_INDEX,
)
tt_valid = (tt_labels != IGNORE_INDEX).sum()
tt_loss = tt_loss_sum / tt_valid.float()

# === HF: no shift in dataset, model shifts internally ===
# HF receives input_ids = full_tokens (length 201)
# HF receives labels = full_tokens with prompt masked (length 201)
hf_labels = full_tokens.clone()       # [t0, t1, ..., t200], length 201
hf_labels[:51] = IGNORE_INDEX         # mask prompt (one more because shift hasn't happened)
# After shift: labels[1:] = [t1_masked, ..., t50_masked, t51, ..., t200]
# Wait -- this is wrong. Let me think about this carefully.

# In HF, the user provides labels = [...] of the same length as input_ids.
# The MODEL shifts: logits = logits[..., :-1, :], labels = labels[..., 1:]
# So if user labels = [MASK, MASK, ..., MASK(50 times), t51, t52, ..., t200]
# After shift: labels becomes [MASK, ..., MASK(49 times), t51, t52, ..., t200]
# And logits become logits for positions 0..199

# To get the SAME effective labels as torchtitan after the model's shift,
# the user-provided labels need the mask shifted back by 1:
# User labels: [MASK]*51 + [t52, t53, ..., t200]  -- 51 masked, 150 response
# After model shift (labels[1:]): [MASK]*50 + [t52, t53, ..., t200] -- same as tt_labels

# But that changes the actual training target. In practice, TRL masks based on
# the prompt TEXT boundary, which is the same character position. The tokenization
# produces the same token count. The shift just means:
# - torchtitan: label[i] = token[i+1], mask applied to label positions 0..49
# - HF: label[i] = token[i] (pre-shift), mask applied to positions 0..50,
#        after model shift: label[i] = token[i+1], mask on positions 0..49
# So they SHOULD be identical.

# Let's test this directly:
# HF-style: labels match input tokens, prompt positions masked
hf_input = full_tokens  # length 201
hf_labels_pre = full_tokens.clone()
hf_labels_pre[:51] = IGNORE_INDEX  # 51 because after shift by 1, positions 0-49 will be masked

# Model shifts internally
hf_logits = torch.randn(1, SEQ_LEN + 1, VOCAB_SIZE, dtype=torch.bfloat16)
# Use the SAME logits as torchtitan for positions 0..199
torch.manual_seed(123)
hf_logits[0, :SEQ_LEN] = torch.randn(SEQ_LEN, VOCAB_SIZE, dtype=torch.bfloat16)

# HF model shift
hf_shift_logits = hf_logits[:, :-1, :]  # [0..199], length 200
hf_shift_labels = hf_labels_pre[1:]      # [1..200], length 200

# Now hf_shift_logits[0] should equal logits_200[0] (same seed)
# And hf_shift_labels should have same mask pattern as tt_labels

print("=== Verify alignment ===")
print(f"logits match: {torch.equal(hf_shift_logits[0], logits_200[0])}")
print(f"labels match: {torch.equal(hf_shift_labels, tt_labels)}")
print(f"tt valid tokens: {tt_valid.item()}")
hf_valid = (hf_shift_labels != IGNORE_INDEX).sum()
print(f"hf valid tokens (post-shift): {hf_valid.item()}")
print()

# HF loss with num_items_in_batch (sum reduction)
hf_num_items = (hf_labels_pre != IGNORE_INDEX).sum()  # counted pre-shift
hf_loss_sum = F.cross_entropy(
    hf_shift_logits.flatten(0, 1).float(),
    hf_shift_labels.flatten(),
    reduction="sum",
    ignore_index=IGNORE_INDEX,
)
hf_loss = hf_loss_sum / hf_num_items.float()

print("=== Loss comparison ===")
print(f"torchtitan: {tt_loss.item():.10f}  (sum={tt_loss_sum.item():.4f} / {tt_valid.item()} tokens)")
print(f"HF/TRL:     {hf_loss.item():.10f}  (sum={hf_loss_sum.item():.4f} / {hf_num_items.item()} tokens)")
print(f"delta:      {abs(tt_loss.item() - hf_loss.item()):.10f}")
print()

# Check if the num_items difference is the cause
print("=== Token count difference ===")
print(f"tt valid (post-shift labels):  {tt_valid.item()}")
print(f"hf valid (pre-shift labels):   {hf_num_items.item()}")
print(f"hf valid (post-shift labels):  {hf_valid.item()}")
if tt_valid.item() != hf_num_items.item():
    print(f"MISMATCH: HF counts {hf_num_items.item()} pre-shift but actual post-shift is {hf_valid.item()}")
    # What if we use the same denominator?
    hf_loss_corrected = hf_loss_sum / hf_valid.float()
    print(f"HF corrected (post-shift count): {hf_loss_corrected.item():.10f}")
    print(f"delta with correction: {abs(tt_loss.item() - hf_loss_corrected.item()):.10f}")
