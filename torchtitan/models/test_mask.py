# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for document masking and variable length sequence metadata.
"""

import torch
from attention import create_varlen_metadata_for_document_using_sequence_indices


def get_document_mask_cumsum(
    input_ids: torch.Tensor,
    doc_lengths: torch.Tensor,  # [B, Num_Docs] padded with -1
    doc_start_indices: torch.Tensor,  # [B, Num_Docs] padded with -1
):
    """
    Generates attention mask ensuring tokens only attend within their specific document.
    Uses cumsum strategy to generate document IDs.
    """
    device = input_ids.device
    B, S = input_ids.shape[:2]
    document_ids = torch.zeros((B, S), dtype=torch.int32, device=device)
    valid_entries = (doc_start_indices >= 0) & (doc_lengths > 0)
    batch_rows = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand_as(doc_start_indices)[valid_entries]
    )
    start_cols = doc_start_indices[valid_entries].long().clamp(0, S - 1)

    document_ids[batch_rows, start_cols] = 1
    seq_ids = document_ids.cumsum(dim=1)

    doc_ends = doc_start_indices + doc_lengths
    last_valid_end = torch.max(torch.where(valid_entries, doc_ends, -1), dim=1).values

    pos_indices = torch.arange(S, device=device).unsqueeze(0)
    is_padding = pos_indices >= last_valid_end.unsqueeze(1)

    seq_ids = seq_ids.masked_fill(is_padding, -1)

    def document_mask(b, h, q_idx, kv_idx):
        q = seq_ids[b, q_idx]
        k = seq_ids[b, kv_idx]
        return (q == k) & (q > 0)

    return document_mask, seq_ids


# ----- Ground Truth Generators -----
def generate_brute_segment_ids(
    doc_start_indices: torch.Tensor, doc_lengths: torch.Tensor, max_seq_len: int
):
    device = doc_start_indices.device
    B = doc_start_indices.shape[0]

    # Init with -1 (no document / padding)
    gt_map = torch.full((B, max_seq_len), -1, dtype=torch.int32, device=device)

    for b in range(B):
        curr_doc_id = 1
        for j in range(doc_start_indices.shape[1]):
            start = int(doc_start_indices[b, j].item())
            length = int(doc_lengths[b, j].item())

            if start < 0 or length <= 0:
                continue

            end = min(start + length, max_seq_len)
            gt_map[b, start:end] = curr_doc_id
            curr_doc_id += 1

    return gt_map


def generate_brute_varlen_metadata(input_ids, doc_start_indices, doc_lengths):
    """
    Reference implementation for cu_seqlens.
    Flattens the batch to 1D to mimic FlashAttention layout and finds cut points.
    """
    B, S = input_ids.shape[:2]
    device = input_ids.device

    cut_points = []

    # 1. Hard boundaries at batch transitions
    for b in range(B + 1):
        cut_points.append(b * S)

    # 2. Boundaries defined by document start/ends
    for b in range(B):
        offset = b * S
        for i in range(doc_start_indices.shape[1]):
            s = doc_start_indices[b, i].item()
            l = doc_lengths[b, i].item()
            if s != -1:
                cut_points.append(offset + s)
                cut_points.append(offset + s + l)

    # 3. Deduplicate and sort to get cumulative sequence lengths
    t_points = torch.tensor(sorted(cut_points), device=device, dtype=torch.int32)
    cu_seqlens = torch.unique_consecutive(t_points)

    # 4. Calc max seq len from diffs
    diffs = torch.diff(cu_seqlens)
    max_len = diffs.max().item() if diffs.numel() > 0 else 0

    return cu_seqlens, max_len


# ----- Data Mocking Builders -----
def mock_right_padded_row(
    valid_len: int, max_seq_len: int, pad_id: int = 0, device="cpu"
):
    """
    Creates a single row mimicking standard padding: [Data, Data, Pad, Pad]
    """
    # Create tokens
    x = torch.full((max_seq_len,), pad_id, dtype=torch.long, device=device)
    x[:valid_len] = torch.arange(valid_len, device=device) + 10

    starts = torch.full((max_seq_len,), -1, dtype=torch.int32, device=device)
    lens = torch.full((max_seq_len,), -1, dtype=torch.int32, device=device)

    starts[0] = 0
    lens[0] = valid_len

    return x, starts, lens


def mock_packed_row(
    seg_lens: list[int], max_seq_len: int, pad_id: int = 0, device="cpu"
):
    """
    Creates a single row mimicking sample packing: [Doc A, Doc B, Doc C, Pad]
    """
    total_len = sum(seg_lens)
    assert total_len <= max_seq_len, "Packed segments overflow buffer"

    x = torch.full((max_seq_len,), pad_id, dtype=torch.long, device=device)
    x[:total_len] = torch.arange(total_len, device=device) + 100

    lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    if lens_t.numel() == 0:
        starts_t = torch.zeros((0,), dtype=torch.int32, device=device)
    else:
        starts_t = torch.cat([lens_t.new_zeros(1), lens_t.cumsum(0)[:-1]])

    pad_amt = max_seq_len - lens_t.numel()
    starts_t = torch.nn.functional.pad(starts_t, (0, pad_amt), value=-1)
    lens_t = torch.nn.functional.pad(lens_t, (0, pad_amt), value=-1)

    return x, starts_t, lens_t


# ----- Test Runners -----
def verify_varlen_metadata(name, input_ids, doc_start_indices, doc_lengths):
    meta = create_varlen_metadata_for_document_using_sequence_indices(
        input_ids, doc_lengths, doc_start_indices
    )

    gt_cu, gt_max = generate_brute_varlen_metadata(
        input_ids, doc_start_indices, doc_lengths
    )

    assert torch.equal(meta.cu_seq_q, gt_cu), "cu_seq_q mismatch"
    # q and k are identical in self-attention context
    assert torch.equal(meta.cu_seq_k, gt_cu), "cu_seq_k mismatch"
    assert meta.max_q == gt_max, f"max_q mismatch: got {meta.max_q}, expected {gt_max}"

    print(f"[PASS] [VARLEN] - {name}")


def verify_mask_logic(
    name, input_ids, doc_start_indices, doc_lengths, n_random_checks=5000
):
    device = input_ids.device
    B, S = input_ids.shape[:2]

    # Get mask function and the internal ID map
    mask_fn, seq_ids = get_document_mask_cumsum(
        input_ids, doc_lengths, doc_start_indices
    )

    # Generate Truth
    gt_ids = generate_brute_segment_ids(doc_start_indices, doc_lengths, S)

    # Normalize seq_ids for comparison (treat 0 as -1 to match GT format)
    seq_ids_norm = seq_ids.clone()
    seq_ids_norm[seq_ids_norm == 0] = -1

    if not torch.equal(seq_ids_norm, gt_ids):
        print(f"[{name}] Map Mismatch!")
        print(f"Calculated:\n{seq_ids_norm}")
        print(f"Ground Truth:\n{gt_ids}")
        raise AssertionError(f"[{name}] Sequence ID map mismatch")

    b_idx = torch.randint(0, B, (n_random_checks,), device=device)
    q_idx = torch.randint(0, S, (n_random_checks,), device=device)
    k_idx = torch.randint(0, S, (n_random_checks,), device=device)
    h_idx = torch.zeros_like(b_idx)  # Head index unused here

    # Calculate attention allowance
    calculated_allow = mask_fn(b_idx, h_idx, q_idx, k_idx)

    # Truth: Indices match AND are not padding (-1)
    truth_q = gt_ids[b_idx, q_idx]
    truth_k = gt_ids[b_idx, k_idx]
    expected_allow = (truth_q == truth_k) & (truth_q > 0)

    assert torch.equal(
        calculated_allow, expected_allow
    ), f"[{name}] Mask function logic failure"
    print(f"[PASS] [FLEX] - {name}")


def run_suite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_id = 9999

    print(f"Running tests on: {device}")

    T = 32
    rows_padding = [
        mock_right_padded_row(valid_len=5, max_seq_len=T, pad_id=pad_id, device=device),
        mock_right_padded_row(
            valid_len=17, max_seq_len=T, pad_id=pad_id, device=device
        ),
        mock_right_padded_row(
            valid_len=31, max_seq_len=T, pad_id=pad_id, device=device
        ),
        mock_right_padded_row(valid_len=T, max_seq_len=T, pad_id=pad_id, device=device),
    ]
    batch_pad = torch.stack([r[0] for r in rows_padding])
    starts_pad = torch.stack([r[1] for r in rows_padding])
    lens_pad = torch.stack([r[2] for r in rows_padding])

    verify_mask_logic("Right Padding", batch_pad, starts_pad, lens_pad)
    verify_varlen_metadata("Right Padding", batch_pad, starts_pad, lens_pad)

    print("\nAll * Right Padding * passed successfully.")

    T_pack = 64
    rows_packed = [
        mock_packed_row([10, 7, 12], max_seq_len=T_pack, pad_id=pad_id, device=device),
        mock_packed_row(
            [5, 5, 5, 5, 5], max_seq_len=T_pack, pad_id=pad_id, device=device
        ),
        mock_packed_row([33], max_seq_len=T_pack, pad_id=pad_id, device=device),
        mock_packed_row(
            [1, 2, 3, 4, 5, 6], max_seq_len=T_pack, pad_id=pad_id, device=device
        ),
    ]
    batch_pack = torch.stack([r[0] for r in rows_packed])
    starts_pack = torch.stack([r[1] for r in rows_packed])
    lens_pack = torch.stack([r[2] for r in rows_packed])

    verify_mask_logic("Greedy Packing", batch_pack, starts_pack, lens_pack)
    verify_varlen_metadata("Greedy Packing", batch_pack, starts_pack, lens_pack)

    print("\nAll * Greedy Packing * passed successfully.")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_suite()
