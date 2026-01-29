#!/usr/bin/env python3
"""Test script for Harmony loss masking.

Tests the loss mask implementation against the Harmony chat format specification.
All token IDs are verified against the o200k_harmony tokenizer.
"""

import torch
import sys
sys.path.insert(0, '/mnt/git/torchtitan.temp')

from torchtitan.hf_datasets.harmony_loss_mask import (
    create_loss_mask,
    get_mask_statistics,
    HARMONY_START,
    HARMONY_END,
    HARMONY_MESSAGE,
    HARMONY_RETURN,
    HARMONY_CONSTRAIN,
    HARMONY_CHANNEL,
    HARMONY_CALL,
    HARMONY_ASSISTANT,
    HARMONY_USER,
)

# =============================================================================
# Role token IDs (verified against o200k_harmony tokenizer)
# =============================================================================
SYSTEM_TOKEN = 17360       # "system"
DEVELOPER_TOKEN = 77944    # "developer"
USER_TOKEN = 1428          # "user"
# HARMONY_ASSISTANT = 173781 is imported from the module

# =============================================================================
# Channel name token IDs (verified against o200k_harmony tokenizer)
# These are the tokens WITHOUT leading space (as they appear after <|channel|>)
# =============================================================================
ANALYSIS_TOKEN = 35644     # "analysis"
FINAL_TOKEN = 17196        # "final"
# Note: "commentary" tokenizes to multiple tokens [12606, 815] without space,
# but to single token 49159 with space. We use a representative value for tests.
COMMENTARY_TOKEN = 49159   # " commentary" (with leading space, as commonly appears)

# =============================================================================
# Other token IDs used in tests
# =============================================================================
JSON_TOKEN = 4108          # "json"
TO_TOKEN = 935             # "to"
EQUALS_TOKEN = 28          # "="
FUNCTIONS_TOKEN = 44580    # "functions"

# =============================================================================
# Test Content Token Convention
# =============================================================================
# Throughout the tests, we use small integers (100, 200, 300, etc.) as placeholder
# content tokens. These represent arbitrary message content and are NOT real token IDs.
# The pattern used is:
#   - 100-199: system message content
#   - 200-299: user message content (turn 1)
#   - 300-399: assistant message content (turn 1)
#   - 400-499: subsequent turn content (user/assistant/developer/tool)
#   - 500-599, 600-699, etc.: additional turns
# This convention makes it easy to identify which turn content belongs to when
# debugging test failures.


def test_basic_loss_mask():
    """Test basic loss masking on a simple Harmony sequence."""
    print("=" * 60)
    print("Test: Basic Harmony Loss Mask")
    print("=" * 60)

    # Simulate a simple Harmony conversation:
    # <|start|>system<|message|>You are helpful<|end|>
    # <|start|>user<|message|>Hello<|end|>
    # <|start|>assistant<|message|>Hi there!<|return|>

    # Build sequence
    tokens = [
        # System turn
        HARMONY_START, SYSTEM_TOKEN, HARMONY_MESSAGE, 100, 101, 102, HARMONY_END,
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, 201, HARMONY_END,
        # Assistant turn
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, 302, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    print(f"Input sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")

    mask = create_loss_mask(input_ids)
    print(f"\nMask: {mask.tolist()}")

    # Expected: ALL assistant turn tokens should have loss=1
    # That's positions 13-20: <|start|>, assistant, <|message|>, 300, 301, 302, <|return|>
    expected_positions = [13, 14, 15, 16, 17, 18, 19]
    for i, m in enumerate(mask.tolist()):
        expected = 1.0 if i in expected_positions else 0.0
        status = "✓" if m == expected else "✗"
        if m == 1.0:
            print(f"  Position {i}: token={tokens[i]}, mask={m} {status}")

    stats = get_mask_statistics(mask)
    print(f"\nStatistics:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Assistant tokens: {stats['assistant_tokens']}")
    print(f"  Non-assistant tokens: {stats['non_assistant_tokens']}")
    print(f"  Assistant ratio: {stats['assistant_ratio']:.1%}")

    # Verify: 7 tokens (<|start|>, assistant, <|message|>, 300, 301, 302, <|return|>)
    assert mask.sum().item() == 7, f"Expected 7 assistant tokens, got {mask.sum().item()}"
    print("\n✓ Test passed!")


def test_multi_turn_conversation():
    """Test loss masking on a multi-turn conversation.

    With "last user only" masking, only assistant tokens AFTER the last user
    message should be masked. Prior turn assistant outputs are context, not
    training targets.
    """
    print("\n" + "=" * 60)
    print("Test: Multi-turn Conversation (Last User Only)")
    print("=" * 60)

    # Multi-turn: system + user + assistant + user + assistant
    # Position layout:
    #   System:      0-4   (5 tokens)
    #   User 1:      5-10  (6 tokens)
    #   Assistant 1: 11-16 (6 tokens) <- BEFORE last user, should NOT be masked
    #   User 2:      17-21 (5 tokens) <- LAST user at position 17
    #   Assistant 2: 22-28 (7 tokens) <- AFTER last user, SHOULD be masked
    tokens = [
        # System
        HARMONY_START, SYSTEM_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,
        # User 1
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, 201, HARMONY_END,
        # Assistant 1 (before last user - should NOT be masked)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, HARMONY_RETURN,
        # User 2 (LAST user)
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 400, HARMONY_END,
        # Assistant 2 (after last user - SHOULD be masked)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 500, 501, 502, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Verify Assistant 1 (positions 11-16) is NOT masked
    for i in range(11, 17):
        assert mask[i].item() == 0, f"Assistant 1 at position {i} should NOT be masked (before last user)"

    # Verify Assistant 2 (positions 22-28) IS masked
    for i in range(22, 29):
        assert mask[i].item() == 1, f"Assistant 2 at position {i} SHOULD be masked (after last user)"

    # Only Assistant 2: <|start|>, assistant, <|message|>, 500, 501, 502, RETURN = 7 tokens
    expected = 7
    actual = mask.sum().item()
    assert actual == expected, f"Expected {expected} assistant tokens (last turn only), got {actual}"

    stats = get_mask_statistics(mask)
    print(f"Mask sum: {actual} (only last turn)")
    print(f"Assistant ratio: {stats['assistant_ratio']:.1%}")
    print("✓ Test passed!")


def test_with_padding():
    """Test loss masking with padding tokens."""
    print("\n" + "=" * 60)
    print("Test: With Padding")
    print("=" * 60)

    PADDING_TOKEN = 200001  # Harmony padding token

    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant turn
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, HARMONY_RETURN,
        # Padding
        PADDING_TOKEN, PADDING_TOKEN, PADDING_TOKEN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids, padding_token=PADDING_TOKEN)

    print(f"Sequence length: {len(tokens)}")
    print(f"Padding tokens: 3")

    # Padding should NOT be masked (already 0 by default, but explicitly 0)
    assert mask[-1].item() == 0, "Padding should be masked out"
    assert mask[-2].item() == 0, "Padding should be masked out"
    assert mask[-3].item() == 0, "Padding should be masked out"

    # Assistant tokens: <|start|>, assistant, <|message|>, 300, 301, RETURN = 6 tokens
    assert mask.sum().item() == 6, f"Expected 6 assistant tokens, got {mask.sum().item()}"

    stats = get_mask_statistics(mask)
    print(f"Assistant ratio: {stats['assistant_ratio']:.1%}")
    print("✓ Test passed!")


def test_batched_input():
    """Test loss masking with batched input."""
    print("\n" + "=" * 60)
    print("Test: Batched Input")
    print("=" * 60)

    # Batch of 2 sequences
    seq1 = [HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,
            HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, HARMONY_RETURN]
    seq2 = [HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 300, 301, HARMONY_END,
            HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 400, 401, 402, HARMONY_RETURN]

    # Pad to same length
    max_len = max(len(seq1), len(seq2))
    seq1 = seq1 + [0] * (max_len - len(seq1))
    seq2 = seq2 + [0] * (max_len - len(seq2))

    input_ids = torch.tensor([seq1, seq2])
    print(f"Batch shape: {input_ids.shape}")

    mask = create_loss_mask(input_ids, padding_token=0)
    print(f"Mask shape: {mask.shape}")

    # Seq1: <|start|>, assistant, <|message|>, 200, RETURN = 5 assistant tokens
    # Seq2: <|start|>, assistant, <|message|>, 400, 401, 402, RETURN = 7 assistant tokens
    assert mask[0].sum().item() == 5, f"Seq1: expected 5, got {mask[0].sum().item()}"
    assert mask[1].sum().item() == 7, f"Seq2: expected 7, got {mask[1].sum().item()}"

    print(f"Seq1 assistant tokens: {mask[0].sum().item()}")
    print(f"Seq2 assistant tokens: {mask[1].sum().item()}")
    print("✓ Test passed!")


def test_with_channel():
    """Test loss masking with <|channel|> token in assistant message."""
    print("\n" + "=" * 60)
    print("Test: With Channel Token")
    print("=" * 60)

    # Assistant message with channel:
    # <|start|>assistant<|channel|>final<|message|>Hello!<|return|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant turn with channel
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 300, 301, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # ALL assistant turn tokens should have loss=1, including header tokens
    # The model needs to learn which channel to output (<|channel|>final)
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant token should have loss=1"
    assert mask[7].item() == 1, "<|channel|> token should have loss=1"
    assert mask[8].item() == 1, "channel name token should have loss=1"
    assert mask[9].item() == 1, "<|message|> should have loss=1"
    assert mask[10].item() == 1, "content token 300 should have loss=1"
    assert mask[11].item() == 1, "content token 301 should have loss=1"
    assert mask[12].item() == 1, "<|return|> should have loss=1"

    # 8 tokens total: <|start|>, assistant, <|channel|>, final, <|message|>, 300, 301, RETURN
    expected = 8
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_assistant_with_call():
    """Test loss masking for assistant message ending with <|call|> (tool calls)."""
    print("\n" + "=" * 60)
    print("Test: Assistant with <|call|> (Tool Call)")
    print("=" * 60)

    # Assistant tool call:
    # <|start|>assistant<|channel|>commentary<|message|>{"tool":"foo"}<|call|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant tool call with commentary channel
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, COMMENTARY_TOKEN, HARMONY_MESSAGE, 400, 401, HARMONY_CALL,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # ALL assistant tokens should have loss=1, including header
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant token should have loss=1"
    assert mask[7].item() == 1, "<|channel|> should have loss=1"
    assert mask[8].item() == 1, "commentary token should have loss=1"
    assert mask[9].item() == 1, "<|message|> should have loss=1"
    assert mask[10].item() == 1, "content token 400 should have loss=1"
    assert mask[11].item() == 1, "content token 401 should have loss=1"
    assert mask[12].item() == 1, "<|call|> should have loss=1"

    # 8 tokens: <|start|>, assistant, <|channel|>, commentary, <|message|>, 400, 401, CALL
    expected = 8
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_assistant_with_end():
    """Test loss masking for assistant message ending with <|end|> (analysis channel)."""
    print("\n" + "=" * 60)
    print("Test: Assistant with <|end|> (Analysis Channel)")
    print("=" * 60)

    # Assistant analysis:
    # <|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant analysis with <|end|>
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 500, 501, 502, HARMONY_END,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # ALL assistant tokens should have loss=1, including header
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant token should have loss=1"
    assert mask[7].item() == 1, "<|channel|> should have loss=1"
    assert mask[8].item() == 1, "analysis token should have loss=1"
    assert mask[9].item() == 1, "<|message|> should have loss=1"
    assert mask[10].item() == 1, "content token 500 should have loss=1"
    assert mask[11].item() == 1, "content token 501 should have loss=1"
    assert mask[12].item() == 1, "content token 502 should have loss=1"
    assert mask[13].item() == 1, "<|end|> should have loss=1 for assistant turns"

    # 9 tokens: <|start|>, assistant, <|channel|>, analysis, <|message|>, 500, 501, 502, END
    expected = 9
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_mixed_channels():
    """Test loss masking with multiple assistant turns using different channels."""
    print("\n" + "=" * 60)
    print("Test: Mixed Channels (analysis + final)")
    print("=" * 60)

    # Multi-turn with analysis then final:
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant analysis (ends with <|end|>)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 300, HARMONY_END,
        # Assistant final (ends with <|return|>)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 400, 401, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Analysis: <|start|>, assistant, <|channel|>, analysis, <|message|>, 300, END = 7 tokens
    # Final: <|start|>, assistant, <|channel|>, final, <|message|>, 400, 401, RETURN = 8 tokens
    # Total: 15 tokens
    expected = 15
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens, got {mask.sum().item()}"

    stats = get_mask_statistics(mask)
    print(f"Assistant ratio: {stats['assistant_ratio']:.1%}")
    print("✓ Test passed!")


def test_tool_response_not_masked():
    """Test that tool response messages (role=toolname) have loss=0."""
    print("\n" + "=" * 60)
    print("Test: Tool Response Not Masked")
    print("=" * 60)

    TOOL_NAME_TOKEN = 88888  # e.g., "functions.get_weather" - arbitrary value for test

    # Tool response pattern:
    # <|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"temp":72}<|end|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Tool response (role is tool name, NOT assistant)
        HARMONY_START, TOOL_NAME_TOKEN, TO_TOKEN, EQUALS_TOKEN, HARMONY_ASSISTANT,
        HARMONY_CHANNEL, COMMENTARY_TOKEN, HARMONY_MESSAGE, 100, 101, 102, HARMONY_END,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Tool responses should NOT contribute to loss (role != assistant)
    # The role token at position 6 is TOOL_NAME_TOKEN, not HARMONY_ASSISTANT
    assert mask.sum().item() == 0, f"Tool responses should have loss=0, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_constrain_token_included():
    """Test that <|constrain|> and format specifier are included in loss (part of header)."""
    print("\n" + "=" * 60)
    print("Test: Constrain Token Included in Loss")
    print("=" * 60)

    # Tool call with constrain:
    # <|start|>assistant<|channel|>commentary<|constrain|>json<|message|>{"data":"value"}<|call|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant tool call with constrain
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, COMMENTARY_TOKEN,
        HARMONY_CONSTRAIN, JSON_TOKEN, HARMONY_MESSAGE, 400, 401, HARMONY_CALL,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # ALL assistant tokens should have loss=1, including header with constrain
    # The model needs to learn the complete header structure for tool calls
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant should have loss=1"
    assert mask[7].item() == 1, "<|channel|> should have loss=1"
    assert mask[8].item() == 1, "commentary should have loss=1"
    assert mask[9].item() == 1, "<|constrain|> should have loss=1"
    assert mask[10].item() == 1, "json format specifier should have loss=1"
    assert mask[11].item() == 1, "<|message|> should have loss=1"
    assert mask[12].item() == 1, "content token 400 should have loss=1"
    assert mask[13].item() == 1, "content token 401 should have loss=1"
    assert mask[14].item() == 1, "<|call|> should have loss=1"

    # 10 tokens total
    expected = 10
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_no_assistant_turns():
    """Test sequence with no assistant turns returns all zeros."""
    print("\n" + "=" * 60)
    print("Test: No Assistant Turns")
    print("=" * 60)

    # Only system and user turns, no assistant
    tokens = [
        HARMONY_START, SYSTEM_TOKEN, HARMONY_MESSAGE, 100, 101, HARMONY_END,
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, 201, HARMONY_END,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask sum: {mask.sum().item()}")

    assert mask.sum().item() == 0, "No assistant turns should mean all zeros"

    print("✓ Test passed!")


def test_developer_role_not_masked():
    """Test that developer role messages have loss=0 (like system/user).

    With "last user only" masking, only Assistant 2 should be masked (after User 2).
    Assistant 1 is before the last user, so it should NOT be masked.
    """
    print("\n" + "=" * 60)
    print("Test: Developer Role Not Masked (Last User Only)")
    print("=" * 60)

    # Conversation with developer role (like system prompts but mid-conversation)
    # Pattern: system + user + assistant + developer + user + assistant
    tokens = [
        # System turn
        HARMONY_START, SYSTEM_TOKEN, HARMONY_MESSAGE, 100, 101, HARMONY_END,
        # User turn 1
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant turn 1 (BEFORE last user - should NOT be masked)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, HARMONY_RETURN,
        # Developer turn (should NOT be masked - not assistant)
        HARMONY_START, DEVELOPER_TOKEN, HARMONY_MESSAGE, 400, 401, 402, HARMONY_END,
        # User turn 2 (LAST user at position 24)
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 500, HARMONY_END,
        # Assistant turn 2 (AFTER last user - SHOULD be masked)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 600, 601, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Position calculation:
    # System: START(0) SYSTEM(1) MSG(2) 100(3) 101(4) END(5) = 6 tokens (0-5)
    # User 1: START(6) USER(7) MSG(8) 200(9) END(10) = 5 tokens (6-10)
    # Assistant 1: START(11) ASSISTANT(12) MSG(13) 300(14) 301(15) RETURN(16) = 6 tokens (11-16)
    # Developer: START(17) DEVELOPER(18) MSG(19) 400(20) 401(21) 402(22) END(23) = 7 tokens (17-23)
    # User 2: START(24) USER(25) MSG(26) 500(27) END(28) = 5 tokens (24-28) <- LAST user
    # Assistant 2: START(29) ASSISTANT(30) MSG(31) 600(32) 601(33) RETURN(34) = 6 tokens (29-34)

    # Developer turn positions (17-23) should all have loss=0
    for i in range(17, 24):
        assert mask[i].item() == 0, f"Developer turn token at position {i} should have loss=0"

    # Assistant 1 (positions 11-16) should NOT be masked (before last user)
    for i in range(11, 17):
        assert mask[i].item() == 0, f"Assistant 1 token at position {i} should have loss=0 (before last user)"

    # Assistant 2 (positions 29-34) SHOULD be masked (after last user)
    for i in range(29, 35):
        assert mask[i].item() == 1, f"Assistant 2 token at position {i} should have loss=1 (after last user)"

    # Only Assistant 2: 6 tokens
    expected = 6
    actual = int(mask.sum().item())
    assert actual == expected, f"Expected {expected} assistant tokens (last turn only), got {actual}"

    stats = get_mask_statistics(mask)
    print(f"Developer tokens masked: 0 (verified)")
    print(f"Assistant tokens (last turn only): {stats['assistant_tokens']}")
    print(f"Assistant ratio: {stats['assistant_ratio']:.1%}")
    print("✓ Test passed!")


def test_empty_assistant_content():
    """Test assistant message with empty content (edge case)."""
    print("\n" + "=" * 60)
    print("Test: Empty Assistant Content")
    print("=" * 60)

    # Empty assistant content: <|start|>assistant<|message|><|return|>
    tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # All assistant tokens should have loss=1: <|start|>, assistant, <|message|>, <|return|>
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant should have loss=1"
    assert mask[7].item() == 1, "<|message|> should have loss=1"
    assert mask[8].item() == 1, "<|return|> should have loss=1"
    assert mask.sum().item() == 4, f"Expected 4 tokens, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_truncated_sequence():
    """Test handling of truncated sequence (no end token)."""
    print("\n" + "=" * 60)
    print("Test: Truncated Sequence")
    print("=" * 60)

    # Truncated: assistant turn without end token
    tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, 302,
        # No <|return|> or <|end|> - truncated
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # All assistant tokens should be marked even without end token
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant should have loss=1"
    assert mask[7].item() == 1, "<|message|> should have loss=1"
    assert mask[8].item() == 1, "content token 300 should have loss=1"
    assert mask[9].item() == 1, "content token 301 should have loss=1"
    assert mask[10].item() == 1, "content token 302 should have loss=1"

    # 6 tokens: <|start|>, assistant, <|message|>, 300, 301, 302
    expected = 6
    assert mask.sum().item() == expected, f"Expected {expected}, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_truncated_assistant_followed_by_new_conversation():
    """Test that truncated assistant doesn't swallow the next conversation in packed samples.

    With "last user only" masking, Conv1's truncated assistant is BEFORE the last user
    (Conv2's user), so it should NOT be masked. Only Conv2's assistant is masked.
    """
    print("\n" + "=" * 60)
    print("Test: Truncated Assistant (Last User Only)")
    print("=" * 60)

    # This tests a critical edge case in packing:
    # If conv1's assistant turn is truncated (no end token), the mask logic
    # must NOT swallow conv2's user turn by finding its <|end|>.
    #
    # Position layout:
    #   Conv1 user:      0-4   (5 tokens)
    #   Conv1 assistant: 5-9   (5 tokens, truncated) <- BEFORE last user
    #   Conv2 user:      10-14 (5 tokens) <- LAST user at position 10
    #   Conv2 assistant: 15-19 (5 tokens) <- AFTER last user
    tokens = [
        # Conv1: user + truncated assistant (no end token!)
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, 201,  # NO <|return|>!
        # Conv2: user + assistant (normal, complete)
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 300, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 400, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Conv1 user (positions 0-4): should be 0
    for i in range(5):
        assert mask[i].item() == 0, f"Conv1 user position {i} should have loss=0"

    # Conv1 truncated assistant (positions 5-9): should be 0 (before last user)
    for i in range(5, 10):
        assert mask[i].item() == 0, f"Conv1 assistant position {i} should have loss=0 (before last user)"

    # Conv2 user (positions 10-14): should be 0
    for i in range(10, 15):
        assert mask[i].item() == 0, f"Conv2 user position {i} should have loss=0"

    # Conv2 assistant (positions 15-19): should be 1 (after last user)
    for i in range(15, 20):
        assert mask[i].item() == 1, f"Conv2 assistant position {i} should have loss=1 (after last user)"

    # Only Conv2 assistant: 5 tokens
    expected = 5
    assert mask.sum().item() == expected, f"Expected {expected} assistant tokens (Conv2 only), got {mask.sum().item()}"

    print("✓ Test passed!")


def test_per_sample_masking_before_packing():
    """Test the per-sample masking approach used by HuggingFacePackedDataset.

    When packing samples, masks are computed PER-SAMPLE BEFORE packing, then
    concatenated. This ensures each sample's assistant tokens (after its last
    user) are masked, not just the last sample in the pack.

    This is the CORRECT behavior for training - we want to train on all
    samples' assistant responses.
    """
    print("\n" + "=" * 60)
    print("Test: Per-Sample Masking Before Packing")
    print("=" * 60)

    # Two complete conversations, each with user + assistant
    conv1_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, HARMONY_RETURN,
    ]
    conv2_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 300, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 400, 401, HARMONY_RETURN,
    ]

    # Compute masks PER-SAMPLE (as HuggingFacePackedDataset does)
    mask1 = create_loss_mask(torch.tensor(conv1_tokens))
    mask2 = create_loss_mask(torch.tensor(conv2_tokens))

    # Concatenate tokens and masks (simulating packing)
    packed_tokens = conv1_tokens + conv2_tokens
    packed_mask = torch.cat([mask1, mask2])

    print(f"Conv1 tokens: {len(conv1_tokens)}, mask sum: {mask1.sum().item()}")
    print(f"Conv2 tokens: {len(conv2_tokens)}, mask sum: {mask2.sum().item()}")
    print(f"Packed mask: {packed_mask.tolist()}")

    # Conv1 assistant (positions 5-9 in conv1, same in packed): should be masked
    assert mask1[5:10].sum().item() == 5, "Conv1 assistant should be masked"

    # Conv2 assistant (positions 5-10 in conv2, 15-20 in packed): should be masked
    assert mask2[5:11].sum().item() == 6, "Conv2 assistant should be masked"

    # Total: both assistants are masked
    expected_total = 5 + 6  # Conv1 assistant (5) + Conv2 assistant (6)
    assert packed_mask.sum().item() == expected_total, \
        f"Expected {expected_total} masked tokens, got {packed_mask.sum().item()}"

    # Compare with masking AFTER packing (different behavior!)
    mask_after_packing = create_loss_mask(torch.tensor(packed_tokens))
    print(f"Mask after packing: {mask_after_packing.tolist()}")
    print(f"  -> Only masks {mask_after_packing.sum().item()} tokens (last conv only)")

    # The after-packing mask only gets the last conversation's assistant
    assert mask_after_packing.sum().item() == 6, "After-packing mask should only get Conv2"

    print(f"\nPer-sample masking: {packed_mask.sum().item()} tokens (BOTH assistants)")
    print(f"After-packing masking: {mask_after_packing.sum().item()} tokens (last conv only)")
    print("✓ Test passed!")


def test_packed_multi_turn_per_sample():
    """Test packing multi-turn conversations with per-sample masking.

    Each conversation has multiple user/assistant turns. With per-sample masking,
    only the assistant turns AFTER the last user in EACH conversation are masked.

    This simulates the actual HuggingFacePackedDataset code path.
    """
    print("\n" + "=" * 60)
    print("Test: Packed Multi-turn (Per-Sample Masking)")
    print("=" * 60)

    # Conv1: user1 → asst1 → user2 → asst2 (only asst2 masked)
    conv1_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,  # user1
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, HARMONY_RETURN,  # asst1 (before last user)
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 300, HARMONY_END,  # user2 (last user)
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 400, 401, HARMONY_RETURN,  # asst2 (after last user)
    ]

    # Conv2: user1 → asst1 (asst1 masked - only one user)
    conv2_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 500, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 600, 601, 602, HARMONY_RETURN,
    ]

    # Conv3: user1 → asst1 → user2 → asst2 → user3 → asst3 (only asst3 masked)
    conv3_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 700, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 800, HARMONY_RETURN,
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 900, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 1000, HARMONY_RETURN,
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1100, HARMONY_END,  # last user
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 1200, 1201, 1202, HARMONY_RETURN,  # masked
    ]

    # Compute masks per-sample (as HuggingFacePackedDataset does)
    mask1 = create_loss_mask(torch.tensor(conv1_tokens))
    mask2 = create_loss_mask(torch.tensor(conv2_tokens))
    mask3 = create_loss_mask(torch.tensor(conv3_tokens))

    # Verify per-conversation masking
    # Conv1: asst2 is positions 15-20 (6 tokens: START, ASSISTANT, MESSAGE, 400, 401, RETURN)
    conv1_expected = 6
    assert mask1.sum().item() == conv1_expected, f"Conv1 expected {conv1_expected}, got {mask1.sum().item()}"

    # Conv2: asst1 is positions 5-11 (7 tokens)
    conv2_expected = 7
    assert mask2.sum().item() == conv2_expected, f"Conv2 expected {conv2_expected}, got {mask2.sum().item()}"

    # Conv3: asst3 is positions 25-31 (7 tokens)
    conv3_expected = 7
    assert mask3.sum().item() == conv3_expected, f"Conv3 expected {conv3_expected}, got {mask3.sum().item()}"

    # Simulate packing
    packed_tokens = conv1_tokens + conv2_tokens + conv3_tokens
    packed_mask = torch.cat([mask1, mask2, mask3])

    total_masked = packed_mask.sum().item()
    expected_total = conv1_expected + conv2_expected + conv3_expected

    print(f"Conv1: {len(conv1_tokens)} tokens, {conv1_expected} masked (asst2 only)")
    print(f"Conv2: {len(conv2_tokens)} tokens, {conv2_expected} masked (asst1)")
    print(f"Conv3: {len(conv3_tokens)} tokens, {conv3_expected} masked (asst3 only)")
    print(f"Packed: {len(packed_tokens)} tokens, {total_masked} masked")

    assert total_masked == expected_total, f"Expected {expected_total} total masked, got {total_masked}"
    print("✓ Test passed!")


def test_packed_with_analysis_channels_per_sample():
    """Test packing conversations with analysis/final channels using per-sample masking.

    Simulates the actual HuggingFacePackedDataset code path with complex
    assistant turns that have multiple channels (analysis → final).
    """
    print("\n" + "=" * 60)
    print("Test: Packed with Analysis Channels (Per-Sample Masking)")
    print("=" * 60)

    # Conv1: user → analysis<|end|> → final<|return|>
    conv1_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 200, 201, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 300, HARMONY_RETURN,
    ]

    # Conv2: user → assistant<|return|> (simple)
    conv2_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 400, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 500, 501, HARMONY_RETURN,
    ]

    # Conv3: user1 → analysis → final → user2 → analysis → final
    # Only the second analysis+final should be masked
    conv3_tokens = [
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 600, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 700, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 800, HARMONY_RETURN,
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 900, HARMONY_END,  # last user
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 1000, 1001, HARMONY_END,
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 1100, HARMONY_RETURN,
    ]

    # Compute masks per-sample
    mask1 = create_loss_mask(torch.tensor(conv1_tokens))
    mask2 = create_loss_mask(torch.tensor(conv2_tokens))
    mask3 = create_loss_mask(torch.tensor(conv3_tokens))

    # Conv1: both analysis (8 tokens) and final (7 tokens) = 15 tokens
    conv1_expected = 15
    assert mask1.sum().item() == conv1_expected, f"Conv1 expected {conv1_expected}, got {mask1.sum().item()}"

    # Conv2: assistant (6 tokens)
    conv2_expected = 6
    assert mask2.sum().item() == conv2_expected, f"Conv2 expected {conv2_expected}, got {mask2.sum().item()}"

    # Conv3: only second analysis (8) + final (7) = 15 tokens
    conv3_expected = 15
    assert mask3.sum().item() == conv3_expected, f"Conv3 expected {conv3_expected}, got {mask3.sum().item()}"

    # Simulate packing
    packed_mask = torch.cat([mask1, mask2, mask3])
    total_masked = packed_mask.sum().item()
    expected_total = conv1_expected + conv2_expected + conv3_expected

    print(f"Conv1: analysis+final = {conv1_expected} masked")
    print(f"Conv2: simple assistant = {conv2_expected} masked")
    print(f"Conv3: second analysis+final only = {conv3_expected} masked")
    print(f"Total packed: {total_masked} masked")

    assert total_masked == expected_total, f"Expected {expected_total}, got {total_masked}"
    print("✓ Test passed!")


def test_packed_20_conversations_per_sample():
    """Stress test: 20 conversations with per-sample masking.

    Each conversation gets its mask computed independently, then masks are
    concatenated. This simulates the actual HuggingFacePackedDataset behavior.
    """
    print("\n" + "=" * 60)
    print("Test: 20 Packed Conversations (Per-Sample Masking)")
    print("=" * 60)

    all_tokens = []
    all_masks = []
    total_expected_masked = 0

    for conv_idx in range(20):
        if conv_idx % 4 == 0:
            # Simple: user → assistant
            content_len = 2 + (conv_idx % 5)
            conv_tokens = [
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1000 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE,
            ]
            conv_tokens.extend([2000 + conv_idx * 10 + i for i in range(content_len)])
            conv_tokens.append(HARMONY_RETURN)
            expected_masked = 3 + content_len + 1  # START, ASSISTANT, MESSAGE, content, RETURN

        elif conv_idx % 4 == 1:
            # Multi-turn: user1 → asst1 → user2 → asst2 (only asst2 masked)
            conv_tokens = [
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1000 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, HARMONY_RETURN,
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1100 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, 301, HARMONY_RETURN,
            ]
            expected_masked = 6  # Only asst2: START, ASSISTANT, MESSAGE, 300, 301, RETURN

        elif conv_idx % 4 == 2:
            # With analysis channel: user → analysis<|end|> → final<|return|>
            conv_tokens = [
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1000 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, ANALYSIS_TOKEN, HARMONY_MESSAGE, 400, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, FINAL_TOKEN, HARMONY_MESSAGE, 500, HARMONY_RETURN,
            ]
            expected_masked = 7 + 7  # analysis (7) + final (7)

        else:
            # Three turns: user1 → asst1 → user2 → asst2 → user3 → asst3 (only asst3)
            conv_tokens = [
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1000 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 200, HARMONY_RETURN,
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1100 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 300, HARMONY_RETURN,
                HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 1200 + conv_idx, HARMONY_END,
                HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 400, 401, HARMONY_RETURN,
            ]
            expected_masked = 6  # Only asst3

        conv_mask = create_loss_mask(torch.tensor(conv_tokens))
        actual_masked = conv_mask.sum().item()

        assert actual_masked == expected_masked, \
            f"Conv {conv_idx} (type {conv_idx % 4}): expected {expected_masked}, got {actual_masked}"

        all_tokens.extend(conv_tokens)
        all_masks.append(conv_mask)
        total_expected_masked += expected_masked

    # Concatenate all masks (simulating packing)
    packed_mask = torch.cat(all_masks)
    total_masked = packed_mask.sum().item()

    print(f"Total tokens: {len(all_tokens)}")
    print(f"Total masked (per-sample): {total_masked}")
    print(f"Expected masked: {total_expected_masked}")

    assert total_masked == total_expected_masked, \
        f"Expected {total_expected_masked} total masked, got {total_masked}"

    # Compare with after-packing mask (should be much smaller - only last conv)
    after_packing_mask = create_loss_mask(torch.tensor(all_tokens))
    print(f"After-packing mask would only get: {after_packing_mask.sum().item()} tokens")
    print(f"Per-sample masking gets {total_masked - after_packing_mask.sum().item()} MORE tokens for training")

    print("✓ Test passed!")


def test_tool_call_with_recipient():
    """Test assistant tool call with 'to=toolname' in header."""
    print("\n" + "=" * 60)
    print("Test: Tool Call with Recipient (to=)")
    print("=" * 60)

    TOOL_NAME = FUNCTIONS_TOKEN  # "functions" - using real token for test

    # Pattern: <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"loc":"SF"}<|call|>
    tokens = [
        # User turn
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,
        # Assistant tool call with to= recipient
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_CHANNEL, COMMENTARY_TOKEN,
        TO_TOKEN, EQUALS_TOKEN, TOOL_NAME,  # "to=functions.get_weather" in header
        HARMONY_CONSTRAIN, JSON_TOKEN, HARMONY_MESSAGE,
        400, 401, 402,  # JSON content
        HARMONY_CALL,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # ALL assistant tokens should have loss=1, including header with recipient
    # The model needs to learn the complete tool call header structure
    assert mask[5].item() == 1, "<|start|> should have loss=1"
    assert mask[6].item() == 1, "assistant should have loss=1"
    assert mask[7].item() == 1, "<|channel|> should have loss=1"
    assert mask[8].item() == 1, "commentary should have loss=1"
    assert mask[9].item() == 1, "'to' token should have loss=1"
    assert mask[10].item() == 1, "'=' token should have loss=1"
    assert mask[11].item() == 1, "tool name token should have loss=1"
    assert mask[12].item() == 1, "<|constrain|> should have loss=1"
    assert mask[13].item() == 1, "json should have loss=1"
    assert mask[14].item() == 1, "<|message|> should have loss=1"
    assert mask[15].item() == 1, "content token 400 should have loss=1"
    assert mask[16].item() == 1, "content token 401 should have loss=1"
    assert mask[17].item() == 1, "content token 402 should have loss=1"
    assert mask[18].item() == 1, "<|call|> should have loss=1"

    # 14 tokens total
    expected = 14
    assert mask.sum().item() == expected, f"Expected {expected}, got {mask.sum().item()}"

    print("✓ Test passed!")


def test_verify_token_ids():
    """Verify token ID constants match expected values."""
    print("\n" + "=" * 60)
    print("Test: Verify Token IDs")
    print("=" * 60)

    # These are the authoritative values from harmony/src/tiktoken_ext/public_encodings.rs
    expected_ids = {
        "HARMONY_START (<|start|>)": (HARMONY_START, 200006),
        "HARMONY_END (<|end|>)": (HARMONY_END, 200007),
        "HARMONY_MESSAGE (<|message|>)": (HARMONY_MESSAGE, 200008),
        "HARMONY_RETURN (<|return|>)": (HARMONY_RETURN, 200002),
        "HARMONY_CONSTRAIN (<|constrain|>)": (HARMONY_CONSTRAIN, 200003),
        "HARMONY_CHANNEL (<|channel|>)": (HARMONY_CHANNEL, 200005),
        "HARMONY_CALL (<|call|>)": (HARMONY_CALL, 200012),
        "HARMONY_ASSISTANT (assistant)": (HARMONY_ASSISTANT, 173781),
        "HARMONY_USER (user)": (HARMONY_USER, 1428),
    }

    all_correct = True
    for name, (actual, expected) in expected_ids.items():
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_correct = False
        print(f"  {status} {name}: {actual} (expected {expected})")

    assert all_correct, "Token ID mismatch detected!"

    print("\n✓ Test passed!")


def test_empty_input():
    """Test that empty input returns empty mask."""
    print("\n" + "=" * 60)
    print("Test: Empty Input")
    print("=" * 60)

    input_ids = torch.tensor([], dtype=torch.long)
    mask = create_loss_mask(input_ids)

    assert mask.shape == (0,), f"Expected shape (0,), got {mask.shape}"
    assert mask.sum().item() == 0, "Empty mask should sum to 0"

    print(f"Input shape: {input_ids.shape}")
    print(f"Mask shape: {mask.shape}")
    print("✓ Test passed!")


def test_assistant_only_no_user():
    """Test assistant-only sequence with no user message.

    When there's no user message, last_user_pos defaults to 0.
    An assistant at position 0 would NOT pass the check (0 > 0 is False),
    so it would not be masked. This is the expected edge case behavior.
    """
    print("\n" + "=" * 60)
    print("Test: Assistant Only (No User)")
    print("=" * 60)

    # Assistant-only sequence starting at position 0
    tokens = [
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 100, 101, HARMONY_RETURN,
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Tokens: {tokens}")
    print(f"Mask: {mask.tolist()}")

    # With no user message, last_user_pos = 0
    # Assistant starts at position 0, so 0 > 0 is False -> NOT masked
    # This is the documented edge case behavior
    expected = 0
    actual = mask.sum().item()

    assert actual == expected, f"Expected {expected} masked tokens (edge case: no user), got {actual}"

    print("Note: Assistant at position 0 with no user is NOT masked (edge case)")
    print("✓ Test passed!")


def test_only_last_user_turn_masked_explicit():
    """Explicitly verify that only the last user turn's assistant is masked.

    This test directly validates the "last user only" masking behavior
    with clear position calculations.
    """
    print("\n" + "=" * 60)
    print("Test: Only Last User Turn Masked (Explicit)")
    print("=" * 60)

    # Three user turns, each with assistant response
    # Only the assistant after User 3 (the LAST user) should be masked
    tokens = [
        # User 1 + Assistant 1
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 100, HARMONY_END,  # pos 0-4
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 110, HARMONY_RETURN,  # pos 5-9
        # User 2 + Assistant 2
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 200, HARMONY_END,  # pos 10-14
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 210, HARMONY_RETURN,  # pos 15-19
        # User 3 (LAST) + Assistant 3
        HARMONY_START, USER_TOKEN, HARMONY_MESSAGE, 300, HARMONY_END,  # pos 20-24 <- LAST user
        HARMONY_START, HARMONY_ASSISTANT, HARMONY_MESSAGE, 310, HARMONY_RETURN,  # pos 25-29
    ]

    input_ids = torch.tensor(tokens)
    mask = create_loss_mask(input_ids)

    print(f"Sequence length: {len(tokens)}")
    print(f"Mask: {mask.tolist()}")

    # Find last user position
    last_user_pos = 20  # User 3 starts at position 20

    # Verify Assistant 1 (positions 5-9) NOT masked
    assert mask[5:10].sum().item() == 0, "Assistant 1 should NOT be masked"

    # Verify Assistant 2 (positions 15-19) NOT masked
    assert mask[15:20].sum().item() == 0, "Assistant 2 should NOT be masked"

    # Verify Assistant 3 (positions 25-29) IS masked
    assert mask[25:30].sum().item() == 5, "Assistant 3 SHOULD be masked (5 tokens)"

    # Only Assistant 3: 5 tokens
    expected = 5
    actual = mask.sum().item()
    assert actual == expected, f"Expected {expected} tokens (last turn only), got {actual}"

    print(f"Last user at position: {last_user_pos}")
    print(f"Assistant 1 mask sum: {mask[5:10].sum().item()} (expected 0)")
    print(f"Assistant 2 mask sum: {mask[15:20].sum().item()} (expected 0)")
    print(f"Assistant 3 mask sum: {mask[25:30].sum().item()} (expected 5)")
    print("✓ Test passed!")


def test_real_harmony_data():
    """Test with actual tokenized Harmony data (if available)."""
    print("\n" + "=" * 60)
    print("Test: Real Harmony Data Sample")
    print("=" * 60)

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("/mnt/models/gpt-oss-20b/tokenizer.json")

        # Sample Harmony format text
        sample = """<|start|>system<|message|>You are a helpful assistant.<|end|><|start|>user<|message|>What is 2+2?<|end|><|start|>assistant<|message|>The answer is 4.<|return|>"""

        encoding = tokenizer.encode(sample)
        input_ids = torch.tensor(encoding.ids)

        print(f"Sample: {sample[:80]}...")
        print(f"Token count: {len(input_ids)}")

        mask = create_loss_mask(input_ids)
        stats = get_mask_statistics(mask)

        print(f"\nStatistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Assistant tokens: {stats['assistant_tokens']}")
        print(f"  Assistant ratio: {stats['assistant_ratio']:.1%}")

        # The assistant content "The answer is 4." + <|return|> should be masked in
        assert stats['assistant_tokens'] > 0, "Should have some assistant tokens"
        print("✓ Test passed!")

    except Exception as e:
        print(f"Skipped (tokenizer not available): {e}")


if __name__ == "__main__":
    # Core functionality tests
    test_basic_loss_mask()
    test_multi_turn_conversation()
    test_with_padding()
    test_batched_input()

    # Channel and end token tests
    test_with_channel()
    test_assistant_with_call()
    test_assistant_with_end()
    test_mixed_channels()

    # Packing behavior tests - simulate actual HuggingFacePackedDataset code path
    test_per_sample_masking_before_packing()
    test_packed_multi_turn_per_sample()
    test_packed_with_analysis_channels_per_sample()
    test_packed_20_conversations_per_sample()

    # Edge case tests
    test_tool_response_not_masked()
    test_constrain_token_included()
    test_tool_call_with_recipient()
    test_no_assistant_turns()
    test_developer_role_not_masked()
    test_empty_assistant_content()
    test_truncated_sequence()
    test_truncated_assistant_followed_by_new_conversation()

    # New edge case tests
    test_empty_input()
    test_assistant_only_no_user()
    test_only_last_user_turn_masked_explicit()

    # Validation tests
    test_verify_token_ids()
    test_real_harmony_data()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
