"""Debug script to inspect chat template rendering and label masking for SFT.

Shows the full formatted text, per-message token boundaries using the
incremental prefix re-tokenization approach, and the resulting label mask.

Usage:
    python scripts/debug_chat_template.py --tokenizer_path ./assets/hf/Qwen3-8B
    python scripts/debug_chat_template.py --tokenizer_path ./tests/assets/tokenizer
"""

import argparse

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./assets/hf/Qwen3-8B",
    )
    args = parser.parse_args()

    tok = HuggingFaceTokenizer(tokenizer_path=args.tokenizer_path)

    messages = [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": "2 + 3 = 5. #### 5"},
    ]

    # Full conversation
    full_text = tok.apply_chat_template(messages)
    full_tokens = tok.encode(full_text, add_bos=True, add_eos=True)

    print("=== FULL TEXT ===")
    print(repr(full_text))
    print(f"\nFull tokens: {len(full_tokens)}")
    print()

    # Build label mask using incremental prefix re-tokenization (delta approach)
    # This matches the logic in SFTDataset._tokenize_sample
    input_ids = full_tokens[:-1]
    label_ids = list(full_tokens[1:])

    prev_token_len = 0
    print("=== PER-MESSAGE BOUNDARIES ===")
    for i, message in enumerate(messages):
        prefix_messages = messages[: i + 1]
        is_last = i == len(messages) - 1
        prefix_text = tok.apply_chat_template(
            prefix_messages, add_generation_prompt=not is_last
        )
        prefix_tokens = tok.encode(prefix_text, add_bos=True, add_eos=False)
        curr_token_len = len(prefix_tokens)

        delta = curr_token_len - prev_token_len
        masked = message["role"] != "assistant"

        print(f"  [{message['role']}] tokens {prev_token_len}-{curr_token_len} "
              f"(delta={delta}) {'MASKED' if masked else 'TRAINED'}")

        if masked:
            mask_start = max(prev_token_len - 1, 0)
            mask_end = min(curr_token_len - 1, len(label_ids))
            for j in range(mask_start, mask_end):
                label_ids[j] = IGNORE_INDEX

        prev_token_len = curr_token_len

    print()

    # Show token-by-token breakdown
    print("=== TOKEN BREAKDOWN ===")
    for i, tid in enumerate(input_ids):
        label = label_ids[i]
        token_str = tok.decode([tid])
        if label == IGNORE_INDEX:
            status = "MASKED"
        else:
            status = f"label={label}"
        print(f"  [{i:3d}] {tid:6d} {token_str!r:20s}  {status}")


if __name__ == "__main__":
    main()
