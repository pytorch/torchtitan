"""
Example of processing h4_10710 dataset with chat templates.
Handles system/human/gpt conversations with proper masking.
"""

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from pretokenizer import Pretokenizer
from tqdm import tqdm
from transformers import AutoTokenizer


def create_chat_conversation(
    conversations: List[dict], tokenizer: AutoTokenizer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a chat-formatted conversation.

    Args:
        conversations: List of {"from": "human"/"gpt"/"system", "value": "text"} dicts
        tokenizer: The tokenizer to use

    Returns:
        tokens: Token IDs
        masks: Loss masks (-100 for system/user turns, token IDs for assistant turns)
    """
    # Convert to standard chat format
    messages = []
    for msg in conversations:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "gpt":
            role = "assistant"
        elif msg["from"] == "system":
            role = "system"
        else:
            raise ValueError(f"Unknown message type: {msg['from']}")

        messages.append({"role": role, "content": msg["value"]})

    # Apply chat template to get all tokens
    all_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    # Find where each turn ends by applying template incrementally
    masks = []
    current_pos = 0

    for i, msg in enumerate(messages):
        # Apply template up to and including this message
        tokens_up_to_here = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )

        # Length of this turn
        turn_length = len(tokens_up_to_here) - current_pos

        # Mask if it's NOT an assistant turn (mask system + user)
        if msg["role"] != "assistant":
            masks.extend([-100] * turn_length)
        else:
            # Use actual tokens for assistant turns
            masks.extend(tokens_up_to_here[current_pos:])

        current_pos = len(tokens_up_to_here)

    return np.array(all_tokens, dtype=np.int64), np.array(masks, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Process h4_10710 dataset with chat templates"
    )
    parser.add_argument(
        "--input-parquet", type=str, required=True, help="Input parquet file"
    )
    parser.add_argument(
        "--output-prefix", type=str, required=True, help="Output prefix for numpy files"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizers/hermes-4-14b-vq",
        help="Tokenizer path",
    )

    args = parser.parse_args()

    print("🚀 Processing h4_10710 dataset with chat templates")
    print("=" * 80)
    print(f"   Input: {args.input_parquet}")
    print(f"   Output: {args.output_prefix}")

    # Load tokenizer
    print(f"\n📝 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"   Vocab size: {len(tokenizer):,}")

    # Load parquet
    print("\n📂 Loading parquet...")
    df = pd.read_parquet(args.input_parquet)
    print(f"   Loaded {len(df):,} examples")

    # Create pretokenizer
    print("\n🏭 Creating pretokenizer...")
    pretokenizer = Pretokenizer(output_prefix=args.output_prefix, chunk_size=1000)

    # Process conversations
    print("\n🔄 Processing conversations...")
    sequences = []
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        conversations = row["conversations"]

        try:
            tokens, masks = create_chat_conversation(conversations, tokenizer)
            sequences.append((tokens, masks))

            # Add to pretokenizer periodically
            if len(sequences) >= 500:
                pretokenizer.add_sequences(sequences)
                sequences = []

        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first few errors
                print(f"\n⚠️  Warning: Failed to process example {idx}: {e}")
            continue

    # Add remaining sequences
    if sequences:
        pretokenizer.add_sequences(sequences)

    # Finalize
    print("\n✨ Finalizing pretokenization...")
    pretokenizer.finalize()

    if errors > 0:
        print(f"\n⚠️  {errors} examples failed to process")

    print("\n✅ Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
