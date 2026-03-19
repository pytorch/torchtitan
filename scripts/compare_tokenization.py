"""Verify that SFTDataset's tokenization matches HuggingFace's apply_chat_template.

Compares in three phases:
1. Chat template rendering: TorchTitan vs HF produce the same text
2. Full tokenization: encoded token IDs match
3. Prompt/label split: label masking boundary and valid label tokens are correct

Usage:
    python -m scripts.compare_tokenization --tokenizer_path ./assets/hf/Qwen3-0.6B
    python -m scripts.compare_tokenization --tokenizer_path ./assets/hf/Qwen3-8B
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.sft.dataset import SFTDataset


def process_sample(sample):
    answer = sample["answer"]
    reasoning, final_answer = answer.rsplit("####", 1)
    return [
        {"role": "user", "content": sample["question"]},
        {
            "role": "assistant",
            "reasoning_content": reasoning.strip(),
            "content": final_answer.strip(),
        },
    ]


def compare_chat_template(samples, process_sample, tt_tok, hf_tok):
    """Phase 1: Verify chat template text is identical between TT and HF."""
    print("=== Phase 1: Chat template rendering ===")
    failures = []

    for idx, sample in enumerate(samples):
        messages = process_sample(sample)
        tt_text = tt_tok.apply_chat_template(messages)
        hf_text = hf_tok.apply_chat_template(messages, tokenize=False)

        if tt_text != hf_text:
            failures.append(idx)
            print(f"  FAIL sample {idx}: text differs")
            print(f"    TT ({len(tt_text)} chars): {tt_text[:200]!r}...")
            print(f"    HF ({len(hf_text)} chars): {hf_text[:200]!r}...")
        else:
            print(f"  OK   sample {idx} ({len(tt_text)} chars)")

    if failures:
        print(f"  FAILED: {len(failures)}/{len(samples)} samples\n")
        return False
    print(f"  PASSED: all {len(samples)} samples\n")
    return True


def compare_full_tokenization(samples, process_sample, tt_tok, hf_tok, sft_ds):
    """Phase 2: Verify encoded token IDs match between SFTDataset and HF."""
    print("=== Phase 2: Full tokenization ===")
    failures = []

    for idx, sample in enumerate(samples):
        messages = process_sample(sample)

        result = sft_ds._tokenize_sample(sample)
        if result is None:
            print(f"  SKIP sample {idx}: _tokenize_sample returned None")
            continue
        tt_input_ids, _ = result

        # HF reference: SFTDataset does encode(text, add_bos=True, add_eos=True)
        # then input_ids = full_tokens[:-1], so input_ids should equal HF's tokens
        hf_ids = hf_tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]

        if len(tt_input_ids) != len(hf_ids):
            failures.append(idx)
            print(f"  FAIL sample {idx}: length differs TT={len(tt_input_ids)} HF={len(hf_ids)}")
            continue

        mismatches = [i for i in range(len(hf_ids)) if tt_input_ids[i] != hf_ids[i]]
        if mismatches:
            failures.append(idx)
            print(f"  FAIL sample {idx}: {len(mismatches)} token mismatches at positions {mismatches}")
            for pos in mismatches:
                print(f"    pos {pos}: TT={tt_input_ids[pos]} ({tt_tok.decode([tt_input_ids[pos]])!r}) "
                      f"HF={hf_ids[pos]} ({hf_tok.decode([hf_ids[pos]])!r})")
        else:
            print(f"  OK   sample {idx} ({len(tt_input_ids)} tokens)")

    if failures:
        print(f"  FAILED: {len(failures)}/{len(samples)} samples\n")
        return False
    print(f"  PASSED: all {len(samples)} samples\n")
    return True


def compare_prompt_label_split(samples, process_sample, tt_tok, hf_tok, sft_ds):
    """Phase 3: Verify label masking boundary and label token values are correct."""
    print("=== Phase 3: Prompt/label split ===")
    failures = []

    for idx, sample in enumerate(samples):
        messages = process_sample(sample)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]

        result = sft_ds._tokenize_sample(sample)
        if result is None:
            print(f"  SKIP sample {idx}: _tokenize_sample returned None")
            continue
        _, tt_label_ids = result

        hf_full_ids = hf_tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
        hf_prompt_ids = hf_tok.apply_chat_template(
            prompt_msgs, tokenize=True, return_dict=True, add_generation_prompt=True
        )["input_ids"]

        sample_ok = True
        errors = []

        # Check HF prompt is a prefix of HF full (sanity check for BPE consistency)
        if hf_full_ids[:len(hf_prompt_ids)] != hf_prompt_ids:
            errors.append("HF prompt is not a prefix of full tokens (BPE boundary issue)")
            sample_ok = False

        # Compare masking boundary
        tt_valid_positions = [i for i, l in enumerate(tt_label_ids) if l != IGNORE_INDEX]
        tt_first_valid = tt_valid_positions[0] if tt_valid_positions else -1
        hf_first_valid = len(hf_prompt_ids) - 1  # after next-token-prediction shift

        if tt_first_valid != hf_first_valid:
            errors.append(
                f"masking boundary: TT first_valid={tt_first_valid} "
                f"HF first_valid={hf_first_valid} (delta={tt_first_valid - hf_first_valid})"
            )
            sample_ok = False

        # Compare valid token count
        # SFTDataset has +1 from add_eos (extra valid label at the end)
        tt_valid_count = len(tt_valid_positions)
        hf_completion_count = len(hf_full_ids) - len(hf_prompt_ids)
        if tt_valid_count != hf_completion_count + 1:
            errors.append(
                f"valid count: TT={tt_valid_count} HF_completion={hf_completion_count} "
                f"(expected TT = HF + 1 due to add_eos)"
            )
            sample_ok = False

        # Verify label token IDs match at valid positions (skip last = add_eos)
        label_mismatches = []
        for pos in tt_valid_positions[:-1]:
            if pos + 1 < len(hf_full_ids) and tt_label_ids[pos] != hf_full_ids[pos + 1]:
                label_mismatches.append(pos)
        if label_mismatches:
            errors.append(f"{len(label_mismatches)} label value mismatches at {label_mismatches[:10]}")
            sample_ok = False

        if not sample_ok:
            failures.append(idx)
            print(f"  FAIL sample {idx}:")
            for e in errors:
                print(f"    - {e}")
        else:
            print(
                f"  OK   sample {idx} "
                f"(prompt={len(hf_prompt_ids)}, completion={hf_completion_count}, "
                f"valid_labels={tt_valid_count})"
            )

    if failures:
        print(f"  FAILED: {len(failures)}/{len(samples)} samples\n")
        return False
    print(f"  PASSED: all {len(samples)} samples\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="./assets/hf/Qwen3-0.6B")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    tt_tok = HuggingFaceTokenizer(tokenizer_path=args.tokenizer_path)
    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    samples = [dataset[i] for i in range(args.num_samples)]

    sft_ds = SFTDataset(
        dataset=dataset,
        tokenizer=tt_tok,
        sample_processor=process_sample,
        seq_len=4096,
    )

    results = {}

    # Phase 1: Chat template
    results["chat_template"] = compare_chat_template(samples, process_sample, tt_tok, hf_tok)

    # Phase 2: Full tokenization
    results["tokenization"] = compare_full_tokenization(samples, process_sample, tt_tok, hf_tok, sft_ds)

    # Phase 3: Prompt/label split
    results["prompt_label_split"] = compare_prompt_label_split(samples, process_sample, tt_tok, hf_tok, sft_ds)

    # Summary
    print("=== Summary ===")
    all_passed = True
    for phase, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {phase}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nAll phases passed for {args.num_samples} samples.")
    else:
        print(f"\nSome phases failed.")
        exit(1)


if __name__ == "__main__":
    main()
