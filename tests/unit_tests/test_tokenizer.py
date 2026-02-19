# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

from requests.exceptions import HTTPError
from scripts.download_hf_assets import download_hf_assets
from tokenizers import Tokenizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer


class TestTokenizerIntegration(unittest.TestCase):
    """Test integration between download_hf_assets and load_tokenizer functions."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _compare_tokenizers(self, our_tokenizer, reference_tokenizer, test_repo_id):
        """
        Comprehensive comparison between our tokenizer and a reference tokenizer.
        Supports both tokenizers library and transformers library tokenizers.

        Args:
            our_tokenizer: Our HuggingFaceTokenizer instance or underlying tokenizer
            reference_tokenizer: Reference tokenizer (tokenizers.Tokenizer or transformers tokenizer)
            test_repo_id: Repository ID for context in error messages
        """
        # Detect tokenizer type and create adapter functions
        is_transformers = hasattr(reference_tokenizer, "vocab_size") and not hasattr(
            reference_tokenizer, "get_vocab_size"
        )

        if is_transformers:
            # Transformers tokenizer API
            def get_vocab_size(tokenizer):
                return len(tokenizer.get_vocab())

            def get_vocab(tokenizer):
                return tokenizer.get_vocab()

            def encode_text(tokenizer, text):
                return tokenizer.encode(text)

            def decode_tokens(tokenizer, tokens):
                return tokenizer.decode(tokens)

            def get_added_tokens_func(tokenizer):
                # Transformers doesn't have get_added_tokens_decoder, so we'll skip this comparison
                return {}

            tokenizer_type = "transformers"
        else:
            # Tokenizers library API
            def get_vocab_size(tokenizer):
                return len(tokenizer.get_vocab())

            def get_vocab(tokenizer):
                return tokenizer.get_vocab()

            def encode_text(tokenizer, text):
                return tokenizer.encode(text).ids

            def decode_tokens(tokenizer, tokens):
                return tokenizer.decode(tokens)

            def get_added_tokens_func(tokenizer):
                return tokenizer.get_added_tokens_decoder()

            tokenizer_type = "tokenizers"

        # 1. Compare vocabulary sizes
        self.assertEqual(
            our_tokenizer.get_vocab_size(),
            get_vocab_size(reference_tokenizer),
            f"Vocabulary sizes should match for {test_repo_id} ({tokenizer_type})",
        )

        # 2. Compare vocabularies with more comprehensive sampling
        our_vocab = our_tokenizer.get_vocab()
        reference_vocab = get_vocab(reference_tokenizer)

        # Test common tokens
        common_test_tokens = [
            "hello",
            "world",
            "the",
            "and",
            "is",
            "a",
            "to",
            "of",
            "in",
            "for",
        ]
        for token in common_test_tokens:
            if token in our_vocab and token in reference_vocab:
                self.assertEqual(
                    our_vocab[token],
                    reference_vocab[token],
                    f"Token '{token}' should have the same ID in both tokenizers for {test_repo_id} ({tokenizer_type})",
                )

        # Test a random sample of tokens (more comprehensive than just common words)
        import random

        vocab_keys = list(our_vocab.keys())
        if len(vocab_keys) > 50:
            # Sample 50 random tokens for comparison
            sample_tokens = random.sample(vocab_keys, 50)
            for token in sample_tokens:
                if token in reference_vocab:
                    self.assertEqual(
                        our_vocab[token],
                        reference_vocab[token],
                        f"Random sampled token '{token}' should have the same ID in \
both tokenizers for {test_repo_id} ({tokenizer_type})",
                    )

        # 3. Compare special tokens (only for tokenizers library, not transformers)
        if not is_transformers:
            our_added_tokens = our_tokenizer.get_added_tokens_decoder()
            reference_added_tokens = get_added_tokens_func(reference_tokenizer)

            self.assertEqual(
                len(our_added_tokens),
                len(reference_added_tokens),
                f"Number of added special tokens should match for {test_repo_id} ({tokenizer_type})",
            )

            # Compare each added token
            for token_id, our_token in our_added_tokens.items():
                if token_id in reference_added_tokens:
                    reference_token = reference_added_tokens[token_id]
                    self.assertEqual(
                        our_token.content,
                        reference_token.content,
                        f"Special token content should match for ID {token_id} in {test_repo_id} ({tokenizer_type})",
                    )
                    # Compare token properties if they exist
                    if hasattr(our_token, "special") and hasattr(
                        reference_token, "special"
                    ):
                        self.assertEqual(
                            our_token.special,
                            reference_token.special,
                            f"Special token 'special' property should match \
for token '{our_token.content}' in {test_repo_id} ({tokenizer_type})",
                        )

        # 4. Functional testing - encode/decode comparison
        test_texts = [
            "Hello world!",
            "This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Special characters: @#$%^&*()",
            "Numbers: 123456789",
            "Mixed: Hello123 World!@#",
            "",  # Empty string
            " ",  # Single space
            "  ",  # Multiple spaces
        ]

        for text in test_texts:
            # Compare encoding - handle different tokenizer types
            if hasattr(our_tokenizer, "tokenizer"):
                # Our wrapper tokenizer - returns list directly
                our_tokens = our_tokenizer.encode(text)
            else:
                # Underlying HF tokenizer - returns object with .ids
                our_encoded = our_tokenizer.encode(text)
                our_tokens = (
                    our_encoded.ids if hasattr(our_encoded, "ids") else our_encoded
                )

            reference_tokens = encode_text(reference_tokenizer, text)

            self.assertEqual(
                our_tokens,
                reference_tokens,
                f"Encoded tokens should match for text '{text}' in {test_repo_id} ({tokenizer_type})",
            )

            # Compare decoding:
            # for transformers-Tokenizers, skip_special_tokens=False by default
            # for tokenizers library, skip_special_tokens=True by default
            skip_special_tokens = not is_transformers
            our_decoded = our_tokenizer.decode(
                our_tokens, skip_special_tokens=skip_special_tokens
            )
            reference_decoded = decode_tokens(reference_tokenizer, reference_tokens)

            self.assertEqual(
                our_decoded,
                reference_decoded,
                f"Decoded text should match for '{text}' in {test_repo_id} ({tokenizer_type})",
            )

        # 5. Edge case testing
        edge_cases = [
            "🚀🌟✨",  # Emojis
            "café naïve résumé",  # Accented characters
            "こんにちは世界",  # Non-Latin scripts (Japanese)
            "Здравствуй мир",  # Cyrillic
            "\n\t\r",  # Whitespace characters
            "a" * 1000,  # Very long repeated character
        ]

        for text in edge_cases:
            # Handle different tokenizer types for edge cases too
            if hasattr(our_tokenizer, "tokenizer"):
                our_tokens = our_tokenizer.encode(text)
            else:
                our_encoded = our_tokenizer.encode(text)
                our_tokens = (
                    our_encoded.ids if hasattr(our_encoded, "ids") else our_encoded
                )

            reference_tokens = encode_text(reference_tokenizer, text)

            self.assertEqual(
                our_tokens,
                reference_tokens,
                f"Edge case tokens should match for text '{text[:50]}...' in {test_repo_id} ({tokenizer_type})",
            )

    @parametrize(
        "test_repo_id",
        [
            "meta-llama/Llama-3.1-8B",
            "deepseek-ai/DeepSeek-V3",
            # "black-forest-labs/FLUX.1-dev", TODO: load the actual tokenizer
            "Qwen/Qwen2-7B",
        ],
    )
    def test_download_and_build_tokenizer(self, test_repo_id):
        """
        Test downloading tokenizer files and loading them, comparing with official APIs.

        This test:
        1. Downloads tokenizer files using download_hf_tokenizer_files
        2. Loads tokenizer using our load_tokenizer function
        3. Compares behavior with official Tokenizer library
        4. Compares with transformers AutoTokenizer (if available)
        """
        # Step 1: Download tokenizer files
        try:
            download_hf_assets(
                repo_id=test_repo_id,
                local_dir=self.temp_dir,
                asset_types="tokenizer",
            )
        except HTTPError as e:
            if test_repo_id == "meta-llama/Llama-3.1-8B":
                self.skipTest(
                    f"Could not download tokenizer files for Llama-3.1-8B: {e}"
                )
            else:
                raise e

        # Step 2: Load tokenizer using our function
        model_name = test_repo_id.split("/")[-1]
        tokenizer_dir = "tokenizer" if model_name == "FLUX.1-dev" else "."
        tokenizer_path = os.path.join(self.temp_dir, model_name, tokenizer_dir)
        our_tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)

        # Step 3: Load tokenizer using official Tokenizer library (if available)
        official_tokenizer = None
        try:
            official_tokenizer = Tokenizer.from_pretrained(test_repo_id)
        except Exception as e:
            print(f"Warning: Could not load official tokenizer for {test_repo_id}: {e}")

        # Step 4: Load tokenizer using transformers AutoTokenizer (if available)
        transformers_tokenizer = None
        try:
            from transformers import AutoTokenizer

            transformers_tokenizer = AutoTokenizer.from_pretrained(test_repo_id)
        except Exception as e:
            print(f"Warning: Could not load AutoTokenizer for {test_repo_id}: {e}")

        # Step 5: Compare underlying tokenizer attributes (only if official tokenizer is available)
        if official_tokenizer:
            self._compare_tokenizers(
                our_tokenizer.tokenizer, official_tokenizer, test_repo_id
            )

        # Step 6: Compare with transformers tokenizer if available
        if transformers_tokenizer:
            self._compare_tokenizers(
                our_tokenizer, transformers_tokenizer, test_repo_id
            )


instantiate_parametrized_tests(TestTokenizerIntegration)

if __name__ == "__main__":
    unittest.main()
