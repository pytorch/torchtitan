# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

from scripts.download_tokenizer import hf_download_tokenizer

from tokenizers import Tokenizer

from torchtitan.components.tokenizer import load_tokenizer


class TestTokenizerIntegration(unittest.TestCase):
    """Test integration between download_tokenizer and load_tokenizer functions."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_download_and_load_tokenizer_integration(self):
        """
        Test downloading tokenizer files and loading them, comparing with official APIs.

        This test:
        1. Downloads tokenizer files using hf_download_tokenizer
        2. Loads tokenizer using our load_tokenizer function
        3. Compares behavior with official Tokenizer library
        4. Compares with transformers AutoTokenizer (if available)
        """
        # Use a smaller, accessible model for testing
        test_repo_id = "deepseek-ai/DeepSeek-V3"

        # Step 1: Download tokenizer files
        hf_download_tokenizer(
            repo_id=test_repo_id,
            local_dir=self.temp_dir,
            hf_token=None,  # Public model, no token needed
        )

        # Step 2: Load tokenizer using our function
        model_name = test_repo_id.split("/")[-1]
        tokenizer_path = os.path.join(self.temp_dir, model_name)
        our_tokenizer = load_tokenizer(tokenizer_path)

        # Step 3: Load tokenizer using official Tokenizer library
        official_tokenizer = Tokenizer.from_pretrained(test_repo_id)

        # Step 4: Load tokenizer using transformers AutoTokenizer (if available)
        transformers_tokenizer = None
        try:
            from transformers import AutoTokenizer

            transformers_tokenizer = AutoTokenizer.from_pretrained(test_repo_id)
        except Exception:
            pass  # Skip transformers comparison if not available

        # Step 5: Compare underlying tokenizer attributes
        # Test that our_tokenizer.tokenizer has the same attributes as official_tokenizer

        # Get the underlying tokenizer from our wrapper
        our_underlying_tokenizer = our_tokenizer.tokenizer

        # Compare key attributes that should be identical
        # Vocabulary size
        self.assertEqual(
            our_underlying_tokenizer.get_vocab_size(),
            official_tokenizer.get_vocab_size(),
            "Vocabulary sizes should match",
        )

        # Compare vocabularies (this might be large, so we'll sample some tokens)
        our_vocab = our_underlying_tokenizer.get_vocab()
        official_vocab = official_tokenizer.get_vocab()

        # Test a few common tokens to ensure vocabularies match
        common_test_tokens = ["hello", "world", "the", "and", "is", "a"]
        for token in common_test_tokens:
            if token in our_vocab and token in official_vocab:
                self.assertEqual(
                    our_vocab[token],
                    official_vocab[token],
                    f"Token '{token}' should have the same ID in both tokenizers",
                )

        # Compare special tokens if they exist
        # Get added tokens from both tokenizers
        our_added_tokens = our_underlying_tokenizer.get_added_tokens_decoder()
        official_added_tokens = official_tokenizer.get_added_tokens_decoder()

        # Compare the number of added tokens
        self.assertEqual(
            len(our_added_tokens),
            len(official_added_tokens),
            "Number of added special tokens should match",
        )

        # Compare each added token
        for token_id, our_token in our_added_tokens.items():
            if token_id in official_added_tokens:
                official_token = official_added_tokens[token_id]
                self.assertEqual(
                    our_token.content,
                    official_token.content,
                    f"Special token content should match for ID {token_id}",
                )
                # Compare token properties if they exist
                if hasattr(our_token, "special") and hasattr(official_token, "special"):
                    self.assertEqual(
                        our_token.special,
                        official_token.special,
                        f"Special token 'special' property should match for token '{our_token.content}'",
                    )

        # Step 6: Compare with transformers tokenizer if available
        if transformers_tokenizer:
            # Test text encoding/decoding with transformers tokenizer
            text = "Hello world! This is a test."

            # Get tokens from our tokenizer (using the wrapper's encode method)
            our_tokens = our_tokenizer.encode(text)
            our_decoded_text = our_tokenizer.decode(our_tokens)

            # Verify our tokenizer produces expected output
            self.assertIsInstance(our_tokens, list)
            self.assertEqual(our_decoded_text, text)

            # Get tokens from transformers tokenizer
            transformers_tokens = transformers_tokenizer.encode(text)
            transformers_decoded = transformers_tokenizer.decode(transformers_tokens)

            # Compare our tokens with transformers tokens
            self.assertEqual(
                our_tokens,
                transformers_tokens,
                f"Tokens should match between our tokenizer and transformers tokenizer for input: '{text}'",
            )


if __name__ == "__main__":
    unittest.main()
