# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.download_hf_assets import download_hf_assets


class TestDownloadHfAssets(unittest.TestCase):
    """Tests for the download_hf_assets script

    We mock `huggingface_hub.list_repo_files` and `huggingface_hub.hf_hub_download` to simulate the meta-llama/Llama-3.1-8B repo
    """

    # Complete file list from the actual meta-llama/Llama-3.1-8B repository
    COMPLETE_REPO_FILES = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "original/consolidated.00.pth",
        "original/params.json",
        "original/tokenizer.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "LICENSE",
        "README.md",
        "USE_POLICY.md",
    ]

    # Expected files for each asset type
    EXPECTED_FILES = {
        "tokenizer": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "original/tokenizer.model",
        ],
        "config": ["config.json", "generation_config.json"],
        "safetensors": [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
        ],
        "index": ["model.safetensors.index.json"],
    }

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_id = "meta-llama/Llama-3.1-8B"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _setup_mocks(self, mock_download, mock_list_files, repo_files=None):
        """Helper to setup mock configurations"""
        mock_list_files.return_value = repo_files or self.COMPLETE_REPO_FILES
        mock_download.return_value = None

    def _get_downloaded_files(self, mock_download):
        """Helper to extract downloaded filenames from mock calls"""
        return [call[1]["filename"] for call in mock_download.call_args_list]

    def _assert_files_downloaded(self, mock_download, expected_files):
        """Helper to assert expected files were downloaded"""
        self.assertEqual(mock_download.call_count, len(expected_files))
        downloaded_files = self._get_downloaded_files(mock_download)
        for expected_file in expected_files:
            self.assertIn(expected_file, downloaded_files)

    def _call_download_hf_assets(self, **kwargs):
        """Helper to call download_hf_assets with common defaults"""
        defaults = {
            "repo_id": self.repo_id,
            "local_dir": self.temp_dir,
        }
        defaults.update(kwargs)
        return download_hf_assets(**defaults)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_download_single_asset_types(self, mock_download, mock_list_files):
        """Test downloading individual asset types"""
        self._setup_mocks(mock_download, mock_list_files)

        # Test each asset type individually
        for asset_type, expected_files in self.EXPECTED_FILES.items():
            with self.subTest(asset_type=asset_type):
                mock_download.reset_mock()
                self._call_download_hf_assets(asset_types=[asset_type])
                self._assert_files_downloaded(mock_download, expected_files)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_download_multiple_asset_types(self, mock_download, mock_list_files):
        """Test downloading multiple asset types together"""
        self._setup_mocks(mock_download, mock_list_files)

        # Get all expected files (removing duplicates)
        all_expected_files = set()
        for files in self.EXPECTED_FILES.values():
            all_expected_files.update(files)

        self._call_download_hf_assets(asset_types=list(self.EXPECTED_FILES.keys()))
        self._assert_files_downloaded(mock_download, all_expected_files)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_download_all_files(self, mock_download, mock_list_files):
        """Test downloading all files with --all option"""
        self._setup_mocks(mock_download, mock_list_files)

        self._call_download_hf_assets(asset_types=[], download_all=True)
        self._assert_files_downloaded(mock_download, self.COMPLETE_REPO_FILES)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_additional_patterns(self, mock_download, mock_list_files):
        """Test downloading with additional file patterns"""
        test_files = ["tokenizer.json", "custom_file.txt", "README.md"]
        self._setup_mocks(mock_download, mock_list_files, repo_files=test_files)

        self._call_download_hf_assets(
            asset_types=["tokenizer"], additional_patterns=["*.txt"]
        )

        # Only tokenizer.json and custom_file.txt should be downloaded
        expected_files = ["tokenizer.json", "custom_file.txt"]
        self._assert_files_downloaded(mock_download, expected_files)

    @patch("huggingface_hub.hf_hub_download")
    def test_list_files(self, mock_download):
        """Tests that list files returns correct list of files by using real huggingface_hub.list_files"""
        """This test uses larger deepseek-ai/DeepSeek-V3 repo for more thorough test"""

        # Setup mock download
        mock_download.return_value = None

        # Test downloading safetensors asset type
        self._call_download_hf_assets(
            repo_id="deepseek-ai/DeepSeek-V3",
            asset_types=["safetensors"],
        )

        # Verify all 163 safetensors files plus index file are downloaded
        expected_safetensors_files = [
            f"model-{i:05d}-of-000163.safetensors" for i in range(1, 164)
        ]
        expected_files = expected_safetensors_files + [
            "model.safetensors.index.json",
        ]

        self._assert_files_downloaded(mock_download, expected_files)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_nested_directory_handling(self, mock_download, mock_list_files):
        """Tests that files in nested directory files are detected and downloaded correctly"""
        test_files = [
            "tokenizer.json",
            "original/tokenizer.model",
            "original/consolidated.00.pth",  # Should NOT be downloaded (no .pth pattern)
            "config.json",
        ]
        self._setup_mocks(mock_download, mock_list_files, repo_files=test_files)

        self._call_download_hf_assets(asset_types=["tokenizer", "config"])

        # Verify nested tokenizer file is downloaded but .pth file is not
        expected_files = ["tokenizer.json", "original/tokenizer.model", "config.json"]
        self._assert_files_downloaded(mock_download, expected_files)

        # Verify .pth file was NOT downloaded
        downloaded_files = self._get_downloaded_files(mock_download)
        self.assertNotIn("original/consolidated.00.pth", downloaded_files)

    @patch("huggingface_hub.list_repo_files")
    def test_missing_files_warning(self, mock_list_files):
        """Test warning when requested files are not found"""
        mock_list_files.return_value = ["config.json", "README.md"]

        with patch("builtins.print") as mock_print:
            self._call_download_hf_assets(asset_types=["tokenizer"])

            # Check that warning was printed
            warning_calls = [
                call
                for call in mock_print.call_args_list
                if "Warning: No matching files found for asset_type 'tokenizer'"
                in str(call)
            ]
            self.assertTrue(len(warning_calls) > 0)

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_download_failure_handling(self, mock_download, mock_list_files):
        """Test handling of download failures"""
        from requests.exceptions import HTTPError

        self._setup_mocks(
            mock_download,
            mock_list_files,
            repo_files=["tokenizer.json", "missing_file.json"],
        )

        # Mock 404 error for missing file
        def download_side_effect(*args, **kwargs):
            if kwargs["filename"] == "missing_file.json":
                response = Mock()
                response.status_code = 404
                raise HTTPError(response=response)
            return None

        mock_download.side_effect = download_side_effect

        with patch("builtins.print") as mock_print:
            self._call_download_hf_assets(
                asset_types=["tokenizer"], additional_patterns=["missing_file.json"]
            )

            # Check that 404 error was handled gracefully
            error_calls = [
                call
                for call in mock_print.call_args_list
                if "File missing_file.json not found, skipping..." in str(call)
            ]
            self.assertTrue(len(error_calls) > 0)

    def test_invalid_repo_id_format(self):
        """Test error handling for invalid repo_id format"""
        with self.assertRaises(ValueError) as context:
            self._call_download_hf_assets(
                repo_id="invalid-repo-id", asset_types=["tokenizer"]
            )
        self.assertIn("Invalid repo_id format", str(context.exception))

    def test_unknown_asset_type(self):
        """Test error handling for unknown asset type"""
        with self.assertRaises(ValueError) as context:
            self._call_download_hf_assets(asset_types=["unknown_type"])
        self.assertIn("Unknown asset type unknown_type", str(context.exception))


if __name__ == "__main__":
    unittest.main()
