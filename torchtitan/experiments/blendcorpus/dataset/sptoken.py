# torchtitan/datasets/tokenizer/sptoken.py
import os
from typing import List
import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path: str):
        assert isinstance(model_path, (str, os.PathLike)) and model_path, f"SP model path must be a non-empty string, got: {model_path!r}"
        model_path = str(model_path)
        # Accept a directory containing tokenizer.model or a direct .model file
        spm_file = model_path if model_path.endswith('.model') else os.path.join(model_path, 'tokenizer.model')
        assert os.path.exists(spm_file), f"SP model not found: {spm_file}"
        self.sp = spm.SentencePieceProcessor(model_file=spm_file)

        # Attributes expected by Titan
        self.vocab_size = self.sp.vocab_size()
        self.n_words = self.vocab_size
        print(f"[SPTokenizer] Loaded model: {spm_file}, vocab size: {self.vocab_size}")

        # Common special token ids (use -1 when absent)
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else -1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else -1
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else -1
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() != -1 else -1

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Titan may call encode(..., bos=True, eos=True). Respect those flags.
        """
        ids = self.sp.encode(text, out_type=int)
        if bos and self.bos_id != -1:
            ids = [self.bos_id] + ids
        if eos and self.eos_id != -1:
            ids = ids + [self.eos_id]
        # Safety: range check
        if ids:
            mn, mx = min(ids), max(ids)
            assert mn >= 0 and mx < self.vocab_size, (
                f"Token IDs out of range: min={mn}, max={mx}, vocab_size={self.vocab_size}"
            )
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

def build_sentencepiece_tokenizer(job_config):
    # Prefer explicit tokenizer_path; fall back to hf_assets_path
    model_path = getattr(job_config.model, 'tokenizer_path', None) or getattr(job_config.model, 'hf_assets_path', None)
    assert model_path, (
        "Neither job_config.model.tokenizer_path nor job_config.model.hf_assets_path is set for SentencePiece tokenizer."
    )
    print(f"[SPTokenizer] Using model path: {model_path}")
    return SPTokenizer(model_path)

