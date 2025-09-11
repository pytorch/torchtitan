from torchtitan.experiments.blendcorpus.dataset.sptoken import build_sentencepiece_tokenizer
from torchtitan.components.tokenizer import build_hf_tokenizer

def build_tokenizer(job_config):
    backend = str(getattr(job_config.model, 'tokenizer_backend', '')).strip().lower()
    if backend in {'', 'sptoken', 'sentencepiece', 'sp', 'spm'}:
        print('[Tokenizer] Using backend: sptoken (SentencePiece)')
        return build_sentencepiece_tokenizer(job_config)
    if backend in {'huggingface', 'hf'}:
        print('[Tokenizer] Using backend: huggingface (HF AutoTokenizer)')
        return build_hf_tokenizer(job_config)
    if backend == 'tiktoken':
        raise NotImplementedError("tokenizer_backend='tiktoken' is not supported for this training recipe; use 'sptoken'.")
    raise Exception(f"Unknown tokenizer_backend '{backend}'. Choose 'sptoken' or 'hf'.")

