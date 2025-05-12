from torchtitan.tools.logging import logger
from transformers import AutoTokenizer


def get_hf_tokenizer(model_id: str):
    logger.info(f"Instantiating tokenizer for {model_id}")
    return AutoTokenizer.from_pretrained(model_id)
