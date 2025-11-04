from dataclasses import dataclass, field

@dataclass
class HFTransformers:
    model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen3-4B-Instruct-2507')"""

@dataclass
class JobConfig:
    hf_transformers: HFTransformers = field(default_factory=HFTransformers)