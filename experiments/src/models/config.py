from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    hf_repo: str
    quantization: str
    max_new_tokens: int
    do_sample: bool
    skip_reasoning: bool
    notes: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen2.5-32b": ModelConfig(
        name="Qwen2.5-32B-Instruct",
        hf_repo="Qwen/Qwen2.5-32B-Instruct",
        quantization="bitsandbytes 4-bit FP4, float16 compute, double quant",
        max_new_tokens=512,
        do_sample=False,
        skip_reasoning=False,
        notes="Loaded from the native checkpoint with BitsAndBytesConfig",
    ),
    "llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B Instruct",
        hf_repo="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        quantization="AWQ INT4 pre-quantized checkpoint",
        max_new_tokens=512,
        do_sample=False,
        skip_reasoning=False,
        notes="Loaded from the pre-quantized AWQ checkpoint",
    ),
    "deepseek-r1-distill-qwen-32b": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-32B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quantization="bitsandbytes 4-bit FP4, float16 compute, double quant",
        max_new_tokens=700,
        do_sample=False,
        skip_reasoning=True,
        notes="The reasoning block is skipped by forcing the closing think token sequence before generation",
    ),
}
