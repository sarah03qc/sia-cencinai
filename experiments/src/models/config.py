"""Configuración de los 3 modelos del benchmark (ver tabla en CLAUDE.md).

Generación determinística (do_sample=False) en los 3 modelos: para un
benchmark de QA factual necesitamos poder reproducir las respuestas y
comparar los modelos entre sí sin que la variabilidad del muestreo afecte
las métricas.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    hf_repo: str
    quantization: str
    max_new_tokens: int
    do_sample: bool
    notes: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen2.5-32b": ModelConfig(
        name="Qwen2.5-32B-Instruct",
        hf_repo="Qwen/Qwen2.5-32B-Instruct",
        quantization="bitsandbytes (4-bit u 8-bit, por definir)",
        max_new_tokens=512,
        do_sample=False,
        notes="Mejor desempeño en el mini-experimento de filtrado.",
    ),
    "llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B Instruct",
        hf_repo="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        quantization="AWQ-INT4 (pre-cuantizado)",
        max_new_tokens=512,
        do_sample=False,
        notes=(
            "Requiere aceptar la licencia de Meta en HuggingFace. "
            "NO usar el repo oficial de Meta sin cuantizar (140GB, no cabe en Kabré)."
        ),
    ),
    "deepseek-r1-distill-llama-70b": ModelConfig(
        name="DeepSeek-R1-Distill-Llama-70B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        quantization="AWQ o bitsandbytes (por definir)",
        max_new_tokens=800,
        do_sample=False,
        notes=(
            "Tercer modelo del benchmark, PENDIENTE DE CONFIRMAR. "
            "Es un modelo de razonamiento (<think>...</think>): necesita "
            "max_new_tokens alto o la respuesta se corta antes de concluir. "
            "Falta evaluar si instruirlo a responder directo sin mostrar el razonamiento."
        ),
    ),
}
