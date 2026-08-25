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
    skip_reasoning: bool
    notes: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen2.5-32b": ModelConfig(
        name="Qwen2.5-32B-Instruct",
        hf_repo="Qwen/Qwen2.5-32B-Instruct",
        quantization="bitsandbytes (4-bit u 8-bit, por definir)",
        max_new_tokens=512,
        do_sample=False,
        skip_reasoning=False,
        notes="Mejor desempeño en el mini-experimento de filtrado.",
    ),
    "llama3.3-70b": ModelConfig(
        name="Llama 3.3 70B Instruct",
        hf_repo="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        quantization="AWQ-INT4 (pre-cuantizado)",
        max_new_tokens=512,
        do_sample=False,
        skip_reasoning=False,
        notes=(
            "Requiere aceptar la licencia de Meta en HuggingFace. "
            "NO usar el repo oficial de Meta sin cuantizar (140GB, no cabe en Kabré)."
        ),
    ),
    "deepseek-r1-distill-llama-70b": ModelConfig(
        name="DeepSeek-R1-Distill-Llama-70B",
        hf_repo="RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16",
        quantization="compressed-tensors w4a16 (pre-cuantizado, requiere `pip install compressed-tensors`)",
        max_new_tokens=300,  # TEMPORAL: bajado de 600 para el smoke test, ver nota abajo
        do_sample=False,
        skip_reasoning=True,
        notes=(
            "CONFIRMADO como tercer modelo del benchmark. Es un modelo de "
            "razonamiento (<think>...</think>): se salta el razonamiento "
            "inyectando el token de cierre '</think>' directo en el input "
            "antes de generar (técnica validada por Caleb en su repo), en "
            "vez de pedirlo por prompt en lenguaje natural (menos confiable). "
            "La carga con transformers + compressed-tensors SÍ funciona en "
            "Kabré (confirmado: 43.9GB/46GB VRAM), pero generate() se colgó "
            "en el primer smoke test sin margen para el KV-cache. "
            "max_new_tokens bajado temporalmente a 300 (de 600) y "
            "run_model.py ajustado con max_memory={0: \"40GiB\"} y "
            "attn_implementation=\"sdpa\" para dejar margen real de VRAM. "
            "PENDIENTE: confirmar que esto resuelve el cuelgue y subir "
            "max_new_tokens de vuelta gradualmente (300 -> 600)."
        ),
    ),
}