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
        max_new_tokens=600,
        do_sample=False,
        skip_reasoning=True,
        notes=(
            "CONFIRMADO como tercer modelo del benchmark. Es un modelo de "
            "razonamiento (<think>...</think>): se salta el razonamiento "
            "inyectando el token de cierre '</think>' directo en el input "
            "antes de generar (técnica validada por Caleb en su repo), en "
            "vez de pedirlo por prompt en lenguaje natural (menos confiable). "
            "max_new_tokens=600 es una estimación intermedia: más bajo que "
            "los 800 originales porque skip_reasoning elimina el bloque de "
            "pensamiento, pero más alto que los 300 que usó Caleb porque "
            "nuestras preguntas abiertas piden más síntesis que las suyas. "
            "PENDIENTE: verificar que el repo RedHatAI cargue bien con "
            "transformers + compressed-tensors en Kabré; si falla, plan B es "
            "bitsandbytes 4-bit sobre el repo original de DeepSeek, liberando "
            "espacio suficiente en /work primero (~140GB necesarios)."
        ),
    ),
}