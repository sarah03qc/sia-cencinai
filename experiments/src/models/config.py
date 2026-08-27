"""Configuración de los 3 modelos del benchmark

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
    "deepseek-r1-distill-qwen-32b": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-32B",
        hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quantization="bitsandbytes (4-bit u 8-bit, por definir)",
        max_new_tokens=700,
        do_sample=False,
        skip_reasoning=True,
        notes=(
            "DECISIÓN FINAL como tercer modelo del benchmark (ya no pendiente). "
            "Se bajó de DeepSeek-R1-Distill-Llama-70B a este de 32B después de "
            "varios intentos fallidos con la versión de 70B en Kabré: fallo de "
            "compilación GGUF, quota de disco excedida con el repo original sin "
            "cuantizar, y torch.OutOfMemoryError con la versión cuantizada "
            "compressed-tensors (caching_allocator_warmup de transformers no "
            "dejaba margen suficiente en una GPU de 46GB). 32B es el mismo "
            "tamaño que Qwen2.5-32B, que ya carga sin problemas con "
            "bitsandbytes en esta misma infraestructura. Sigue siendo un "
            "modelo de razonamiento (<think>...</think>) de la familia "
            "R1-Distill: se salta el razonamiento con el mismo cierre forzado "
            "de '</think>' inyectado en el input antes de generar (técnica "
            "validada por Caleb en su repo). max_new_tokens=700: valor "
            "generoso otra vez porque ya no estamos al límite de VRAM como "
            "con el 70B, pero sigue dejando espacio para el bloque <think>."
        ),
    ),
}