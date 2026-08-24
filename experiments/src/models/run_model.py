"""Carga y generación para los 3 modelos del benchmark, usando la
configuración de config.py.

Cada modelo usa un mecanismo de cuantización distinto (bitsandbytes, AWQ,
compressed-tensors), así que la carga se resuelve explícitamente por
`model_key` en vez de parsear el campo `quantization` de config.py, que es
descriptivo para humanos, no una API.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_CONFIGS

# La plantilla de chat de los modelos de razonamiento (DeepSeek-R1) ya abre
# <think> como parte del prompt, antes de que el modelo genere nada. Basta
# con cerrarlo para que el modelo salte el bloque de razonamiento y genere
# la respuesta final directamente. Es una técnica a nivel de tokens, no un
# prompt en lenguaje natural pidiéndole que no razone.
FORCED_THINK_CLOSE = "\n\n</think>\n\n"


def strip_reasoning(raw_output: str) -> str:
    """Devuelve solo el texto después de `</think>`, descartando el bloque
    de razonamiento.

    No forma parte del flujo de generación por defecto: con
    `skip_reasoning=True` el modelo ya no produce un bloque de razonamiento
    que recortar (ver `build_generate_fn`). Esta función existe para
    normalizar resultados en corridas donde se decida NO saltar el
    razonamiento (ej. una corrida de ablación con `skip_reasoning=False`
    para comparar contra la versión saltada), antes de que esos textos
    lleguen a evaluation/.

    Si no encuentra `</think>` en el texto, la generación se cortó antes de
    terminar de razonar (max_new_tokens insuficiente) — en ese caso NO
    devuelve el texto crudo, porque contaminaría las métricas con una
    respuesta a medio razonar; devuelve un marcador explícito en su lugar.
    """
    if "</think>" in raw_output:
        return raw_output.split("</think>", 1)[-1].strip()
    return "[INCOMPLETE_REASONING: generation cut off before </think>; increase max_new_tokens]"


def _load_qwen2_5_32b(hf_repo: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def _load_llama3_3_70b(hf_repo: str):
    # AWQ pre-cuantizado: transformers detecta la cuantización desde los
    # metadatos del checkpoint, no necesita BitsAndBytesConfig. Requiere
    # `autoawq` instalado.
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def _load_deepseek_r1_distill_llama_70b(hf_repo: str):
    # compressed-tensors w4a16 pre-cuantizado: igual que AWQ, transformers
    # detecta la cuantización automáticamente desde el checkpoint. Requiere
    # `compressed-tensors` instalado (pip install compressed-tensors).
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer


_LOADERS = {
    "qwen2.5-32b": _load_qwen2_5_32b,
    "llama3.3-70b": _load_llama3_3_70b,
    "deepseek-r1-distill-llama-70b": _load_deepseek_r1_distill_llama_70b,
}


def load_model(model_key: str):
    """Carga modelo + tokenizer para `model_key` (una clave de MODEL_CONFIGS)."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Modelo desconocido: {model_key}. Opciones: {list(MODEL_CONFIGS)}")
    if model_key not in _LOADERS:
        raise NotImplementedError(f"Falta definir la carga para '{model_key}' en run_model.py")

    config = MODEL_CONFIGS[model_key]
    model, tokenizer = _LOADERS[model_key](config.hf_repo)
    model.eval()
    return model, tokenizer


def build_generate_fn(model, tokenizer, model_key: str):
    """Devuelve generate_fn(prompt: str) -> str para `model_key`.

    Aplica la plantilla de chat del modelo y, si `skip_reasoning=True` en
    su config, el cierre forzado del bloque <think> antes de generar.
    """
    config = MODEL_CONFIGS[model_key]

    def generate_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if config.skip_reasoning:
            close_ids = tokenizer(FORCED_THINK_CLOSE, add_special_tokens=False, return_tensors="pt")
            encoded["input_ids"] = torch.cat([encoded["input_ids"], close_ids["input_ids"]], dim=1)
            encoded["attention_mask"] = torch.cat(
                [encoded["attention_mask"], torch.ones_like(close_ids["input_ids"])], dim=1
            )

        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate_fn


def get_generate_fn(model_key: str):
    """Carga `model_key` y devuelve su generate_fn lista para usar."""
    model, tokenizer = load_model(model_key)
    return build_generate_fn(model, tokenizer, model_key)


if __name__ == "__main__":
    # Smoke test manual: carga un modelo y prueba una generación corta.
    # Requiere GPU + acceso a Hugging Face (correr en Kabré, no acá).
    #   python run_model.py deepseek-r1-distill-llama-70b
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_CONFIGS:
        print(f"Uso: python run_model.py <model_key>\nOpciones: {list(MODEL_CONFIGS)}")
        sys.exit(1)

    model_key = sys.argv[1]
    config = MODEL_CONFIGS[model_key]
    print(f"Cargando {model_key} ({config.hf_repo})...")
    generate_fn = get_generate_fn(model_key)
    print("Carga exitosa. Probando generación...")
    answer = generate_fn("¿Cuál es la capital de Costa Rica?")
    print(f"Respuesta: {answer}")
