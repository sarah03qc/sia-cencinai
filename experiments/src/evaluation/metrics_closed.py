"""Métricas para las categorías "sino" y "corta".

"corta": F1 a nivel de token (estilo SQuAD) sobre bag-of-words normalizado
— sigue siendo apropiado, no es una respuesta binaria.

"sino": la métrica principal es accuracy sobre la etiqueta Sí/No extraída
de `respuesta_generada` (ver `extract_binary_label`), no F1 de token. El
F1 de token no distingue "acertó pero fue verboso" de "se equivocó"
(ambos dan un score bajo si la respuesta es larga) — ver el detalle real
en la conversación que llevó a este cambio. El F1 de token se sigue
calculando y queda visible en el detalle por pregunta, como referencia.
"""

import re
import string
import unicodedata
from collections import Counter


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = _strip_accents(text)
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """F1 estilo SQuAD sobre bag-of-words normalizado (minúsculas, sin
    acentos, sin puntuación, tokenizado por espacios).

    Devuelve 0.0 si no hay overlap de tokens, incluyendo el caso de que
    `prediction` o `ground_truth` queden vacíos tras normalizar.
    """
    pred_tokens = _normalize_tokens(prediction)
    gold_tokens = _normalize_tokens(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_binary_label(respuesta_generada: str) -> str | None:
    """Extrae la etiqueta Sí/No de una respuesta larga, para la categoría 'sino'.

    Devuelve 'sí', 'no', o None si no se puede determinar con confianza.

    Solo mira si la respuesta ARRANCA con "Sí" o "No" (sin distinguir
    mayúsculas/acentos, ignorando signos de puntuación pegados justo
    después, ej. "No, el Instrumento..." -> "no"). No busca la etiqueta
    en cualquier parte del texto — una respuesta que dice "no" recién a
    mitad de la explicación, sin haber arrancado con la etiqueta, cuenta
    como sin etiqueta clara (None), no como una detección positiva.

    Limitación conocida y aceptada: "si" sin tilde es ambiguo en español
    entre "sí" (afirmación) y "si" (condicional, "si el niño presenta...").
    Como se ignoran acentos, una respuesta que arranca con una oración
    condicional se puede leer como un "sí" afirmativo. No se intenta
    resolver esa ambigüedad aquí — está cubierta por el conteo de "sin
    etiqueta clara" solo cuando no hay match, no cuando hay un falso
    positivo por este motivo.
    """
    if not isinstance(respuesta_generada, str) or not respuesta_generada.strip():
        return None

    normalized = _strip_accents(respuesta_generada.strip().lower())
    normalized = normalized.lstrip(string.punctuation + " ")

    first_word = re.split(r"[\s,.;:!?]", normalized, maxsplit=1)[0]

    if first_word == "si":
        return "sí"
    if first_word == "no":
        return "no"
    return None


def evaluate_closed(resultados: list[dict]) -> dict:
    """Evalúa las categorías "sino" y "corta" de una lista de resultados
    crudos (con claves "categoria", "respuesta_generada", "ground_truth").

    "sino": accuracy sobre la etiqueta extraída con `extract_binary_label`
    (aplicada tanto a `respuesta_generada` como a `ground_truth`, que ya
    viene limpio como "Sí"/"No" pero se pasa por el mismo extractor para
    no duplicar la lógica de normalización). Una respuesta sin etiqueta
    clara (None) cuenta como incorrecta en el accuracy, pero se reporta
    aparte en "sino_sin_etiqueta_clara" — no se descarta en silencio.

    "corta": F1 de token, sin cambios.

    Preguntas con "error" o respuesta vacía cuentan igual (como
    incorrectas / F1=0.0, según corresponda) — un error de generación es
    una respuesta incorrecta para efectos de la métrica, no un dato
    faltante a ignorar.

    Returns:
        {
          "promedios": {
            "sino_accuracy": float | None,
            "sino_f1_token_promedio": float | None,  # referencia, no la métrica principal
            "sino_sin_etiqueta_clara": int,
            "corta": float | None,
          },
          "detalle": [
            # categoria == "sino":
            {"pregunta_id", "categoria", "f1_token", "etiqueta_predicha", "etiqueta_esperada", "acierto"},
            # categoria == "corta":
            {"pregunta_id", "categoria", "f1_token"},
          ]
        }
        Los promedios quedan en None si esa categoría no tiene preguntas
        en `resultados` (para no reportar un 0.0 engañoso).
    """
    detalle = []

    sino_f1_scores: list[float] = []
    sino_aciertos = 0
    sino_con_etiqueta = 0
    sino_sin_etiqueta_clara = 0
    sino_total = 0

    corta_f1_scores: list[float] = []

    for item in resultados:
        categoria = item.get("categoria")
        if categoria not in ("sino", "corta"):
            continue

        prediction = item.get("respuesta_generada") or ""
        ground_truth = item.get("ground_truth") or ""
        f1_token = compute_f1(prediction, ground_truth)

        if categoria == "corta":
            corta_f1_scores.append(f1_token)
            detalle.append({
                "pregunta_id": item.get("pregunta_id"),
                "categoria": categoria,
                "f1_token": f1_token,
            })
            continue

        # categoria == "sino"
        sino_total += 1
        sino_f1_scores.append(f1_token)

        etiqueta_predicha = extract_binary_label(prediction)
        etiqueta_esperada = extract_binary_label(ground_truth)

        if etiqueta_predicha is None:
            sino_sin_etiqueta_clara += 1
            acierto = False
        else:
            sino_con_etiqueta += 1
            acierto = etiqueta_predicha == etiqueta_esperada
            if acierto:
                sino_aciertos += 1

        detalle.append({
            "pregunta_id": item.get("pregunta_id"),
            "categoria": categoria,
            "f1_token": f1_token,
            "etiqueta_predicha": etiqueta_predicha,
            "etiqueta_esperada": etiqueta_esperada,
            "acierto": acierto,
        })

    promedios = {
        "sino_accuracy": (sino_aciertos / sino_total) if sino_total else None,
        "sino_f1_token_promedio": (sum(sino_f1_scores) / len(sino_f1_scores)) if sino_f1_scores else None,
        "sino_sin_etiqueta_clara": sino_sin_etiqueta_clara,
        "corta": (sum(corta_f1_scores) / len(corta_f1_scores)) if corta_f1_scores else None,
    }

    return {"promedios": promedios, "detalle": detalle}
