"""BERTScore + ROUGE-L + METEOR para la categoría "abierta".

Ninguna de las 3 métricas necesita GPU: BERTScore corre sobre CPU (más
lento que con GPU, pero funciona), ROUGE-L y METEOR son puro Python. Ver
las instrucciones de instalación en el README/mensaje de evaluate_all.py.
"""

import os
from pathlib import Path

METEOR_LIMITATION_NOTE = (
    "METEOR usa la implementación de nltk.translate.meteor_score, que se "
    "apoya en WordNet y en PorterStemmer — ambos pensados para inglés. "
    "nltk no conecta automáticamente el Open Multilingual WordNet en "
    "español para esta función, y el stemmer sigue siendo el de Porter "
    "(inglés), así que no hace stemming real de palabras en español. En "
    "la práctica, para texto en español METEOR termina comportándose "
    "mucho más cerca de una coincidencia exacta de palabras (similar a "
    "ROUGE/F1) que de una métrica con sinónimos y variantes "
    "morfológicas reales. Es una limitación conocida de la librería, no "
    "un bug de esta implementación — citar en la sección de limitaciones "
    "del paper, no interpretar los valores de METEOR como si tuvieran el "
    "mismo poder de matching semántico que tendrían evaluando inglés."
)

_nltk_ready = False


def _ensure_nltk_data():
    """Descarga los recursos de nltk que faltan (wordnet, omw-1.4, punkt,
    punkt_tab), respetando NLTK_DATA si está seteada para no escribir en
    home (ver scripts/kabre_setup.sh)."""
    global _nltk_ready
    if _nltk_ready:
        return

    import nltk

    nltk_dir = os.environ.get("NLTK_DATA") or str(Path.home() / "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    resource_paths = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for resource, check_path in resource_paths.items():
        try:
            nltk.data.find(check_path)
        except LookupError:
            nltk.download(resource, download_dir=nltk_dir, quiet=True)

    _nltk_ready = True


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """BERTScore F1 por par, con lang="es".

    MODELO REAL USADO (verificado en el log de una corrida real, no supuesto):
    con `bert_score` 0.3.13, `lang="es"` resuelve internamente a
    `bert-base-multilingual-cased` (mBERT) — NO a XLM-RoBERTa-large. El
    mapeo interno de `bert_score` (`lang2model`) no tiene una entrada
    específica para español en esta versión de la librería, así que cae
    al default multilingüe genérico. Decisión: se deja así (no se fuerza
    `model_type="xlm-roberta-large"`), así que citar en la metodología del
    paper que BERTScore usó mBERT (`bert-base-multilingual-cased`), no
    XLM-RoBERTa-large.

    Intenta primero con rescale_with_baseline=True (más interpretable:
    ~0 para un par al azar, ~1 para un match perfecto). Si bert_score no
    tiene una baseline calculada para "es" y tira error, cae a
    rescale_with_baseline=False sin interrumpir la evaluación completa.
    """
    from bert_score import score as bertscore_score

    try:
        _, _, f1 = bertscore_score(
            predictions, references, lang="es", rescale_with_baseline=True, verbose=False,
        )
        used_baseline = True
    except Exception:
        _, _, f1 = bertscore_score(
            predictions, references, lang="es", rescale_with_baseline=False, verbose=False,
        )
        used_baseline = False

    if not used_baseline:
        print("BERTScore: no hay baseline para 'es', se usó rescale_with_baseline=False.")

    return f1.tolist()


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """ROUGE-L F-measure por par, con rouge_score (Google)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


def compute_meteor(predictions: list[str], references: list[str]) -> list[float]:
    """METEOR por par, con nltk. Ver METEOR_LIMITATION_NOTE arriba."""
    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower()) if isinstance(pred, str) and pred.strip() else []
        ref_tokens = word_tokenize(ref.lower()) if isinstance(ref, str) and ref.strip() else []
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        scores.append(meteor_score([ref_tokens], pred_tokens))
    return scores


def evaluate_open(resultados: list[dict]) -> dict:
    """Calcula BERTScore, ROUGE-L y METEOR por pregunta para la categoría
    "abierta" de una lista de resultados crudos.

    Returns:
        {
          "promedios": {"bertscore": float | None, "rouge_l": float | None, "meteor": float | None},
          "detalle": [{"pregunta_id", "categoria", "bertscore", "rouge_l", "meteor"}, ...],
          "nota_meteor": str,
        }
    """
    abiertas = [item for item in resultados if item.get("categoria") == "abierta"]

    if not abiertas:
        return {
            "promedios": {"bertscore": None, "rouge_l": None, "meteor": None},
            "detalle": [],
            "nota_meteor": METEOR_LIMITATION_NOTE,
        }

    predictions = [item.get("respuesta_generada") or "" for item in abiertas]
    references = [item.get("ground_truth") or "" for item in abiertas]

    bertscore_f1 = compute_bertscore(predictions, references)
    rouge_l_f1 = compute_rouge_l(predictions, references)
    meteor = compute_meteor(predictions, references)

    detalle = [
        {
            "pregunta_id": item.get("pregunta_id"),
            "categoria": "abierta",
            "bertscore": bs,
            "rouge_l": rl,
            "meteor": mt,
        }
        for item, bs, rl, mt in zip(abiertas, bertscore_f1, rouge_l_f1, meteor)
    ]

    promedios = {
        "bertscore": sum(bertscore_f1) / len(bertscore_f1),
        "rouge_l": sum(rouge_l_f1) / len(rouge_l_f1),
        "meteor": sum(meteor) / len(meteor),
    }

    return {"promedios": promedios, "detalle": detalle, "nota_meteor": METEOR_LIMITATION_NOTE}
