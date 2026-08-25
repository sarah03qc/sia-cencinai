"""Convierte experiments/data/benchmark_300.xlsx a benchmark_300.json.

El Excel tiene 3 hojas ("Sí o No", "Respuesta Corta", "Abiertas"), cada una
con columnas N° / Pregunta / Respuesta propuesta y 100 filas de datos. Se
aplana a una sola lista de 300 objetos {id, categoria, pregunta,
ground_truth}, con id = prefijo de categoría + N° con 3 dígitos
(ej. "sino_001").

Uso (desde cualquier ubicación):
    python experiments/scripts/excel_to_json.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # experiments/
XLSX_PATH = PROJECT_ROOT / "data" / "benchmark_300.xlsx"
JSON_PATH = PROJECT_ROOT / "data" / "benchmark_300.json"

# Nombre de hoja en el Excel -> prefijo de categoría en el JSON de salida.
SHEET_TO_CATEGORIA = {
    "Sí o No": "sino",
    "Respuesta Corta": "corta",
    "Abiertas": "abierta",
}

REQUIRED_COLUMNS = {"N°", "Pregunta", "Respuesta propuesta"}
QUESTIONS_PER_SHEET = 100


def convert(xlsx_path: Path = XLSX_PATH, json_path: Path = JSON_PATH) -> list[dict]:
    all_questions = []

    for sheet_name, categoria in SHEET_TO_CATEGORIA.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        df.columns = [str(c).strip() for c in df.columns]

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Hoja '{sheet_name}': faltan columnas {missing}. "
                f"Columnas encontradas: {list(df.columns)}"
            )

        if len(df) != QUESTIONS_PER_SHEET:
            raise ValueError(
                f"Hoja '{sheet_name}': se esperaban {QUESTIONS_PER_SHEET} filas de datos, "
                f"se encontraron {len(df)}."
            )

        for _, row in df.iterrows():
            numero = int(row["N°"])
            all_questions.append({
                "id": f"{categoria}_{numero:03d}",
                "categoria": categoria,
                "pregunta": str(row["Pregunta"]).strip(),
                "ground_truth": str(row["Respuesta propuesta"]).strip(),
            })

    expected_total = QUESTIONS_PER_SHEET * len(SHEET_TO_CATEGORIA)
    if len(all_questions) != expected_total:
        raise ValueError(f"Se esperaban {expected_total} preguntas en total, se generaron {len(all_questions)}.")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    return all_questions


if __name__ == "__main__":
    questions = convert()
    print(f"{len(questions)} preguntas escritas en {JSON_PATH}")
