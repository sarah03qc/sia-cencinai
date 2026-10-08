import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

FILES = {
    "qwen": RAW / "qwen2.5-32b.json",
    "llama": RAW / "llama3.3-70b.json",
    "deepseek": RAW / "deepseek-r1-distill-qwen-32b.json",
}

def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(x["pregunta_id"]): x for x in data}

def chunks(item):
    return [
        (x["source_document"], int(x["chunk_index"]))
        for x in (item.get("chunks_usados") or [])
    ]

data = {name: load(path) for name, path in FILES.items()}

print("Preguntas por modelo:")
for name, items in data.items():
    print(f"  {name}: {len(items)}")

ids_q = set(data["qwen"])
ids_l = set(data["llama"])
ids_d = set(data["deepseek"])

if not (ids_q == ids_l == ids_d):
    print("\nERROR: los modelos no contienen exactamente los mismos IDs")
    print("Qwen-Llama:", sorted(ids_q ^ ids_l)[:20])
    print("Qwen-DeepSeek:", sorted(ids_q ^ ids_d)[:20])
    sys.exit(1)

mismatches = []
missing_chunks = []

for qid in sorted(ids_q):
    cq = chunks(data["qwen"][qid])
    cl = chunks(data["llama"][qid])
    cd = chunks(data["deepseek"][qid])

    if len(cq) != 5 or len(cl) != 5 or len(cd) != 5:
        missing_chunks.append((qid, len(cq), len(cl), len(cd)))

    if not (cq == cl == cd):
        mismatches.append({
            "id": qid,
            "qwen": cq,
            "llama": cl,
            "deepseek": cd,
        })

print()
print("=" * 65)
print("RESULTADO")
print("=" * 65)

total = len(ids_q)
identical = total - len(mismatches)

print(f"Preguntas comparadas: {total}")
print(f"Retrievals idénticos: {identical}/{total}")
print(f"Retrievals diferentes: {len(mismatches)}")
print(f"Preguntas sin exactamente 5 chunks: {len(missing_chunks)}")

if missing_chunks:
    print("\nCasos con cantidad distinta de 5:")
    for x in missing_chunks[:20]:
        print(x)

if mismatches:
    print("\nPrimeros retrievals diferentes:")
    for m in mismatches[:10]:
        print("\nID:", m["id"])
        print("Qwen:    ", m["qwen"])
        print("Llama:   ", m["llama"])
        print("DeepSeek:", m["deepseek"])
    sys.exit(1)

print("\nOK: los 3 modelos recibieron exactamente los mismos Top-5.")
