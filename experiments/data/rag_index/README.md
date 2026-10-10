# RAG index files

The complete FAISS index and full `chunks.json` are not included in this package

`retrieved_chunks_snapshot.json` contains every unique chunk that appeared in the historical Top-5 used by the three benchmark runs

That snapshot is enough to rebuild `historical_top5_final_299.json` from the archived raw model outputs

Rebuilding the original FAISS index still requires the seven source PDFs
