#!/bin/bash
# Variables de entorno para correr el benchmark en Kabré (ver CLAUDE.md).
# Correr con `source scripts/kabre_setup.sh` después de `module load mamba/HuggingFace`,
# dentro de una sesión tmux, antes de lanzar cualquier script de src/.

module load mamba/HuggingFace

export HF_HOME=/work/squesada/hf_cache
export TRANSFORMERS_CACHE=/work/squesada/hf_cache
export HF_HUB_CACHE=/work/squesada/hf_cache/hub

# NLTK por defecto descarga a ~/nltk_data (home, casi lleno) — necesario
# para METEOR en evaluation/metrics_open.py. Se redirige a /work junto con
# el resto de las cachés.
export NLTK_DATA=/work/squesada/nltk_data

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$NLTK_DATA"
