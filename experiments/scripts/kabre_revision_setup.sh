#!/bin/bash
# Limpia rutas de Python que pueden contaminar el entorno
unset PYTHONPATH
unset PYTHONHOME

# Activa el entorno usado para las métricas de revisión
source /work/squesada/venvs/cencinai-revision/bin/activate

# Mantiene los paquetes y las cachés fuera del home
export PYTHONNOUSERSITE=1
export HF_HOME=/work/squesada/.cache/huggingface
export NLTK_DATA=/work/squesada/.cache/nltk
export XDG_CACHE_HOME=/work/squesada/.cache

hash -r
