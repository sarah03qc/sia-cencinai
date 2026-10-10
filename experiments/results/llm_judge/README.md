# LLM-as-a-Judge files

`llm_judge_200_fixed_ids.json` is the blinded input for 200 free-form questions

`llm_judge_evaluaciones_600.json` is the archived set of 600 returned evaluations

`llm_judge_model_key.txt` contains the fixed A B C mapping and was kept separate from the judge input

The exact hosted judge runtime is not pinned, so the external inference call is not fully reproducible

All downstream summaries are reproducible from the archived evaluations
