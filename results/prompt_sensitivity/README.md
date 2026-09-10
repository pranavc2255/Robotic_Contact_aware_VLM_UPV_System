# Prompt sensitivity: system prompts

Five system prompts, copied unchanged from the saved Qwen2.5-32B study on
89 candidate pairs across 21 scenes:

- `prompts/original_system.txt`: original wording; reported original results
  reuse historical predictions, not a fresh control run.
- `prompts/paraphrase_system.txt`: paraphrased wording.
- `prompts/instruction_order_system.txt`: reordered instructions.
- `prompts/concise_system.txt`: concise wording.
- `prompts/terminology_system.txt`: alternative terminology.

Source run: `outputs/qwen25_32b_prompt_sensitivity89/20260908_144111_377965`.
The study retained the user prompt/schema and varied the system prompt. These
files contain system prompts only, not full per-scene requests, input images,
responses or results. No inference was rerun to prepare this addition.
