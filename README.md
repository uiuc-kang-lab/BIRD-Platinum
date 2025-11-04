# BIRD-Platinum: State-of-the-Art Text-to-SQL via Reinforcement Learning on a Clean Dataset

This repository contains the code and data to reproduce the results reported in the BIRD-Platinum paper.

## Repository structure

```text
.
├── README.md
├── baselines                 # Code and setup used to run baselines
├── evaluation                # Generated SQL, grading script, and evaluation results
└── rlvr_tinker               # Code to train models with multi-turn RLVR and BIRD-Platinum
```

## RLVR with BIRD-Platinum

Trained checkpoints are available on Hugging Face.

To train from scratch using Tinker’s API:

1) Set up credentials
```bash
export CLOUDFLARE_ACCESS_CLIENT_ID=<your_id>
export CLOUDFLARE_ACCESS_CLIENT_SECRET=<your_secret>
export TINKER_API_KEY=<your_key>
```

2) Change into the training directory
```bash
cd rlvr_tinker
```

3) Launch training
```bash
bash experiments/bird/run_bird.sh \
  --model <model_name> \
  --add_noise <dataset_variant> \
  --base_dir <dataset_directory> \
  --run_name <name> \
  --learning_rate 5e-5 \
  --n_epochs 10 \
  --max_output_tokens_per_turn 3072 \
  --max_input_tokens 32768 \
  --use_convo_prefix True \
  --use_system_prompt True \
  --renderer_name default
```

4) Key arguments
- model_name: one of
  - Qwen/Qwen3-235B-Instruct-2507
  - Qwen/Qwen3-32B
  - Qwen/Qwen3-8B
  - meta-llama/Llama-3.1-70B
- add_noise: set to False, or use a subset of {'db', 'sql', 'question'}
- base_dir: directory containing the datasets

## Baselines

See the README in each baseline folder for setup and usage:
1) Contextual-SQL: baselines/Contextual-SQL/README.md
2) CSC-SQL: baselines/csc_sql/README.md
3) Gena-SQL: baselines/GenaSQL/README.md
4) OpenSearch-SQL: baselines/OpenSearch-SQL/README.md
5) OmniSQL-32B: baselines/OmniSQL/README.md
6) Arctic-text2SQL-R1-7B: baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md
7) Arctic-ExCot-*B: baselines/ArcticTraining/projects/excot_dpo/README.md
8) SQL-R1-14B: baselines/SQL-R1/README.md

Additional notes:
- SkyRL-SQL
  1) cd baselines/SkyRL-SQL/skyrl-train/
  2) Run:
     ```bash
     bash rl_noise/text-to-sql/grpo_noise.sh --base_dir=<dataset_directory>
     ```
- Infly-RL-SQL-32B
  - Uses the same setup as Arctic-text2SQL-R1-7B. See baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md
- XiYanSQL-32B
  1) cd baselines/XiYanSQL-QwenCoder
  2) Run inference:
     ```bash
     python infer.py --data_path <dataset_path> --db_base_path <db_path>
     ```

## Evaluation

- Generated SQL queries are provided in evaluation/results to reproduce the numbers reported in the paper.
- The grading script is in evaluation/grade.py.

Example (adjust paths/flags as needed):
```bash
python evaluation/grade.py \
  --db_base_path <db_base_path> \
  --infer_results <generated_queries> \
  --data_path <data_path>
```
