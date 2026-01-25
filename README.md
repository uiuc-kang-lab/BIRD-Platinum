# BIRD-Platinum: State-of-the-Art Text-to-SQL via Reinforcement Learning on a Clean Dataset

This repository provides:
- A cleaned Text-to-SQL dataset (BIRD-Platinum) for robust training.
- Multi-turn RLVR training recipes built on Tinker.
- Baseline implementations and ready-to-run evaluation scripts.

## Repository Layout

```text
.
├── README.md
├── data/                      # BIRD and Evaluation datasets (Original and Platinum)
├── baselines/                 # Code and setup used to run baselines
├── evaluation/                # Grading script and saved evaluation results
└── rlvr_tinker/               # Tinker-based RLVR training and recipes
```

## Data and Model Release

We have released training data, evaluation data, and trained model checkpoints.

- Data artifacts in [data](data):
  - [data/BIRD-Platinum-600.json](data/BIRD-Platinum-600.json): Platinum-cleaned 600-example subset used in our experiments.
  - [data/BIRD-Original-600.json](data/BIRD-Original-600.json): Original BIRD 600-example subset for comparison.
  - [data/corrected_spider1.csv](data/corrected_spider1.csv): Supporting corrections snapshot used during spider evaluation.

- Model checkpoints (Tinker paths) in [rlvr_tinker/checkpoints.json](rlvr_tinker/checkpoints.json):

Download via Tinker REST client:

```python
import tinker

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

tinker_path = "tinker://<uuid>/sampler_weights/epoch00_batch000100"  # choose from above
future = rest_client.download_checkpoint_archive_from_tinker_path(tinker_path)
with open("model-checkpoint.tar.gz", "wb") as f:
    f.write(future.result())
```

or CLI:

```bash
tinker checkpoint download $TINKER_CHECKPOINT_PATH
```

## Quick Start

Evaluate a set of generated SQL queries against BIRD databases:

```bash
python evaluation/grade.py \
  --db_base_path <db_base_path> \
  --infer_results <path/to/generated_queries.json> \
  --data_path <path/to/BIRD-Platinum-*.json>
```

- See example results in [evaluation/results](evaluation/results).
- The grader accepts paths to BIRD-Original or BIRD-Platinum JSONs from [data](data).

## Data

This repo includes curated BIRD splits to support cleaner evaluation and training:
- [data/BIRD-Platinum-600.json](data/BIRD-Platinum-600.json): Platinum-cleaned subset used in our experiments.
- [data/BIRD-Original-600.json](data/BIRD-Original-600.json): Original subset for comparison.

You will also need the BIRD database files referenced by the benchmark (set via `--db_base_path`).

## Train with RLVR (Tinker)

We build training pipelines on the Tinker SDK. For installation and concepts, see the [rlvr_tinker README](rlvr_tinker/README.md).

Prerequisites:
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

Each baseline has its own setup and scripts; consult the linked READMEs:
1) Contextual-SQL: [baselines/Contextual-SQL/README.md](baselines/Contextual-SQL/README.md)
2) CSC-SQL: [baselines/csc_sql/README.md](baselines/csc_sql/README.md)
3) GenaSQL: [baselines/GenaSQL/README.md](baselines/GenaSQL/README.md)
4) OpenSearch-SQL: [baselines/OpenSearch-SQL/readme.md](baselines/OpenSearch-SQL/readme.md)
5) OmniSQL-32B: [baselines/OmniSQL/README.md](baselines/OmniSQL/README.md)
6) Arctic-text2SQL-R1-7B: [baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md](baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md)
7) Arctic-ExCoT-*B: [baselines/ArcticTraining/projects/excot_dpo/README.md](baselines/ArcticTraining/projects/excot_dpo/README.md)
8) SQL-R1-14B: [baselines/SQL-R1/README.md](baselines/SQL-R1/README.md)

Additional notes:
- SkyRL-SQL
  1) `cd` to [baselines/SkyRL-SQL/skyrl-train](baselines/SkyRL-SQL/skyrl-train)
  2) Run:
     ```bash
     bash rl_noise/text-to-sql/grpo_noise.sh --base_dir=<dataset_directory>
     ```
- Infly-RL-SQL-32B
  - Follows Arctic-text2SQL-R1-7B setup. See [baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md](baselines/ArcticTraining/projects/arctic_text2sql_r1/README.md).
- XiYanSQL-32B
  1) `cd` to [baselines/XiYanSQL](baselines/XiYanSQL)
  2) Run inference:
     ```bash
     python infer.py --data_path <dataset_path> --db_base_path <db_path>
     ```

## Evaluation

- Generated SQL queries can be graded with [evaluation/grade.py](evaluation/grade.py).
- Saved evaluation outputs are in [evaluation/results](evaluation/results).

Example grading (adjust paths as needed):
```bash
python evaluation/grade.py \
  --db_base_path <db_base_path> \
  --infer_results <generated_queries> \
  --data_path <data_path>
```
