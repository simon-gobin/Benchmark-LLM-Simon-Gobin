# Benchmark-LLM-Simon-Gobin

Benchmark scripts for culturally grounded question answering with Gemma 3 and Qwen 3 on BLEnD-style multiple-choice questions.

## Project scope

This repository contains the code used to evaluate open-weight large language models on culturally specific everyday-knowledge questions. The main benchmark is based on BLEnD Track B (multiple-choice questions in English with culturally distinct answer options).

Repository URL:

- https://github.com/simon-gobin/Benchmark-LLM-Simon-Gobin

The current project compares:

- `google/gemma-3-12b-it`
- `Qwen/Qwen3-8B`

under both:

- a `baseline` prompt
- a `locale-aware confidence` prompt

The current locale list is:

- `US`
- `UK`
- `Iran`
- `China`
- `Azerbaijan`

## Repository contents

- `benchmark_mcq_2.py`
  Main MCQ benchmark with:
  - model loading
  - batch inference
  - accuracy evaluation
  - post-evaluation sampling
  - structured JSON parsing

- `data/`
  Prompt configurations and BLEnD-style input data.

- `requirements.txt`
  Python dependencies.

## Current experiments

`benchmark_mcq_2.py` currently runs these four experiments:

- `gemma_baseline`
- `qwen_baseline`
- `gemma_locale`
- `qwen_locale`

The locale-aware configuration asks the model to answer with country awareness and emit a structured confidence field.

## Main outputs

All outputs are written under `outputs/`.

Main benchmark files:

- `questions_answer_{model_label}_{prompt_no}.csv`
- `questions_answer_evaluated_{model_label}_{prompt_no}.csv`
- `evaluation_results_mcq.csv`

Post-evaluation files:

- `questions_answer_post_eval_{model_label}_{prompt_no}.csv`

Typical post-evaluation columns include:

- `post_eval_predicted_answer`
- `post_eval_confidence`
- `post_eval_reasoning_summary`
- `post_eval_error_type`
- `post_eval_likely_failure_source`
- `post_eval_mentions_country_specific_cue`
- `post_eval_is_correct`

## Post-evaluation design

The post-evaluation stage is a second-pass reflective analysis step. It does **not** rerun the full benchmark.

Instead, it:

- loads the evaluated MCQ file
- samples correct and incorrect rows
- reruns only that subset with a longer reasoning-oriented prompt
- stores structured fields extracted from the JSON response

The current script supports country-balanced sampling for post-evaluation. In the working configuration used for analysis:

- sampling is balanced by country
- up to `10` correct rows and `10` incorrect rows are sampled per country
- with 5 countries, this yields up to `100` post-evaluation rows total

This makes the post-evaluation analysis much cheaper than rerunning the full benchmark and reduces GPU memory pressure.

## Reproducibility settings

The benchmark script applies explicit global seeding before running any experiment:

- `SEED = 42`
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA, when available

The project also uses deterministic decoding through:

- `do_sample=False`

In addition, post-evaluation sampling uses a fixed random state so that sampled subsets can be reproduced across runs.

## Local setup

Create and activate a virtual environment if you want an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the benchmark:

```bash
python3 benchmark_mcq_2.py
```

## Hugging Face access

`google/gemma-3-12b-it` is a gated model. Before running the benchmark, you need:

1. access approved on your Hugging Face account
2. a valid Hugging Face token
3. login before model loading

Local login:

```bash
python3 -c "from huggingface_hub import login; login()"
```

Colab login with a secret named `HF_TOKEN`:

```python
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get("HF_TOKEN")
login(token=hf_token)
```

## Recommended Colab workflow

Colab notebook used for the benchmark:

- https://colab.research.google.com/drive/1JGspDp7tKdqi8JFHmqv-n4a2aPMmQv9x?usp=sharing

```python
!git clone https://github.com/simon-gobin/Benchmark-LLM-Simon-Gobin.git
%cd Benchmark-LLM-Simon-Gobin
!git checkout main
!pip install -r requirements.txt
!pip install -q huggingface_hub

from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get("HF_TOKEN")
login(token=hf_token)

!python benchmark_mcq_2.py
```

To save outputs to Google Drive:

```python
!mkdir -p /content/drive/MyDrive/benchmark_outputs
!cp -r outputs/* /content/drive/MyDrive/benchmark_outputs
```

## Memory notes

The post-evaluation prompt is longer than the main MCQ prompt and can cause out-of-memory errors more easily.

The script mitigates this by using a dedicated post-evaluation mini-batch size:

- `POST_EVAL_INFER_BATCH_SIZE`

If memory is still tight, reduce:

- `POST_EVAL_INFER_BATCH_SIZE`
- `max_new_tokens` used in post-evaluation

This is especially relevant for Gemma 3 12B, which is much heavier than Qwen 3 8B.

## Reproducibility notes

The benchmark should also be documented with:

- exact model names
- exact locale list
- exact prompt configuration
- fixed global seed
- batch sizes used for inference and post-evaluation
- hardware environment used for the run

## Submission checklist

For assignment submission, make sure to include:

- the PDF report
- the code
- `requirements.txt`
- `README.md`
- prediction / output CSV files used for evaluation
- the exact locale list
- a note about Hugging Face authentication for Gemma
