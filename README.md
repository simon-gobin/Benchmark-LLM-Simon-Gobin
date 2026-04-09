# Benchmark-LLM-Simon-Gobin

Benchmark scripts for cultural question answering with Gemma and Qwen.

## What is in this repo

- `benchmark_mcq_2.py`: multiple-choice benchmark with batch inference, MCQ accuracy evaluation, and a post-evaluation pass for error analysis.
- `benchmarck_3.py`: earlier free-answer benchmark and BLEnD-style evaluation script.
- `data/`: prompts, questions, annotations, and MCQ datasets.
- `requirements.txt`: Python dependencies.

## Current MCQ benchmark setup

`benchmark_mcq_2.py` currently runs these experiments:

- `gemma_baseline`: `google/gemma-3-12b-it` with the baseline MCQ prompt
- `qwen_locale`: `Qwen/Qwen3-8B` with the locale-aware confidence prompt
- `gemma_locale`: `google/gemma-3-12b-it` with the locale-aware confidence prompt

Countries currently included:

- `US`
- `UK`
- `Iran`
- `China`
- `Azerbaijan`

## Outputs

The script writes outputs under `outputs/`.

Main benchmark files:

- `questions_answer_{model_label}_{prompt_no}.csv`
- `questions_answer_evaluated_{model_label}_{prompt_no}.csv`
- `evaluation_results_mcq.csv`

Post-evaluation files:

- `questions_answer_post_eval_{model_label}_{prompt_no}.csv`

The post-evaluation step samples 10% of correct rows and 10% of incorrect rows from the evaluated MCQ CSV, reruns them with a reasoning-oriented prompt, and stores structured JSON-derived fields such as:

- `post_eval_predicted_answer`
- `post_eval_confidence`
- `post_eval_reasoning_summary`
- `post_eval_error_type`
- `post_eval_likely_failure_source`
- `post_eval_mentions_country_specific_cue`
- `post_eval_is_correct`

## Local setup

Create and activate a virtual environment if you want an isolated install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the MCQ benchmark:

```bash
python3 benchmark_mcq_2.py
```

## Hugging Face access

`google/gemma-3-12b-it` is a gated model. You need:

1. Access approved on your Hugging Face account
2. A valid Hugging Face token
3. Authentication before running the script

If you are working locally:

```bash
python3 -c "from huggingface_hub import login; login()"
```

If you are working in Colab, store your token in Colab Secrets, then log in before running the benchmark.

Example with a secret named `HF_TOKEN`:

```python
from google.colab import userdata
from huggingface_hub import login, whoami

hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)
print(whoami())
```

You can verify gated model access with:

```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/gemma-3-12b-it", token=True)
```

## Colab workflow

Example Colab setup:

```python
!git clone https://github.com/simon-gobin/Benchmark-LLM-Simon-Gobin.git
%cd Benchmark-LLM-Simon-Gobin
!git fetch origin
!git checkout codex/post-eval-mcq
!git pull origin codex/post-eval-mcq
!pip install -r requirements.txt
!pip install -q huggingface_hub

from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)

!python benchmark_mcq_2.py
```

To copy outputs to Google Drive:

```python
!mkdir -p /content/drive/MyDrive/benchmark_outputs
!cp -r outputs/* /content/drive/MyDrive/benchmark_outputs
```

## Notes on memory

The post-evaluation prompt is longer and more memory-intensive than the main MCQ benchmark. The current script reduces this risk by using a dedicated post-evaluation mini-batch size:

- `POST_EVAL_INFER_BATCH_SIZE = 4`

If you still hit GPU memory issues, reduce either:

- `POST_EVAL_INFER_BATCH_SIZE`
- `max_new_tokens` in the post-evaluation generation call

## Branch workflow

Current development branch:

- `codex/post-eval-mcq`

Recommended workflow:

1. Validate the benchmark on the feature branch
2. Review the generated outputs
3. Merge or copy the working changes into `main`

