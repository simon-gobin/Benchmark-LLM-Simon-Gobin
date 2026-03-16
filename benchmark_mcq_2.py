from transformers import pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import logging
from pathlib import Path
import gc
import torch
import time
import json
import re

EXPERIMENTS = {
    "qwen_locale": {
        "model_name": "Qwen/Qwen3-8B",
        "model_label": "qwen3_8b",
        "backend": "qwen",
        "prompt_id": "locale_aware_confidence",
    },
    "gemma_baseline": {
        "model_name": "google/gemma-3-12b-it",
        "model_label": "gemma3_12b_it",
        "backend": "gemma",
        "prompt_id": "baseline",
    },
    "gemma_locale": {
        "model_name": "google/gemma-3-12b-it",
        "model_label": "gemma3_12b_it",
        "backend": "gemma",
        "prompt_id": "locale_aware_confidence",
    },
}

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "benchmarck_3.log"
country_list = ["UK", "Iran", "China", "Azerbaijan"]
MCQ_FILE = PROJECT_ROOT / "data" / "mc_data" / "v1.1" / "mc_questions_file-1.csv"
PROMPT_CONFIG_FILE = PROJECT_ROOT / "data" / "prompt_configs_mcq.json"
BATCH_SIZE = 500
INFER_BATCH_SIZE = 64
logging_level = logging.INFO

def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_prompt_configs() -> dict:
    with PROMPT_CONFIG_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


#=========================Gemma 3 loadder=================

def load_gemma3(model_name: str):
    if torch.cuda.is_available():
        device = 0
        device_label = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        device_label = "mps"
    else:
        device = -1
        device_label = "cpu"

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device,
    )
    log.info("Loading gemma model on %s", device_label)
    return pipe

def generate_gemma3_responses_batch(pipe, system_prompt: str, questions: list[str]) -> list[str]:
    messages_batch = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            },
        ]
        for question in questions
    ]

    gen_config = GenerationConfig(
        max_new_tokens=32,
        do_sample=False,
        max_length=None,
    )

    outputs = pipe(
        messages_batch,
        generation_config=gen_config,
        batch_size=len(questions),
    )

    responses = []
    for output in outputs:
        log.debug(output)
        responses.append(output[0]["generated_text"][-1]["content"].strip())

    return responses

#=========================Qwen 3 loadder=================

def load_qwen3(model_name="Qwen/Qwen3-14B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def generate_qwen3_responses_batch(model, tokenizer, system_prompt: str, questions: list[str]) -> list[str]:
    texts = []

    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        texts.append(text)

    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32,
        do_sample=False,
    )

    responses = []
    for i in range(len(texts)):
        input_len = model_inputs.input_ids[i].shape[0]
        output_ids = generated_ids[i][input_len:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        responses.append(content)

    return responses

def load_model(backend: str, model_name: str):
    if backend == "gemma":
        return {"backend": "gemma", "pipe": load_gemma3(model_name)}

    if backend == "qwen":
        tokenizer, model = load_qwen3(model_name)
        return {"backend": "qwen", "tokenizer": tokenizer, "model": model}

    raise ValueError(f"Unsupported backend: {backend}")

def generate_responses_batch(model_bundle, system_prompt: str, questions: list[str]) -> list[str]:
    if model_bundle["backend"] == "gemma":
        return generate_gemma3_responses_batch(
            model_bundle["pipe"],
            system_prompt=system_prompt,
            questions=questions,
        )

    if model_bundle["backend"] == "qwen":
        return generate_qwen3_responses_batch(
            model_bundle["model"],
            model_bundle["tokenizer"],
            system_prompt=system_prompt,
            questions=questions,
        )

    raise ValueError(f"Unsupported backend: {model_bundle['backend']}")


def run_benchmark(country_list, mcq_file, batch_size, infer_batch_size, prompt_config, model_bundle, model_label):
    df_questions = pd.read_csv(mcq_file)
    prompt_no = prompt_config["prompt_no"]

    BATCH_DIR = OUTPUT_DIR / "batches"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = []

    for country in country_list:
        system_prompt = prompt_config["system_prompt"].format(country=country)
        log.info(f'sysyem_prompt loaded for country {country} : {system_prompt}')
        df_country = df_questions[df_questions["country"] == country].copy()
        total = len(df_country)
        log.info(f"Generating {total} questions for country {country}")

        for batch_id, start in enumerate(range(0, total, batch_size)):

            end = min(start + batch_size, total)
            batch = df_country.iloc[start:end].copy()
            log.info("Batch %s started for %s rows %s-%s", batch_id + 1, country, start + 1, end)

            responses = []
            latencies = []
            mini_batch_total = (len(batch) + infer_batch_size - 1) // infer_batch_size

            for mini_batch_id, mini_start in enumerate(range(0, len(batch), infer_batch_size)):
                mini_end = min(mini_start + infer_batch_size, len(batch))
                mini_batch = batch.iloc[mini_start:mini_end]
                log.info(
                    "Mini-batch %s/%s for %s rows %s-%s",
                    mini_batch_id + 1,
                    mini_batch_total,
                    country,
                    start + mini_start + 1,
                    start + mini_end,
                )

                start_time = time.perf_counter()
                mini_responses = generate_responses_batch(
                    model_bundle,
                    system_prompt=system_prompt,
                    questions=mini_batch["prompt"].tolist(),
                )
                elapsed = time.perf_counter() - start_time
                avg_latency = elapsed / len(mini_batch)

                responses.extend(mini_responses)
                latencies.extend([avg_latency] * len(mini_batch))

            batch["response"] = responses
            batch["latency_sec"] = latencies

            batch_file = BATCH_DIR / f"{country}_{model_label}_{prompt_no}_batch_{batch_id:04d}.csv"
            batch.to_csv(batch_file, index=False)
            batch_files.append(batch_file)

            log.info("Saved %s (%s/%s)", batch_file, end, total)

            del batch
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_df = pd.concat((pd.read_csv(file) for file in batch_files), ignore_index=True)
    final_output = OUTPUT_DIR / f"questions_answer_{model_label}_{prompt_no}.csv"
    final_df.to_csv(final_output, index=False)
    log.info("Saved answers to %s", final_output)

#========================+Evalution Block================

def normalize_mcq_answer(text: str) -> str:
    raw = (text or "").strip()
    upper = raw.upper()

    if upper in {"A", "B", "C", "D"}:
        return upper

    try:
        data = json.loads(raw)
        value = str(data.get("answer_choice", "")).strip().upper()
        if value in {"A", "B", "C", "D"}:
            return value
    except Exception:
        pass

    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)

    return ""

def run_evaluation(model_label: str, prompt_no: str):
    log.info("Running evaluation")
    input_file = OUTPUT_DIR / f"questions_answer_{model_label}_{prompt_no}.csv"
    output_file = OUTPUT_DIR / f"questions_answer_evaluated_{model_label}_{prompt_no}.csv"
    summary_file = OUTPUT_DIR / "evaluation_results_mcq.csv"

    df = pd.read_csv(input_file)

    df["parsed_answer"] = df["response"].apply(normalize_mcq_answer)
    df["is_correct"] = df["parsed_answer"] == df["answer_idx"]
    df["is_correct_int"] = df["is_correct"].astype(int)

    df.to_csv(output_file, index=False)

    per_country = (
        df.groupby("country", as_index=False)["is_correct_int"]
        .mean()
        .rename(columns={"is_correct_int": "score"})
    )
    per_country["model"] = model_label
    per_country["language"] = "English"
    per_country["prompt_no"] = prompt_no
    per_country["eval_method"] = "mcq_accuracy"

    overall = pd.DataFrame(
        [{
            "country": "OVERALL",
            "score": df["is_correct_int"].mean(),
            "model": model_label,
            "language": "English",
            "prompt_no": prompt_no,
            "eval_method": "mcq_accuracy",
        }]
    )

    summary_df = pd.concat([per_country, overall], ignore_index=True)
    summary_df = summary_df[["model", "country", "language", "prompt_no", "eval_method", "score"]]
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        existing_df = existing_df[
            ~(
                (existing_df["model"] == model_label)
                & (existing_df["language"] == "English")
                & (existing_df["prompt_no"] == prompt_no)
                & (existing_df["eval_method"] == "mcq_accuracy")
            )
        ]
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)

    summary_df.to_csv(summary_file, index=False)

    log.info("Saved evaluated answers to %s", output_file)
    log.info("Saved evaluation summary to %s", summary_file)


def release_model(model_bundle) -> None:
    if model_bundle["backend"] == "gemma":
        del model_bundle["pipe"]
    elif model_bundle["backend"] == "qwen":
        del model_bundle["model"]
        del model_bundle["tokenizer"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    setup_logging()
    try:
        log.info("Running benchmark for MCQ")
        prompt_configs = load_prompt_configs()

        for experiment_name, experiment in EXPERIMENTS.items():
            prompt_config = prompt_configs[experiment["prompt_id"]]
            model_bundle = load_model(experiment["backend"], experiment["model_name"])

            log.info("Running experiment %s", experiment_name)
            run_benchmark(
                country_list,
                MCQ_FILE,
                BATCH_SIZE,
                INFER_BATCH_SIZE,
                prompt_config,
                model_bundle,
                experiment["model_label"],
            )
            run_evaluation(experiment["model_label"], prompt_config["prompt_no"])
            release_model(model_bundle)

        log.info("Finished benchmark for MCQ")
    except Exception:
        log.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
