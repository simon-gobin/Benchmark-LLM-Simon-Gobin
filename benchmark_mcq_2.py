from transformers import pipeline, GenerationConfig
import pandas as pd
import logging
from pathlib import Path
import gc
import torch
import time
import json
import re
import warnings



log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "benchmarck_3.log"
country_list = ["UK", "Iran", "China", "Azerbaijan"]
MCQ_FILE = PROJECT_ROOT / "data" / "mc_data" / "v1.1" / "mc_questions_file-1.csv"

def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )



def load_gemma3():
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device=device,
    )
    log.info("Loading gemma model on %s", "cuda" if device == 0 else "cpu")
    return pipe

def generate_gemma3_response(pipe, system_prompt: str, question: str) -> str:
    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        },
    ]]

    gen_config = GenerationConfig(
        max_new_tokens=32,
        do_sample=False,
        max_length=None # Also helps with the previous length warning
    )

    # Pass the config object instead of individual arguments
    outputs = pipe(messages, generation_config=gen_config)
    log.debug("Generating response %s", outputs)
    return outputs[0][0]["generated_text"][-1]["content"].strip()


def run_benchmark(country_list, MCQ_FILE):
    pipe = load_gemma3()
    df_questions = pd.read_csv(MCQ_FILE)

    system_prompt = (
        "You need to answer multiple-choice questions. "
        "Only answer with one letter: A, B, C, or D. "
        "Do not provide any explanation."
    )

    BATCH_SIZE = 500
    BATCH_DIR = OUTPUT_DIR / "batches"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = []

    for country in country_list:
        df_country = df_questions[df_questions["country"] == country].copy()
        total = len(df_country)
        log.info(f"Generating {total} questions for country {country}")

        for batch_id, start in enumerate(range(0, total, BATCH_SIZE)):

            end = min(start + BATCH_SIZE, total)
            batch = df_country.iloc[start:end].copy()
            log.info("Batch %s started for %s rows %s-%s", batch_id + 1, country, start + 1, end)

            for index, row in batch.iterrows():
                log.debug("Question %s", row["MCQID"])
                start_time = time.perf_counter()

                response = generate_gemma3_response(
                    pipe,
                    system_prompt=system_prompt,
                    question=row["prompt"],
                )
                elapsed = time.perf_counter() - start_time

                batch.at[index, "response"] = response
                batch.at[index, "latency_sec"] = elapsed
                log.debug("Generated response %s", response)

            batch_file = BATCH_DIR / f"{country}_batch_{batch_id:04d}.csv"
            batch.to_csv(batch_file, index=False)
            batch_files.append(batch_file)

            log.info("Saved %s (%s/%s)", batch_file, end, total)

            del batch
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_df = pd.concat((pd.read_csv(file) for file in batch_files), ignore_index=True)
    final_output = OUTPUT_DIR / "questions_answer.csv"
    final_df.to_csv(final_output, index=False)
    log.info("Saved answers to %s", final_output)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

def run_evaluation():
    log.info("Running evaluation")
    input_file = OUTPUT_DIR / "questions_answer.csv"
    output_file = OUTPUT_DIR / "questions_answer_evaluated.csv"
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
    per_country["model"] = "gemma3_1b_it"
    per_country["language"] = "English"
    per_country["prompt_no"] = "mcq-debug"
    per_country["eval_method"] = "mcq_accuracy"

    overall = pd.DataFrame(
        [{
            "country": "OVERALL",
            "score": df["is_correct_int"].mean(),
            "model": "gemma3_1b_it",
            "language": "English",
            "prompt_no": "mcq-debug",
            "eval_method": "mcq_accuracy",
        }]
    )

    summary_df = pd.concat([per_country, overall], ignore_index=True)
    summary_df = summary_df[["model", "country", "language", "prompt_no", "eval_method", "score"]]
    summary_df.to_csv(summary_file, index=False)

    log.info("Saved evaluated answers to %s", output_file)
    log.info("Saved evaluation summary to %s", summary_file)


def main():
    setup_logging()
    try:
        log.info("Running benchmark for MCQ")
        run_benchmark(country_list, MCQ_FILE)
        run_evaluation()
        log.info("Finished benchmark for MCQ")
    except Exception:
        log.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
