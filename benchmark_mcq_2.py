from transformers import pipeline
import pandas as pd
import logging
from pathlib import Path
import gc
import torch
import time

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "benchmarck_3.log"
country_list = ["UK", "Iran", "China", "Azerbaijan"]

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

    outputs = pipe(messages, max_new_tokens=32, do_sample=False, max_length=None)
    log.debug("Generating response %s", outputs)
    return outputs[0][0]["generated_text"][-1]["content"].strip()


def run_benchmark(country_list):
    pipe = load_gemma3()
    df_questions = pd.read_csv("BLEnD/evaluation/mc_data/v1.1/mc_questions_file-1.csv")
    df_questions = df_questions.reset_index(drop=True)

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

        for batch_id, start in enumerate(range(0, total, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, total)
            batch = df_country.iloc[start:end].copy()

            for index, row in batch.iterrows():
                log.info("Question %s", row["MCQID"])
                start_time = time.perf_counter()

                response = generate_gemma3_response(
                    pipe,
                    system_prompt=system_prompt,
                    question=row["prompt"],
                )
                elapsed = time.perf_counter() - start_time

                batch.at[index, "response"] = response
                batch.at[index, "latency_sec"] = elapsed
                log.info("Generated response %s", response)

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


def main():
    setup_logging()
    try:
        log.info("Running benchmark for MCQ")
        run_benchmark(country_list)
    except Exception:
        log.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
