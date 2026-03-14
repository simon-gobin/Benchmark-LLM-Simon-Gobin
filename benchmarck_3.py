from transformers import pipeline
import csv
import importlib.util
from pathlib import Path
import logging
import re
import unicodedata
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from BLEnD import ANNOTATIONS_DIR, BLEnD_EVAL_DIR

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "benchmarck_3.log"
COUNTRY = "Algeria"
LANGUAGE = "English"
PROMPT_ID = "debug-system"
MODEL_LABEL = "gemma3_1b_it"
PREDICTIONS_FILE = OUTPUT_DIR / f"{MODEL_LABEL}-{COUNTRY}_{LANGUAGE}_{PROMPT_ID}_result.csv"
CLEANED_PREDICTIONS_FILE = OUTPUT_DIR / f"{MODEL_LABEL}-{COUNTRY}_{LANGUAGE}_{PROMPT_ID}_clean.csv"
EVAL_RESULTS_FILE = OUTPUT_DIR / "evaluation_results.csv"


QUESTIONS_DIR = Path("/Users/simon/PycharmProjects/LLM_assignement_1_simon_gobin/BLEnD/data/questions")


def load_questions(country: str) -> list[dict]:
    question_file = QUESTIONS_DIR / f"{country}_questions.csv"

    with question_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_deep_seek():
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B"
    )
    return pipe

def load_gemma():
    pipe = pipeline("text-generation", model="google/gemma-2b")
    return pipe

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model


def generate_response(pipe, prompt: str) -> str:
    outputs = pipe(prompt, max_new_tokens=256)
    generated_text = outputs[0]["generated_text"]

    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    return response


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.casefold().strip()
    normalized = " ".join(normalized.split())
    return normalized


def clean_answer(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_lines = []
    stop_prefixes = (
        "let me think",
        "okay",
        "explanation",
        "step-by-step explanation",
        "options:",
        "question:",
        "###",
        "```",
    )
    skip_exact = {"answer:", "**final answer**", "final answer"}

    for line in lines:
        lowered = line.casefold()
        if lowered in skip_exact:
            continue
        if any(lowered.startswith(prefix) for prefix in stop_prefixes):
            continue
        if re.fullmatch(r"[-*]?\s*[a-d]\.?", lowered):
            continue
        if lowered.startswith("the answer is"):
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        cleaned_lines = lines

    candidate = cleaned_lines[0] if cleaned_lines else text

    candidate = re.split(r"\b(?:let me think|okay|explanation|options:|question:)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
    candidate = re.sub(r"\*\*final answer\*\*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^answer:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip("`#*- \n\t")

    return normalize_answer(candidate)


def clean_predictions_csv() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing raw predictions file: {PREDICTIONS_FILE}")

    with PREDICTIONS_FILE.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)

    cleaned_rows = []
    for row in rows:
        cleaned_row = dict(row)
        cleaned_row["raw_response"] = row["response"]
        cleaned_row["response"] = clean_answer(row["response"])
        cleaned_rows.append(cleaned_row)

    with CLEANED_PREDICTIONS_FILE.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["ID", "country", "prompt_id", "question", "prompt", "response", "raw_response"],
        )
        writer.writeheader()
        writer.writerows(cleaned_rows)

    log.info("Saved cleaned predictions to %s", CLEANED_PREDICTIONS_FILE)


def load_blend_evaluator():
    evaluate_path = BLEnD_EVAL_DIR / "evaluate.py"
    import sys
    sys.path.insert(0, str(BLEnD_EVAL_DIR))
    spec = importlib.util.spec_from_file_location("blend_evaluate", evaluate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load BLEnD evaluator from {evaluate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_blend_evaluation() -> None:
    evaluator = load_blend_evaluator()
    evaluator.evaluate_all_metrics(
        model=MODEL_LABEL,
        country=COUNTRY,
        language=LANGUAGE,
        prompt_no=PROMPT_ID,
        response_dir=str(OUTPUT_DIR),
        annotation_dir=str(ANNOTATIONS_DIR),
        mc_dir=str(BLEnD_EVAL_DIR / "mc_data"),
        id_col="ID",
        q_col="question",
        r_col="response",
        annotations_key="annotations",
        eval_res_filename=str(EVAL_RESULTS_FILE),
        annotation_template="{country}_data.json",
    )
    log.info("Saved BLEnD evaluation to %s", EVAL_RESULTS_FILE)


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


def save_predictions(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "country", "prompt_id", "question", "prompt", "response"],
        )
        writer.writeheader()
        writer.writerows(rows)

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def output_message_tok(model, tokenizer, messages) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=125,
        temperature = 0,
        do_sample=False,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content.strip()


def load_gemma3():
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device="mps",   # use "mps" or -1 depending on your machine
        #torch_dtype=torch.bfloat16,
    )
    return pipe


def generate_gemma3_response(pipe, system_prompt: str, user_prompt: str) -> str:
    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]]

    outputs = pipe(messages, max_new_tokens=64)
    return outputs[0][0]["generated_text"][-1]["content"].strip()



def test():
    #model_list = ["google/gemma-2b", "Qwen/Qwen3-0.6B"]
    #model_name = model_list[0]
    #tokenizer, model = load_model(model_name)
    pipe = load_gemma3()

    df = pd.read_csv(str(QUESTIONS_DIR / f"{COUNTRY}_questions.csv"))

    system_prompt = (
        "Requirements: consider you are from Algeria and answer the given question. "
        "Answers should relate to your cultural or country-specific background. "
        "The answer should be short/concrete. In general, each question will have one answer only."
    )

    output_file = PREDICTIONS_FILE

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "country", "prompt_id", "question", "prompt", "response"],
        )
        writer.writeheader()

        for _, row in df.iterrows():
            question = row["Translation"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            #response = output_message_tok(model, tokenizer, messages)
            response = generate_gemma3_response(pipe, system_prompt, question)

            writer.writerow(
                {
                    "ID": row["ID"],
                    "country": COUNTRY,
                    "prompt_id": PROMPT_ID,
                    "question": question,
                    "prompt": system_prompt,
                    "response": response,
                }
            )

            log.info("ID: %s", row["ID"])
            log.info("Response: %s", response)

def main():
    setup_logging()
    try:
        test()
        clean_predictions_csv()
        run_blend_evaluation()
    except Exception:
        log.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
