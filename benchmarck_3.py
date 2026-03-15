from transformers import pipeline
import csv
import json
from pathlib import Path
import logging
import gc
import re
import unicodedata
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from BLEnD import ANNOTATIONS_DIR, COUNTRY_LANG

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "benchmarck_3.log"
COUNTRY_LIST = ["UK", "China", "Iran", "Algeria"]
LANGUAGE = "English"
PROMPT_ID = "debug-system"
MODEL_LABEL = "gemma3_1b_it"
EVAL_RESULTS_FILE = OUTPUT_DIR / "evaluation_results.csv"


QUESTIONS_DIR = Path("/data/questions")


def get_predictions_file(country: str, language: str) -> Path:
    return OUTPUT_DIR / f"{MODEL_LABEL}-{country}_{language}_{PROMPT_ID}_result.csv"


def get_question_results_file(country: str, language: str) -> Path:
    return OUTPUT_DIR / f"evaluation_details_{country}_{language}.csv"


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


def tokenize_for_match(text: str, language: str) -> list[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []

    if language == "Chinese":
        try:
            import jieba
            return [token for token in jieba.cut(normalized) if token.strip()]
        except Exception:
            return normalized.split()

    if language == "Korean":
        try:
            from konlpy.tag import Okt
            return [token for token in Okt().morphs(normalized) if token.strip()]
        except Exception:
            return normalized.split()

    if language == "Arabic":
        try:
            from qalsadi.lemmatizer import Lemmatizer
            lemmatizer = Lemmatizer()
            lemmas = lemmatizer.lemmatize(normalized)
            if isinstance(lemmas, list):
                return [token for token in lemmas if str(token).strip()]
        except Exception:
            pass
        return normalized.split()

    if language == "Persian":
        try:
            from hazm import Lemmatizer
            lemmatizer = Lemmatizer()
            return [lemmatizer.lemmatize(token) for token in normalized.split() if token.strip()]
        except Exception:
            return normalized.split()

    if language == "English":
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return [token.lemma_ for token in nlp(normalized) if token.lemma_.strip()]
        except Exception:
            return normalized.split()

    return normalized.split()


def is_reference_match(prediction: str, references: list[str], language: str) -> tuple[bool, str]:
    pred_norm = normalize_answer(prediction)
    pred_tokens = tokenize_for_match(prediction, language)

    for ref in references:
        ref_norm = normalize_answer(ref)
        ref_tokens = tokenize_for_match(ref, language)

        if pred_norm == ref_norm:
            return True, ref
        if pred_norm and (pred_norm in ref_norm or ref_norm in pred_norm):
            return True, ref
        if pred_tokens and ref_tokens:
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            if pred_set == ref_set or pred_set.issubset(ref_set) or ref_set.issubset(pred_set):
                return True, ref

    return False, ""


def Track_A_run_blend_evaluation(country: str, language: str) -> None:
    annotations_file = ANNOTATIONS_DIR / f"{country}_data.json"
    with annotations_file.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    predictions_df = pd.read_csv(get_predictions_file(country, language))

    detail_rows = []
    correct = 0

    for _, row in predictions_df.iterrows():
        qid = row["ID"]
        data = annotations.get(qid)
        if not data:
            continue

        references = []
        weighted_match = 0.0
        max_count = 1
        for annotation in data.get("annotations", []):
            max_count = max(max_count, annotation.get("count", 1))
            references.extend(annotation.get("answers", []))
            references.extend(annotation.get("en_answers", []))

        matched, matched_ref = is_reference_match(str(row["response"]), references, language)
        if matched:
            correct += 1
            for annotation in data.get("annotations", []):
                candidate_refs = annotation.get("answers", []) + annotation.get("en_answers", [])
                if matched_ref in candidate_refs:
                    weighted_match = annotation.get("count", 1) / max_count
                    break

        detail_rows.append(
            {
                "ID": qid,
                "country": country,
                "language": language,
                "prompt_id": PROMPT_ID,
                "prediction": row["response"],
                "matched": int(matched),
                "weight_score": weighted_match,
                "matched_reference": matched_ref,
            }
        )

    total = len(detail_rows)
    accuracy = correct / total if total else 0.0
    weighted_accuracy = (
        sum(row["weight_score"] for row in detail_rows) / total if total else 0.0
    )

    detail_df = pd.DataFrame(detail_rows)
    detail_file = get_question_results_file(country, language)
    detail_df.to_csv(detail_file, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(
        [
            {
                "model": MODEL_LABEL,
                "country": country,
                "language": language,
                "prompt_no": PROMPT_ID,
                "eval_method": "simple_exact",
                "score": accuracy,
            },
            {
                "model": MODEL_LABEL,
                "country": country,
                "language": language,
                "prompt_no": PROMPT_ID,
                "eval_method": "simple_weighted",
                "score": weighted_accuracy,
            },
        ]
    )
    if EVAL_RESULTS_FILE.exists():
        previous_df = pd.read_csv(EVAL_RESULTS_FILE)
        summary_df = pd.concat([previous_df, summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(
            subset=["model", "country", "language", "prompt_no", "eval_method"],
            keep="last",
        )
    summary_df.to_csv(EVAL_RESULTS_FILE, index=False, encoding="utf-8")

    log.info("Saved simple evaluation details to %s", detail_file)
    log.info("Saved simple evaluation summary to %s", EVAL_RESULTS_FILE)


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
    with get_predictions_file("TMP", LANGUAGE).open("w", encoding="utf-8", newline="") as handle:
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



def get_question_text(row: pd.Series, language: str) -> str:
    if language == "English":
        translated = row.get("Translation")
        if isinstance(translated, str) and translated.strip():
            return translated
    return row["Question"]


def Track_A_run(country: str, language: str):
    batch_size = 50
    #model_list = ["google/gemma-2b", "Qwen/Qwen3-0.6B"]
    #model_name = model_list[0]
    #tokenizer, model = load_model(model_name)
    pipe = load_gemma3()

    df = pd.read_csv(str(QUESTIONS_DIR / f"{country}_questions.csv"))

    system_prompt = (
        f"Requirements: consider you are from {country} and answer the given question. "
        "Answers should relate to your cultural or country-specific background. "
        "The answer should be short/concrete."
    )

    output_file = get_predictions_file(country, language)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "country", "prompt_id", "question", "prompt", "response"],
        )
        writer.writeheader()

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            log.info("Processing batch %s-%s", start, min(start + batch_size, len(df)))

            for _, row in batch.iterrows():
                question = get_question_text(row, language)

                #response = output_message_tok(model, tokenizer, messages)
                response = generate_gemma3_response(pipe, system_prompt, question)

                writer.writerow(
                    {
                        "ID": row["ID"],
                        "country": country,
                        "prompt_id": PROMPT_ID,
                        "question": question,
                        "prompt": system_prompt,
                        "response": response,
                    }
                )

                log.info("ID: %s", row["ID"])
                log.info("Response: %s", response)

            handle.flush()
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info("Saved predictions to %s", output_file)

def main():
    setup_logging()
    try:
        if EVAL_RESULTS_FILE.exists():
            EVAL_RESULTS_FILE.unlink()
        for country in COUNTRY_LIST:
            language = LANGUAGE if LANGUAGE else COUNTRY_LANG[country]
            log.info("Running benchmark for %s", country)
            Track_A_run(country, language)
            Track_A_run_blend_evaluation(country, language)
    except Exception:
        log.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()
