import argparse
import csv
import json
import re
from pathlib import Path


def parse_post_eval_response(text: str) -> dict:
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced_blocks = re.findall(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    candidates = [block.strip() for block in fenced_blocks if block.strip()]

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        if raw[idx] != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                candidates.append(raw[idx:idx + end])
            idx += max(end, 1)
        except Exception:
            idx += 1

    parsed_objects = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                parsed_objects.append(obj)
        except Exception:
            pass

    for obj in parsed_objects:
        if "predicted_answer" in obj:
            return obj

    for obj in parsed_objects:
        answer_choice = str(obj.get("answer_choice", "")).strip().upper()
        if answer_choice in {"A", "B", "C", "D"}:
            return {
                "predicted_answer": answer_choice,
                "confidence": "",
                "reasoning_summary": "",
                "error_type": "",
                "likely_failure_source": "",
                "mentions_country_specific_cue": "",
            }

    return {}


def normalize_answer(value: str) -> str:
    answer = str(value or "").strip().upper()
    return answer if answer in {"A", "B", "C", "D"} else ""


def coerce_country_specific_cue(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return ""


def repair_rows(rows: list[dict]) -> list[dict]:
    repaired = []
    for row in rows:
        row = dict(row)
        parsed = parse_post_eval_response(row.get("post_eval_response", ""))

        row["post_eval_predicted_answer"] = normalize_answer(
            parsed.get("predicted_answer", "")
        )
        row["post_eval_confidence"] = str(parsed.get("confidence", "")).strip().lower()
        row["post_eval_reasoning_summary"] = str(
            parsed.get("reasoning_summary", "")
        ).strip()
        row["post_eval_error_type"] = str(parsed.get("error_type", "")).strip()
        row["post_eval_likely_failure_source"] = str(
            parsed.get("likely_failure_source", "")
        ).strip()
        row["post_eval_mentions_country_specific_cue"] = coerce_country_specific_cue(
            parsed.get("mentions_country_specific_cue", "")
        )

        answer_idx = str(row.get("answer_idx", "")).strip().upper()
        predicted = row["post_eval_predicted_answer"]
        is_correct = bool(predicted and answer_idx and predicted == answer_idx)
        row["post_eval_is_correct"] = is_correct
        row["post_eval_is_correct_int"] = 1 if is_correct else 0

        repaired.append(row)
    return repaired


def repair_file(input_path: Path, output_path: Path | None = None) -> Path:
    with input_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    repaired_rows = repair_rows(rows)
    fieldnames = list(repaired_rows[0].keys()) if repaired_rows else []

    if output_path is None:
        output_path = input_path

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repaired_rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reparse existing post-evaluation CSV files without rerunning inference."
    )
    parser.add_argument("inputs", nargs="+", help="CSV files to repair")
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for repaired files, for example '_reparsed'.",
    )
    args = parser.parse_args()

    for raw_input in args.inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if args.suffix:
            output_path = input_path.with_name(
                f"{input_path.stem}{args.suffix}{input_path.suffix}"
            )
        else:
            output_path = input_path
        repaired = repair_file(input_path, output_path)
        print(f"Repaired {input_path} -> {repaired}")


if __name__ == "__main__":
    main()
