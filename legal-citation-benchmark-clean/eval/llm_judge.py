"""
LLM judge evaluation for LegalCiteBench.

This script scores model outputs using an OpenAI-compatible API judge.

Scoring:
  Cat1/2  : F1-based precision/recall on citation lists, scaled to 0--5
  Cat3    : citation error detection and correction accuracy
  Cat4-1  : case identity match
  Cat4-2  : case verification and correction accuracy

Features:
  - Checkpoint/resume support
  - Concurrent requests
  - Config-based model selection
  - Per-model and combined summary reports

Example:
python eval/llm_judge.py \
  --input_dir outputs/model_outputs \
  --output_dir outputs/judge_outputs \
  --model_list_file configs/model_list.json \
  --model_type all \
  --judge_model openai/gpt-4o-mini \
  --max_workers 10

Example with explicit files:
python eval/llm_judge.py \
  --input_files outputs/model_outputs/openai_gpt-4o-mini.jsonl \
  --output_dir outputs/judge_outputs \
  --judge_model openai/gpt-4o-mini
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# API client
# ============================================================

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


# ============================================================
# Judge prompts
# ============================================================

JUDGE_PROMPT_CITATION = """You are evaluating legal citation outputs. Compare the model output against ground truth citations using SUBSTRING matching (case-insensitive).

MODEL OUTPUT:
{output}

GROUND TRUTH CITATIONS:
{ground_truth}

Instructions:
- Extract all citations from the model output.
- A citation in the output is CORRECT if it CONTAINS or IS CONTAINED BY a ground truth citation, using case-insensitive substring matching.
  - Example: "466 U.S. 668" matches "Strickland v. Washington, 466 U.S. 668 (1984)".
  - Example: "231 Ariz. 150" does NOT match "231 Ariz. 145".
- A citation is HALLUCINATED if it does not match any ground truth citation.
- Ignore formatting differences such as markdown or bullet points.
- If the model gives no citation, total_output_count should be 0.
- Do not give extra explanation outside JSON.

Reply JSON only:
{{
  "correct_citations": ["<matched output citations>"],
  "hallucinated_citations": ["<unmatched output citations>"],
  "missed_citations": ["<ground truth citations not found in output>"],
  "total_output_count": <int>,
  "total_gt_count": <int>
}}"""

JUDGE_PROMPT_3_TRUE = """The paragraph has NO citation error. Did the model correctly say so?

OUTPUT:
{output}

Score:
- 5 = correctly confirmed no error
- 2 = wrongly claimed there is an error
- 0 = irrelevant or no response

Reply JSON only:
{{
  "score": <0|2|5>,
  "model_judgment": "<positive/negative/unclear>",
  "reasoning": "<1 sentence>"
}}"""

JUDGE_PROMPT_3_FAKE = """The paragraph contains an INCORRECT citation. Did the model detect and correctly fix it?

OUTPUT:
{output}

CORRECT ANSWER:
{ground_truth}

Score:
- 5 = detected the error AND provided the correct citation
- 2 = detected the error BUT correction is wrong or missing
- 1 = did not detect the error
- 0 = irrelevant or no response

Reply JSON only:
{{
  "score": <0|1|2|5>,
  "detected_error": <true|false>,
  "correction_correct": <true|false>,
  "reasoning": "<1 sentence>"
}}"""

JUDGE_PROMPT_4_1 = """Did the model identify the correct legal case?

OUTPUT:
{output}

GROUND TRUTH CASE:
{ground_truth}

Score:
- 5 = exact match, case name and/or citation matches clearly
- 4 = very minor difference, same case clearly identified
- 3 = related but not exactly correct
- 2 = wrong case entirely
- 0 = irrelevant or no response

Reply JSON only:
{{
  "score": <0|2|3|4|5>,
  "case_match": <true|false>,
  "match_details": "<what matched>",
  "reasoning": "<1 sentence>"
}}"""

JUDGE_PROMPT_4_2_TRUE = """The citation IS correct. Did the model confirm it?

OUTPUT:
{output}

Score:
- 5 = correctly confirmed
- 2 = wrongly rejected
- 0 = irrelevant or no response

Reply JSON only:
{{
  "score": <0|2|5>,
  "model_judgment": "<positive/negative/unclear>",
  "reasoning": "<1 sentence>"
}}"""

JUDGE_PROMPT_4_2_FAKE = """The citation is INCORRECT. Did the model reject it and provide the correct case?

OUTPUT:
{output}

CORRECT CASE:
{ground_truth}

Score:
- 5 = rejected the wrong citation AND identified the correct case
- 2 = rejected the wrong citation BUT did not identify the correct case
- 1 = accepted the wrong citation
- 0 = irrelevant or no response

Reply JSON only:
{{
  "score": <0|1|2|5>,
  "detected_error": <true|false>,
  "case_match": <true|false>,
  "reasoning": "<1 sentence>"
}}"""


# ============================================================
# Helpers
# ============================================================

def sanitize_model_name(model_name: str) -> str:
    """Make model name safe for filenames."""
    return model_name.replace("/", "_").replace(":", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    return records


def write_json(path: Path, obj: Any) -> None:
    """Write JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_models_from_config(model_list_file: str, model_type: str = "all") -> List[str]:
    """
    Load model names from configs/model_list.json.

    Expected format:
    {
      "models": [
        {"model": "openai/gpt-4o-mini", "type": "closed"},
        {"model": "Qwen/Qwen3-14B", "type": "open"}
      ]
    }
    """
    with open(model_list_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = []

    for item in config.get("models", []):
        if model_type == "all" or item.get("type") == model_type:
            models.append(item["model"])

    return models


def format_ground_truth(gt: Any) -> str:
    """Format ground truth for judge prompt."""
    if isinstance(gt, list):
        if gt and isinstance(gt[0], list):
            lines = []
            for item in gt:
                if isinstance(item, list) and len(item) == 2:
                    lines.append(f"  {item[0]}: {item[1]}")
                else:
                    lines.append(f"  {item}")
            return "\n".join(lines)
        return "\n".join(f"  - {c}" for c in gt)

    if isinstance(gt, dict):
        return "\n".join(f"  {k}: {v}" for k, v in gt.items())

    return str(gt)


def get_judge_prompt(record: Dict[str, Any]) -> Optional[str]:
    """Select the correct judge prompt based on qa_style."""
    qa_style = str(record.get("qa_style", ""))
    output = record.get("output", "") or "[NO RESPONSE]"
    gt = record.get("ground_truth")
    gt_str = format_ground_truth(gt)

    if qa_style in ("1", "2"):
        return JUDGE_PROMPT_CITATION.format(
            output=output,
            ground_truth=gt_str,
        )

    if qa_style == "3-true":
        return JUDGE_PROMPT_3_TRUE.format(output=output)

    if qa_style == "3-fake":
        return JUDGE_PROMPT_3_FAKE.format(
            output=output,
            ground_truth=gt_str,
        )

    if qa_style == "4-1":
        return JUDGE_PROMPT_4_1.format(
            output=output,
            ground_truth=gt_str,
        )

    if qa_style == "4-2-true":
        return JUDGE_PROMPT_4_2_TRUE.format(output=output)

    if qa_style == "4-2-fake":
        return JUDGE_PROMPT_4_2_FAKE.format(
            output=output,
            ground_truth=gt_str,
        )

    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly parse JSON from judge output."""
    if not text:
        return None

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def compute_f1_score(judge_result: Dict[str, Any]) -> float:
    """
    Compute citation F1 for Cat1/Cat2 and scale to 0--5.
    """
    correct = len(judge_result.get("correct_citations", []) or [])
    hallucinated = len(judge_result.get("hallucinated_citations", []) or [])
    missed = len(judge_result.get("missed_citations", []) or [])

    total_output = judge_result.get("total_output_count", 0)
    total_gt = judge_result.get("total_gt_count", 0)

    if not isinstance(total_output, int) or total_output < 0:
        total_output = correct + hallucinated

    if not isinstance(total_gt, int) or total_gt <= 0:
        total_gt = correct + missed

    precision = correct / total_output if total_output > 0 else 0.0
    recall = correct / total_gt if total_gt > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 5, 4)


def make_judge_key(record: Dict[str, Any]) -> tuple:
    """
    Unique key for judge checkpointing.

    Include legal_angle if available, because some categories may share id and qa_style.
    """
    q_id = str(record.get("id", ""))
    qa_style = str(record.get("qa_style", ""))
    legal_angle = str(record.get("legal_angle", ""))
    model = str(record.get("model", ""))

    if legal_angle:
        return (q_id, qa_style, legal_angle, model)

    return (q_id, qa_style, model)


def load_processed_keys(output_file: Path) -> set:
    """Load already judged keys."""
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(make_judge_key(record))
            except Exception:
                pass

    return processed


# ============================================================
# Judge call
# ============================================================

def call_judge(
    prompt: str,
    judge_model: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Call judge model with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise legal citation evaluator. "
                            "Always respond with valid JSON only. "
                            "Ignore formatting differences and evaluate content only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=3000,
                temperature=0,
                top_p=1.0,
            )

            text = resp.choices[0].message.content or ""
            parsed = extract_json(text)

            if parsed is None:
                raise json.JSONDecodeError("Could not parse JSON", text, 0)

            return parsed

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2 + random.random())
            else:
                return {
                    "score": -1,
                    "error": "JSON parse failed",
                }

        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate" in msg.lower()

            if attempt < max_retries - 1:
                wait = min(10, 2 ** attempt) + random.random()
                if is_rate_limit:
                    wait = min(30, (2 ** attempt) * 2) + random.random()
                time.sleep(wait)
            else:
                return {
                    "score": -1,
                    "error": str(e),
                }

    return {
        "score": -1,
        "error": "unknown judge failure",
    }


def process_single(
    record: Dict[str, Any],
    judge_model: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Judge a single model output."""
    qa_style = str(record.get("qa_style", ""))
    prompt = get_judge_prompt(record)

    if not prompt:
        return {
            "id": record.get("id"),
            "qa_style": qa_style,
            "legal_angle": record.get("legal_angle"),
            "model": record.get("model"),
            "responded": False,
            "judge_score": -1,
            "judge_model": judge_model,
            "judge_detail": {"error": "unknown qa_style"},
        }

    judge_result = call_judge(
        prompt=prompt,
        judge_model=judge_model,
        max_retries=max_retries,
    )

    if qa_style in ("1", "2"):
        if "error" not in judge_result:
            score = compute_f1_score(judge_result)
            judge_result["score"] = score
        else:
            judge_result["score"] = -1

    return {
        "id": record.get("id"),
        "qa_style": qa_style,
        "legal_angle": record.get("legal_angle"),
        "model": record.get("model"),
        "responded": bool((record.get("output") or "").strip()),
        "judge_score": judge_result.get("score", -1),
        "judge_model": judge_model,
        "judge_detail": judge_result,
    }


# ============================================================
# Judge one model output file
# ============================================================

def judge_file(
    records: List[Dict[str, Any]],
    model_name: str,
    judge_output_file: Path,
    judge_model: str,
    max_workers: int,
    max_retries: int,
) -> None:
    """Judge one model output file."""
    print("\n" + "=" * 60)
    print(f"Judging model outputs from: {model_name}")
    print(f"Judge model: {judge_model}")
    print(f"Output: {judge_output_file}")
    print("=" * 60)

    processed = load_processed_keys(judge_output_file)

    todo = [
        record
        for record in records
        if make_judge_key(record) not in processed
    ]

    print(f"Already judged: {len(processed)}")
    print(f"To judge: {len(todo)}")

    if not todo:
        print("All done for this file.")
        return

    results_count = 0
    error_count = 0
    write_lock = threading.Lock()

    with open(judge_output_file, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single, record, judge_model, max_retries): record
                for record in todo
            }

            pbar = tqdm(total=len(todo), desc=model_name)

            for future in as_completed(futures):
                try:
                    result = future.result()

                    with write_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results_count += 1

                        if results_count % 100 == 0:
                            out.flush()

                except Exception as e:
                    error_count += 1

                    if error_count <= 5:
                        print(f"\nError: {e}")
                    elif error_count == 6:
                        print("\nSuppressing further error messages...")

                pbar.update(1)

            pbar.close()

    print(f"Completed: {results_count} judged, {error_count} errors")


# ============================================================
# Summary
# ============================================================

def compute_summary(judge_file_path: Path) -> Dict[str, Any]:
    """Compute per-qa-style summary for a judge output file."""
    records = load_jsonl(judge_file_path)

    by_style = defaultdict(list)

    for record in records:
        by_style[str(record.get("qa_style", ""))].append(record)

    summary = {}

    for qa_style, recs in by_style.items():
        n = len(recs)

        scores = [
            float(record["judge_score"])
            for record in recs
            if float(record.get("judge_score", -1)) >= 0
        ]

        responded = [
            record
            for record in recs
            if record.get("responded", False)
        ]

        responded_n = len(responded)

        hallucinated = sum(
            1
            for record in responded
            if 0 <= float(record.get("judge_score", -1)) <= 2
        )

        score_dist = defaultdict(int)
        for score in scores:
            score_dist[str(score)] += 1

        summary[qa_style] = {
            "n": n,
            "valid_scores": len(scores),
            "response_rate": responded_n / n if n else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "score_dist": dict(score_dist),
            "hallucination_rate": hallucinated / responded_n if responded_n else 0.0,
        }

    return summary


def compute_overall_summary(all_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten summaries into rows for easy inspection."""
    rows = []

    for model_name, summary in all_summaries.items():
        for qa_style, metrics in summary.items():
            rows.append(
                {
                    "model": model_name,
                    "qa_style": qa_style,
                    "n": metrics["n"],
                    "valid_scores": metrics["valid_scores"],
                    "response_rate": metrics["response_rate"],
                    "avg_score": metrics["avg_score"],
                    "hallucination_rate": metrics["hallucination_rate"],
                }
            )

    return rows


def print_comparison_table(all_summaries: Dict[str, Dict[str, Any]]) -> None:
    """Print a compact comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    all_styles = sorted({qa_style for summary in all_summaries.values() for qa_style in summary})
    model_names = sorted(all_summaries.keys())

    for qa_style in all_styles:
        print(f"\n--- qa_style: {qa_style} ---")
        print(f"{'Model':<45} {'RespRate':>8} {'AvgScore':>8} {'Halluc%':>8}")
        print("-" * 75)

        for model_name in model_names:
            metrics = all_summaries[model_name].get(qa_style)

            if metrics:
                print(
                    f"{model_name:<45} "
                    f"{metrics['response_rate']:>8.1%} "
                    f"{metrics['avg_score']:>8.2f} "
                    f"{metrics['hallucination_rate']:>8.1%}"
                )


# ============================================================
# File selection
# ============================================================

def get_input_files_from_models(input_dir: Path, models: List[str]) -> List[Path]:
    """Map model names to expected sanitized output filenames."""
    files = []

    for model in models:
        path = input_dir / f"{sanitize_model_name(model)}.jsonl"

        if path.exists():
            files.append(path)
        else:
            print(f"Warning: expected output file not found: {path}")

    return files


def infer_model_name_from_records(records: List[Dict[str, Any]], fallback: str) -> str:
    """Infer model name from records."""
    if records:
        return str(records[0].get("model", fallback))
    return fallback


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing model output JSONL files.",
    )

    parser.add_argument(
        "--input_files",
        nargs="*",
        default=None,
        help="Explicit model output JSONL files to judge.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save judge outputs and reports.",
    )

    parser.add_argument(
        "--model_list_file",
        type=str,
        default=None,
        help="Path to configs/model_list.json. Used with --input_dir.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["all", "open", "closed"],
        help="Which models to select from model_list_file.",
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Judge model name for the OpenAI-compatible API.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Number of concurrent judge requests.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for judge API calls.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_files:
        output_files = [Path(path) for path in args.input_files]
    else:
        if not args.input_dir:
            raise ValueError("Please provide either --input_files or --input_dir.")

        input_dir = Path(args.input_dir)

        if args.model_list_file:
            models = load_models_from_config(
                model_list_file=args.model_list_file,
                model_type=args.model_type,
            )
            output_files = get_input_files_from_models(input_dir, models)
        else:
            output_files = sorted(input_dir.glob("*.jsonl"))

    output_files = [path for path in output_files if path.exists()]

    if not output_files:
        print("No input model output files found.")
        return

    print("=" * 60)
    print("LLM Judge Evaluation")
    print(f"Judge model: {args.judge_model}")
    print(f"Concurrency: {args.max_workers}")
    print(f"Files to judge: {len(output_files)}")
    print("=" * 60)

    all_summaries = {}

    for input_file in output_files:
        records = load_jsonl(input_file)
        print(f"\nLoaded {len(records)} records from {input_file.name}")

        model_name = infer_model_name_from_records(records, input_file.stem)
        judge_output_file = output_dir / f"judge_{input_file.name}"

        judge_file(
            records=records,
            model_name=model_name,
            judge_output_file=judge_output_file,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )

        summary = compute_summary(judge_output_file)
        all_summaries[model_name] = summary

        report_path = output_dir / f"report_{input_file.stem}.json"
        write_json(report_path, {model_name: summary})
        print(f"Per-model report saved: {report_path}")

    combined_report_path = output_dir / "combined_judge_report.json"
    write_json(combined_report_path, all_summaries)
    print(f"\nCombined report saved: {combined_report_path}")

    rows_path = output_dir / "combined_judge_rows.json"
    write_json(rows_path, compute_overall_summary(all_summaries))
    print(f"Flattened rows saved: {rows_path}")

    print_comparison_table(all_summaries)


if __name__ == "__main__":
    main()