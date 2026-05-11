"""
Evaluate models through an OpenAI-compatible API endpoint.

This script supports:
1. Running a full model list from configs/model_list.json
2. Running selected models from command line
3. Checkpoint/resume by skipping already processed examples
4. Concurrent API requests

Example full run:
python eval/eval_model.py \
  --input_files \
    data/cat1/cat1_citation_retrieval.jsonl \
    data/cat2/cat2_citation_completeness.jsonl \
    data/cat3/cat3_citation_verification.jsonl \
    data/cat4/cat4-1_case_matching.jsonl \
    data/cat4/cat4-2_case_verification.jsonl \
  --output_dir outputs/model_outputs \
  --model_list_file configs/model_list.json \
  --model_type all \
  --max_workers 200

Example quick test:
python eval/eval_model.py \
  --input_files data/cat1/cat1_citation_retrieval.jsonl \
  --output_dir outputs/test_outputs \
  --models openai/gpt-4o-mini \
  --max_workers 10
"""

import argparse
import json
import os
import random
import threading
import time
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
# Prompts
# ============================================================

SYSTEM_PROMPT_CITATION = """You are a legal research expert.
When asked a question, provide the relevant case citations or legal authorities directly.
Be direct and concise. Only list the citations or cases asked for."""

SYSTEM_PROMPT_VERIFY = """You are a legal research expert.
When asked whether a citation or case reference is correct:
- If it IS correct: Answer "Yes".
- If it is NOT correct: Answer "No", then provide the correct citation or case.
Be direct and concise."""

CITATION_QA_STYLES = {"1", "2", "4-1"}

NO_TEMP_MODELS = {
    "openai/o4-mini",
    "openai/o3-mini",
    "openai/o3",
    "openai/o1",
}

NO_TOKEN_LIMIT_MODELS = {
    "openai/o4-mini",
    "openai/o3-mini",
    "openai/o3",
    "openai/o1",
    "openai/gpt-5-mini",
    "openai/gpt-5.1",
    "moonshotai/kimi-k2.5",
    "moonshotai/Kimi-K2.5",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
}


# ============================================================
# Helpers
# ============================================================

def get_system_prompt(qa_style: str) -> str:
    """Return the system prompt based on task type."""
    if str(qa_style) in CITATION_QA_STYLES:
        return SYSTEM_PROMPT_CITATION
    return SYSTEM_PROMPT_VERIFY


def sanitize_model_name(model_name: str) -> str:
    """Make model name safe for output filenames."""
    return model_name.replace("/", "_").replace(":", "_")


def make_output_key(record: Dict[str, Any], model_name: str) -> tuple:
    """
    Unique key for checkpointing.

    Some records contain legal_angle, so we include it when available.
    """
    q_id = str(record.get("id", "unknown"))
    qa_style = str(record.get("qa_style", ""))
    legal_angle = str(record.get("legal_angle", ""))

    if legal_angle:
        return (q_id, qa_style, legal_angle, model_name)
    return (q_id, qa_style, model_name)


def load_processed(output_file: Path) -> set:
    """Load processed keys from an existing output file."""
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                q_id = str(d.get("id", "unknown"))
                qa_style = str(d.get("qa_style", ""))
                legal_angle = str(d.get("legal_angle", ""))
                model = d.get("model", "")

                if legal_angle:
                    key = (q_id, qa_style, legal_angle, model)
                else:
                    key = (q_id, qa_style, model)

                processed.add(key)
            except Exception:
                pass

    return processed


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    return records


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

    model_type:
    - all
    - open
    - closed
    """
    with open(model_list_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = []

    for item in config.get("models", []):
        if model_type == "all" or item.get("type") == model_type:
            models.append(item["model"])

    return models


def get_generation_params(model_name: str) -> Dict[str, Any]:
    """
    Return generation parameters for different API model families.

    Some reasoning models do not support temperature or explicit token limits
    through OpenAI-compatible endpoints, so we handle them separately.
    """
    if model_name in NO_TOKEN_LIMIT_MODELS:
        gen_params = {}
    elif model_name.startswith("openai/"):
        gen_params = {"max_completion_tokens": 1000}
    else:
        gen_params = {"max_tokens": 1000}

    if model_name not in NO_TEMP_MODELS:
        gen_params["temperature"] = 0

    return gen_params


# ============================================================
# API call
# ============================================================

def get_model_response(
    model_name: str,
    question: str,
    qa_style: str,
    max_retries: int = 3,
) -> str:
    """Call the OpenAI-compatible API with retry logic."""
    system_prompt = get_system_prompt(qa_style)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    gen_params = get_generation_params(model_name)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **gen_params,
            )

            content = resp.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            if attempt < max_retries - 1:
                # Light backoff. Add jitter to reduce synchronized retry bursts.
                wait = min(30, (2 ** attempt) * 5) + random.random()
                time.sleep(wait)
            else:
                raise e

    return ""


def process_single(record: Dict[str, Any], model_name: str, empty_retries: int = 2) -> Dict[str, Any]:
    """Process one question, retrying if the API returns an empty output."""
    qa_style = str(record.get("qa_style", ""))
    answer = ""

    for attempt in range(1 + empty_retries):
        answer = get_model_response(
            model_name=model_name,
            question=record["question"],
            qa_style=qa_style,
        )

        if answer:
            break

        if attempt < empty_retries:
            time.sleep(1)

    return {
        "id": record.get("id"),
        "qa_style": record.get("qa_style"),
        "legal_angle": record.get("legal_angle"),
        "question": record.get("question"),
        "ground_truth": record.get("ground_truth"),
        "model": model_name,
        "output": answer,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model_name: str,
    all_questions: List[Dict[str, Any]],
    output_dir: Path,
    max_workers: int,
) -> None:
    """Evaluate one model with concurrent requests and checkpointing."""
    output_file = output_dir / f"{sanitize_model_name(model_name)}.jsonl"

    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print(f"Output: {output_file}")
    print("=" * 60)

    processed = load_processed(output_file)
    print(f"Already processed: {len(processed)} entries")

    todo = [
        record
        for record in all_questions
        if make_output_key(record, model_name) not in processed
    ]

    print(f"To process: {len(todo)} questions")

    if not todo:
        print("All done for this model.")
        return

    results_count = 0
    error_count = 0
    write_lock = threading.Lock()

    with open(output_file, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single, record, model_name): record
                for record in todo
            }

            pbar = tqdm(total=len(todo), desc=model_name)

            for future in as_completed(futures):
                record = futures[future]

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
                        print(f"\nError on id={record.get('id', '?')}: {e}")
                    elif error_count == 6:
                        print("\nSuppressing further error messages...")

                pbar.update(1)

            pbar.close()

    print(f"Completed {model_name}: {results_count} new, {error_count} errors")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Input benchmark JSONL files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model outputs.",
    )

    parser.add_argument(
        "--model_list_file",
        type=str,
        default=None,
        help="Path to configs/model_list.json.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["all", "open", "closed"],
        help="Which models to run from the config file.",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model names. Overrides --model_list_file.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=200,
        help="Number of concurrent API workers.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print(f"Concurrency: {args.max_workers} workers")
    print("=" * 60)

    input_files = [Path(path) for path in args.input_files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = args.models
    elif args.model_list_file:
        models = load_models_from_config(
            model_list_file=args.model_list_file,
            model_type=args.model_type,
        )
    else:
        raise ValueError("Please provide either --models or --model_list_file.")

    all_questions = []

    for input_file in input_files:
        if not input_file.exists():
            print(f"Warning: {input_file} not found")
            continue

        records = load_jsonl(input_file)
        print(f"Loaded {len(records):>5} from {input_file.name}")
        all_questions.extend(records)

    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Models: {len(models)}")

    for model_name in models:
        print(f"  - {model_name}")

    for model_name in models:
        evaluate_model(
            model_name=model_name,
            all_questions=all_questions,
            output_dir=output_dir,
            max_workers=args.max_workers,
        )

    print("\n" + "=" * 60)
    print("All Evaluations Completed!")
    print("=" * 60)

    for model_name in models:
        output_file = output_dir / f"{sanitize_model_name(model_name)}.jsonl"

        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)

            print(f"  {output_file.name}: {line_count} entries")


if __name__ == "__main__":
    main()