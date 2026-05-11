"""
Generate Category 4-1 case matching questions.

Input:
  Structured root-case summaries.

Output:
  Cat4-1 QA pairs where the question is an anonymized lawyer-style fact pattern,
  and the ground truth is the original source case.

Task:
  Given an anonymized client scenario with no explicit case name, court, year, or
  citation, the model must identify the underlying precedent case.

Example:
python data/cat4/cat4_1_case_matching.py \
  --input_files \
    outputs/case_summaries/openai_gpt-4o-mini_case_summaries.jsonl \
    outputs/case_summaries/google_gemini-2.5-flash_case_summaries.jsonl \
    outputs/case_summaries/anthropic_claude-3.5-haiku_case_summaries.jsonl \
  --models \
    openai/gpt-4o-mini \
    google/gemini-2.5-flash \
    anthropic/claude-3.5-haiku \
  --output_dir data/cat4/cat4-1
"""

import argparse
import json
import os
import time
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
# Defaults and prompts
# ============================================================

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-3.5-haiku",
]

SYSTEM_PROMPT = """You are a legal writing expert. Rewrite case facts from a practicing lawyer's perspective."""


# ============================================================
# IO helpers
# ============================================================

def sanitize_model_name(model_name: str) -> str:
    """Convert model name into filename-safe string."""
    return model_name.replace("/", "_").replace(":", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Append JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def normalize_source_case_id(record: Dict[str, Any]) -> str:
    """Convert root_case_xxx to xxx when possible."""
    return str(record.get("id", "")).replace("root_case_", "")


def make_record_key(record: Dict[str, Any]) -> str:
    """
    Unique checkpoint key.

    For Cat4-1, one output is generated per root case per summary file.
    """
    return str(record.get("id", ""))


def load_processed_ids(output_file: Path) -> set:
    """Load already processed source case ids."""
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(str(data.get("id", "")))
            except Exception:
                pass

    return processed


# ============================================================
# Prompt construction
# ============================================================

def build_anonymize_prompt(record: Dict[str, Any]) -> str:
    """Build prompt for anonymizing a case summary into a lawyer scenario."""
    case_summary = record["case_summary"]
    jurisdiction = record.get("jurisdiction", "the relevant jurisdiction")

    user_content = f"""Rewrite this case summary as a lawyer describing their client's situation to a colleague.

ORIGINAL CASE:
{case_summary}

YOUR TASK:
Rewrite as: "In {jurisdiction}, I represent a client who [situation]..."

Requirements:
1. START with "In {jurisdiction},"
2. Remove ALL specific identifiers:
   - Party names -> "my client", "the opposing party", "a property owner", "a defendant", etc.
   - County names -> "a county in {jurisdiction}", "the county"
   - Organization names -> "the district", "the agency", "the authority", "the company"
   - Case names, court names, citation strings, docket numbers, and years must not appear
3. Preserve the legally distinctive facts:
   - Legal issue
   - Procedural posture
   - Key factual elements
   - Governance, funding, operations, delegation, statutory or doctrinal context when relevant
4. Write from a lawyer's perspective using "I", "my client", "we", or "our"
5. End with exactly this request:
   "What precedent cases address similar issues? List the most relevant cases."
6. Keep it 100-200 words

Example format:
"In Pennsylvania, I represent property owners who are challenging fees imposed by a county conservation district. The district filed preliminary objections claiming it is a Commonwealth agency and the case should be in Commonwealth Court. However, the district operates only within one county, its board is appointed by county commissioners, and it receives mixed state and local funding. We believe it is a local agency and the county court has jurisdiction. What precedent cases address similar issues? List the most relevant cases."

CRITICAL:
- Anonymize names but keep the legal substance identical.
- Do not mention the original case name.
- Do not include the original citation.
- Do not include the court name or decision year.
- Output only the rewritten fact pattern as a question, no other text."""

    return user_content


# ============================================================
# Generation
# ============================================================

def get_max_tokens(model_name: str) -> int:
    """Set max tokens by model family."""
    lower_name = model_name.lower()

    if "gemini" in lower_name:
        return 1200

    if "claude" in lower_name:
        return 1000

    return 800


def generate_anonymized_question(
    record: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_retries: int,
    timeout: float,
) -> str:
    """Generate anonymized lawyer-style fact pattern."""
    user_prompt = build_anonymize_prompt(record)
    max_tokens = get_max_tokens(model_name)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            print(
                f"\nAttempt {attempt + 1}/{max_retries} failed "
                f"for case {record.get('id', 'unknown')}: {str(e)[:200]}"
            )

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                return ""

    return ""


def create_qa_pair(
    record: Dict[str, Any],
    anonymized_question: str,
    model_name: str,
) -> Dict[str, Any]:
    """Create one Cat4-1 QA record."""
    source_case_id = normalize_source_case_id(record)

    return {
        "id": source_case_id,
        "qa_style": "4-1",
        "question": anonymized_question,
        "ground_truth": {
            "case_name": record.get("case_name", ""),
            "citation": record.get("case_cite", ""),
            "year": str(record.get("decision_date", ""))[:4],
            "court": record.get("court", ""),
        },
        "jurisdiction": record.get("jurisdiction", "the relevant jurisdiction"),
        "model": model_name.split("/")[-1],
    }


# ============================================================
# Processing
# ============================================================

def process_input_file(
    input_file: Path,
    model_name: str,
    output_dir: Path,
    start: int,
    end: Optional[int],
    temperature: float,
    max_retries: int,
    timeout: float,
) -> None:
    """Process one case-summary file."""
    print("\n" + "=" * 60)
    print(f"Processing: {input_file}")
    print(f"Using model: {model_name}")
    print("=" * 60)

    all_records = load_jsonl(input_file)

    end_idx = end if end is not None else len(all_records)
    selected_records = all_records[start:end_idx]

    print(f"Total records in file: {len(all_records)}")
    print(f"Processing range: [{start}:{end_idx}]")
    print(f"Selected records: {len(selected_records)}")

    input_stem = input_file.name.replace("_case_summaries.jsonl", "")
    output_file = output_dir / f"cat4-1_{input_stem}.jsonl"

    processed_ids = load_processed_ids(output_file)

    to_process = [
        record
        for record in selected_records
        if normalize_source_case_id(record) not in processed_ids
    ]

    print(f"Already processed: {len(selected_records) - len(to_process)}")
    print(f"To process: {len(to_process)}")
    print(f"Output file: {output_file}")

    if not to_process:
        print("All done for this input file.")
        return

    total_generated = 0
    buffer = []

    for record in tqdm(to_process, desc=model_name.split("/")[-1]):
        anonymized_question = generate_anonymized_question(
            record=record,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )

        if not anonymized_question:
            print(f"Empty generation for {record.get('id', 'unknown')}")
            continue

        qa = create_qa_pair(
            record=record,
            anonymized_question=anonymized_question,
            model_name=model_name,
        )

        buffer.append(qa)
        total_generated += 1

        if len(buffer) >= 100:
            append_jsonl(output_file, buffer)
            buffer = []

    if buffer:
        append_jsonl(output_file, buffer)

    print(f"Total generated: {total_generated}")
    print(f"Saved to: {output_file}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cat4-1 case matching QA pairs."
    )

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Case-summary JSONL files.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Generator models. Must align with input_files order. "
            "If one model is provided for multiple files, it is reused."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save Cat4-1 output files.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index, inclusive.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index, exclusive.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum API retries per case.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="API timeout in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_files = [Path(path) for path in args.input_files]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.models) == 1 and len(input_files) > 1:
        models = args.models * len(input_files)
    elif len(args.models) == len(input_files):
        models = args.models
    else:
        raise ValueError(
            "Number of --models must be either 1 or equal to number of --input_files."
        )

    print("=" * 60)
    print("Category 4-1: Generate Case Matching Dataset")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    for input_file, model_name in zip(input_files, models):
        process_input_file(
            input_file=input_file,
            model_name=model_name,
            output_dir=output_dir,
            start=args.start,
            end=args.end,
            temperature=args.temperature,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )

    print("\n" + "=" * 60)
    print("All completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()