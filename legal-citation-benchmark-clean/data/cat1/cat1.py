"""
Generate Category 1 citation retrieval questions.

Input:
  Case summaries generated from root cases.

Output:
  Cat1 QA pairs where each question asks a lawyer-style citation retrieval
  question, and the ground truth is the full citation list extracted from the
  source root case.

Each root case summary produces 9 questions:
  - 3 legal research questions
  - 3 argument / brief-writing questions
  - 3 compliance / advisory questions

The generated questions are written from a practicing lawyer's perspective and
must start with the jurisdiction phrase, e.g., "In Pennsylvania,".

Example:
python data/cat1/cat1.py \
  --input_files \
    outputs/case_summaries/openai_gpt-4o-mini_case_summaries.jsonl \
    outputs/case_summaries/google_gemini-2.5-flash_case_summaries.jsonl \
    outputs/case_summaries/anthropic_claude-3.5-haiku_case_summaries.jsonl \
  --models \
    openai/gpt-4o-mini \
    google/gemini-2.5-flash \
    anthropic/claude-3.5-haiku \
  --output_dir data/cat1 \
  --start 0
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
# Defaults
# ============================================================

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-3.5-haiku",
]

SYSTEM_PROMPT = """You are a legal research expert. Generate realistic questions that practicing lawyers actually ask. Use conversational language, include specific client scenarios, and write from the lawyer's perspective."""


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
    """Append records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_case_ids(output_file: Path) -> set:
    """Load source case ids already written to an output file."""
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

def build_prompt(record: Dict[str, Any]) -> str:
    """Build prompt for generating Cat1 lawyer-style questions."""

    case_summary = record["case_summary"]
    jurisdiction = record.get("jurisdiction", "the relevant jurisdiction")

    user_content = f"""Based on this {jurisdiction} case summary, generate 9 realistic legal questions that a practicing lawyer would actually ask.

CRITICAL: Each question must be CONCISE, 60-100 words maximum.

CRITICAL REQUIREMENT: ALL questions MUST start with "In {jurisdiction}," to clearly indicate the jurisdiction.

CASE SUMMARY:
{case_summary}

YOUR TASK:
Generate questions in THREE categories. Each question should:
- START with "In {jurisdiction}," exactly
- Be written from a lawyer's perspective using "I", "my client", "we", or "our"
- Include a specific hypothetical client situation
- Use conversational legal language, like a real lawyer asking for research help
- Be distinct from the other questions
- End by asking for cases, precedents, or legal authorities

Category 1: Legal Research Questions
Generate 3 questions in this style:
"In {jurisdiction}, I represent [specific client] who [specific situation]. [Key facts]. [Legal question]. What legal precedents and cases should I cite to answer this question?"

Category 2: Legal Argument / Brief-Writing Questions
Generate 3 questions in this style:
"In {jurisdiction}, I'm drafting a [brief/motion] arguing that [argument]. [Context]. What cases should I cite to support this argument?"

Category 3: Compliance / Advisory Questions
Generate 3 questions in this style:
"In {jurisdiction}, I'm advising [organization] on [issue]. [Situation]. What legal authorities should I reference in this analysis?"

CRITICAL REQUIREMENTS:
- EVERY question MUST start with "In {jurisdiction},"
- Use first person lawyer perspective
- Include specific details and scenarios
- Make it sound natural and practice-oriented
- Each question should be 60-100 words
- All questions must be in English
- Do not include answers
- Do not include citations in the questions unless they are naturally part of the scenario

OUTPUT JSON only:
{{
  "legal_questions": ["Q1", "Q2", "Q3"],
  "argument_questions": ["Q1", "Q2", "Q3"],
  "compliance_questions": ["Q1", "Q2", "Q3"]
}}"""

    return user_content


# ============================================================
# Generation
# ============================================================

def get_max_tokens(model_name: str) -> int:
    """Set max tokens by model family."""
    lower_name = model_name.lower()

    if "gemini" in lower_name:
        return 5000
    if "claude" in lower_name:
        return 4000
    return 2500


def generate_questions(
    record: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_retries: int,
    timeout: float,
) -> Optional[str]:
    """Generate Cat1 questions using an OpenAI-compatible API."""

    user_prompt = build_prompt(record)
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
            return content.strip() if content else None

        except Exception as e:
            print(
                f"\nAttempt {attempt + 1}/{max_retries} failed "
                f"for case {record.get('id', 'unknown')}: {str(e)[:200]}"
            )

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                return None

    return None


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON object from model output."""
    if not text:
        return None

    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            return json.loads(text[start:end])

        return None

    except Exception as e:
        print(f"JSON parse error: {e}")
        return None


def normalize_source_case_id(record: Dict[str, Any]) -> str:
    """Convert root_case_xxx to xxx when possible."""
    raw_id = str(record.get("id", ""))
    return raw_id.replace("root_case_", "")


def flatten_to_qa_pairs(
    record: Dict[str, Any],
    questions_dict: Dict[str, Any],
    model_name: str,
) -> List[Dict[str, Any]]:
    """Flatten generated question groups into Cat1 QA records."""

    qa_pairs = []

    source_case_id = normalize_source_case_id(record)
    jurisdiction = record.get("jurisdiction", "the relevant jurisdiction")

    all_citations = [
        citation["cite"]
        for citation in record.get("citations_list", [])
        if isinstance(citation, dict) and citation.get("cite")
    ]

    all_questions = []
    all_questions.extend(questions_dict.get("legal_questions", []) or [])
    all_questions.extend(questions_dict.get("argument_questions", []) or [])
    all_questions.extend(questions_dict.get("compliance_questions", []) or [])

    for question in all_questions:
        if not isinstance(question, str) or not question.strip():
            continue

        qa_pairs.append(
            {
                "id": source_case_id,
                "qa_style": "1",
                "question": question.strip(),
                "ground_truth": all_citations,
                "jurisdiction": jurisdiction,
                "model": model_name.split("/")[-1],
            }
        )

    return qa_pairs


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
    """Process one case-summary file and write one Cat1 output file."""

    print("\n" + "=" * 60)
    print(f"Processing input file: {input_file}")
    print(f"Question generator model: {model_name}")
    print("=" * 60)

    all_records = load_jsonl(input_file)

    end_idx = end if end is not None else len(all_records)
    selected_records = all_records[start:end_idx]

    print(f"Total records in file: {len(all_records)}")
    print(f"Processing range: [{start}:{end_idx}]")
    print(f"Selected records: {len(selected_records)}")

    input_stem = input_file.name.replace("_case_summaries.jsonl", "")
    output_file = output_dir / f"cat1_{input_stem}.jsonl"

    processed_ids = load_processed_case_ids(output_file)

    to_process = [
        record for record in selected_records
        if normalize_source_case_id(record) not in processed_ids
    ]

    print(f"Already processed in output: {len(selected_records) - len(to_process)}")
    print(f"To process: {len(to_process)}")
    print(f"Output file: {output_file}")

    if not to_process:
        print("All done for this input file.")
        return

    total_qa = 0

    for record in tqdm(to_process, desc=model_name.split("/")[-1]):
        result = generate_questions(
            record=record,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )

        if not result:
            print(f"Empty generation for {record.get('id', 'unknown')}")
            continue

        questions_dict = parse_json_output(result)

        if not questions_dict:
            print(f"Failed to parse JSON for {record.get('id', 'unknown')}")
            continue

        qa_pairs = flatten_to_qa_pairs(
            record=record,
            questions_dict=questions_dict,
            model_name=model_name,
        )

        if qa_pairs:
            append_jsonl(output_file, qa_pairs)
            total_qa += len(qa_pairs)

    denom = len(to_process) if to_process else 1

    print(f"Generated {total_qa} QA pairs")
    print(f"Average: {total_qa / denom:.1f} per case")
    print(f"Saved to: {output_file}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cat1 citation retrieval QA pairs."
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
            "Question generator models. Must align with input_files order. "
            "If one model is provided for multiple files, it is reused."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save Cat1 output files.",
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
        default=0.9,
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
    print("Generate Category 1 Citation Retrieval QA")
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