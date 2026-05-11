"""
Generate structured case summaries from raw root cases.

This script is the first step in the LegalCiteBench construction pipeline.
Given raw case records, it uses an OpenAI-compatible API endpoint to generate
structured 500--700 word case summaries. These summaries are later used to
construct citation retrieval, citation completion, citation verification, case
matching, and case verification tasks.

The script supports:
- multiple generator models
- checkpoint/resume by case id
- configurable input/output paths
- retry logic

Example:
python data/raw/raw_to_case.py \
  --input_file data/raw/raw_1000case.jsonl \
  --output_dir outputs/case_summaries \
  --models openai/gpt-4o-mini google/gemini-2.5-flash anthropic/claude-3.5-haiku
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

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
# Default models
# ============================================================

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-3.5-haiku",
]


SYSTEM_PROMPT = """You are a legal expert creating detailed case summaries for legal research benchmarks.

Your summaries must be:
1. Factually accurate and comprehensive
2. Include all procedurally and substantively important details
3. Well-organized with clear sections
4. Written in objective, professional legal language

This summary will be used to generate various legal research questions, so include enough detail that someone unfamiliar with the case can understand all key aspects."""


# ============================================================
# IO helpers
# ============================================================

def sanitize_model_name(model_name: str) -> str:
    """Convert a model name into a filename-safe string."""
    return model_name.replace("/", "_").replace(":", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_processed_ids(output_file: Path) -> set:
    """Load already processed case IDs for checkpoint/resume."""
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(str(record["id"]))
            except Exception:
                pass

    return processed


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(record: Dict[str, Any]) -> str:
    """Build the user prompt for case summary generation."""

    case_name = record.get("case_name", "")
    court = record.get("court", "")
    date = record.get("decision_date", "")
    case_cite = record.get("case_cite", "")

    question_context = record.get("question_context", {})
    parties = "\n".join(question_context.get("parties", []))
    head_matter = question_context.get("head_matter", "")
    opinion_opening = question_context.get("opinion_opening", "")

    user_content = f"""Create a detailed case summary based on this court opinion.

CASE INFORMATION:
Name: {case_name}
Court: {court}
Date: {date}
Citation: {case_cite}

PARTIES:
{parties}

CASE HEADER:
{head_matter}

OPINION TEXT (Opening):
{opinion_opening}

YOUR TASK:
Write a comprehensive case summary with these sections:

**1. Case Background**
- Identify the parties and their roles (appellant/appellee, plaintiff/defendant)
- Describe the underlying dispute and what triggered the litigation
- Explain what relief or remedy is being sought

**2. Procedural History**
CRITICAL - Be precise about what happened at each court level:
- Trial Court: What did they decide? What was their reasoning? Which party won?
- Appeal: Who appealed? What were their main arguments?
- Key procedural motions or objections raised

**3. Key Facts** (Use numbered list, 8-12 facts)
Include specific factual details such as:
- Operational scope (Does entity operate in one location or statewide?)
- Governance structure (Who appoints leaders? Governor or local officials?)
- Funding sources (State funds, local funds, or mixed?)
- Specific statutory language cited
- Dates and timeline of events
- Any delegation agreements or special arrangements
- Physical location and jurisdiction
- Other legally relevant circumstances

**4. Legal Issue(s)**
State the precise legal question(s) the appellate court must resolve

**5. Court's Legal Analysis**
- What legal test, framework, or standard did the court apply?
- What prior precedents did the court discuss?
- How did the court interpret relevant statutes?
- What factors did the court weigh?

**6. Court's Decision and Holding**
CRITICAL - Be clear about the outcome:
- Did the appellate court AFFIRM, REVERSE, or REMAND?
- What specific order did the court issue?
- What is the binding legal holding?
- What happens next (if remanded)?

**7. Legal Principles Established**
What rules or principles does this case establish or reaffirm?

REQUIREMENTS:
- Length: 500-700 words
- Be factually accurate - extract from the opinion text provided
- Use clear section headers with **double asterisks**
- Write in objective, professional tone
- Include specific details (names, dates, statutory citations)
- DO NOT add any questions at the end
- This is a pure case summary

OUTPUT: [Complete case summary only, no questions]"""

    return user_content


# ============================================================
# Generation
# ============================================================

def generate_summary(
    record: Dict[str, Any],
    model_name: str,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    """Generate one case summary with retry logic."""

    user_prompt = build_prompt(record)

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
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"All retries exhausted for case {record.get('id', 'unknown')}")
                return ""

    return ""


def process_model(
    model_name: str,
    records: List[Dict[str, Any]],
    output_dir: Path,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> None:
    """Generate summaries for all records using one model."""

    output_file = output_dir / f"{sanitize_model_name(model_name)}_case_summaries.jsonl"

    print("\n" + "=" * 60)
    print(f"Processing with: {model_name}")
    print(f"Output file: {output_file}")
    print("=" * 60)

    processed_ids = load_processed_ids(output_file)
    to_process = [
        record for record in records
        if str(record.get("id", "")) not in processed_ids
    ]

    print(f"Already processed: {len(processed_ids)}")
    print(f"To process: {len(to_process)}")

    if not to_process:
        print(f"All cases already processed for {model_name}.")
        return

    success = 0

    with open(output_file, "a", encoding="utf-8") as out:
        for record in tqdm(to_process, desc=model_name):
            summary = generate_summary(
                record=record,
                model_name=model_name,
                max_retries=max_retries,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            if summary:
                result = dict(record)
                result["case_summary"] = summary
                result["model_used"] = model_name

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

                success += 1
            else:
                print(f"Empty summary for case {record.get('id', 'unknown')}")

    print(f"{model_name}: {success}/{len(to_process)} successful")
    print(f"Saved to: {output_file}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input raw case JSONL file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated case summaries.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Generator model names.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts per case.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Maximum output tokens.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API timeout in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Case Summaries")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 60)

    records = load_jsonl(input_file)

    print(f"\nTotal records: {len(records)}")

    for model_name in args.models:
        process_model(
            model_name=model_name,
            records=records,
            output_dir=output_dir,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    print("\n" + "=" * 60)
    print("All models completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()