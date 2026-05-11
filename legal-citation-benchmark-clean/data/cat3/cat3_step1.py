"""
Generate Category 3 intermediate legal analysis paragraphs.

This is Step 1 of Cat3 construction.

Input:
  Structured root-case summaries.

Output:
  Focused legal analysis paragraphs. Each paragraph uses exactly one citation
  from the source case and analyzes it from one legal angle.

Angles:
  - jurisdictional
  - statutory
  - operational
  - delegation

Each eligible root case produces up to 4 paragraphs, one per angle.

Example:
python data/cat3/cat3_step1_generate_analyses.py \
  --input_files \
    outputs/case_summaries/openai_gpt-4o-mini_case_summaries.jsonl \
    outputs/case_summaries/google_gemini-2.5-flash_case_summaries.jsonl \
    outputs/case_summaries/anthropic_claude-3.5-haiku_case_summaries.jsonl \
  --models \
    openai/gpt-4o-mini \
    google/gemini-2.5-flash \
    anthropic/claude-3.5-haiku \
  --output_dir data/cat3/step1 \
  --seed 42
"""

import argparse
import json
import os
import random
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

SYSTEM_PROMPT = """You are an experienced lawyer writing a focused legal analysis. Write concisely and professionally, making each citation count."""

LEGAL_ANGLES = [
    {
        "name": "jurisdictional",
        "focus": "jurisdictional classification and which court has proper authority",
        "guidance": (
            "Focus ONLY on: which court has jurisdiction, why one court rather "
            "than another, standards for determining proper venue, appellate vs. "
            "trial court authority. Do NOT discuss the substantive legal merits."
        ),
    },
    {
        "name": "statutory",
        "focus": "statutory interpretation and construction of the governing law",
        "guidance": (
            "Focus ONLY on: specific statutory provisions cited, how the court "
            "interpreted statutory language, canons of construction applied, "
            "and legislative intent discussed."
        ),
    },
    {
        "name": "operational",
        "focus": "operational scope, governance structure, and organizational authority",
        "guidance": (
            "Focus ONLY on: how the entity operates day-to-day, who governs it, "
            "funding sources, geographic scope, regulatory powers, reporting "
            "requirements, and functional characteristics."
        ),
    },
    {
        "name": "delegation",
        "focus": "delegation of authority and relationship with state agencies",
        "guidance": (
            "Focus ONLY on: whether authority was delegated and by whom, what "
            "powers were transferred, legal effect of delegation agreements, "
            "oversight relationships, and inter-governmental cooperation."
        ),
    },
]

WEAK_REASONS = [
    "emphasis added",
    "quotation omitted",
    "citations omitted",
    "citation omitted",
    "emphasis in original",
    "emphasis supplied",
    "internal quotation marks omitted",
]


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
    """Append records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def normalize_source_case_id(record: Dict[str, Any]) -> str:
    """Convert root_case_xxx to xxx when possible."""
    return str(record.get("id", "")).replace("root_case_", "")


def load_processed_case_ids(output_file: Path) -> set:
    """Load processed source case IDs."""
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(str(record.get("source_case_id", "")))
            except Exception:
                pass

    return processed


# ============================================================
# Citation filtering
# ============================================================

def is_strong_reason(reason: Optional[str]) -> bool:
    """Heuristic filter for substantive citation reasons."""
    if not reason:
        return False

    reason_lower = reason.lower().strip()

    for weak in WEAK_REASONS:
        if reason_lower == weak or reason_lower.startswith(weak):
            return False

    words = reason.split()

    if len(words) < 4:
        return False

    if ", j." in reason_lower or ", j.," in reason_lower:
        if len(words) < 6:
            return False

    if reason_lower in ["dissenting", "concurring", "plurality", "per curiam"]:
        return False

    if reason_lower.startswith("definition of ") and len(words) < 6:
        return False

    return True


def select_citations_with_strong_reasons(
    citations_list: List[Dict[str, Any]],
    num_needed: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Select citations with substantive reasons.

    If fewer than num_needed citations have strong reasons, backfill with
    citations without reasons so that the case can still be used when possible.
    """
    with_strong = [
        citation
        for citation in citations_list
        if citation.get("reason") and is_strong_reason(citation.get("reason"))
    ]

    if len(with_strong) < num_needed:
        without_reason = [
            citation
            for citation in citations_list
            if not citation.get("reason")
        ]

        rng.shuffle(without_reason)
        with_strong.extend(without_reason[: num_needed - len(with_strong)])

    rng.shuffle(with_strong)
    return with_strong[:num_needed]


# ============================================================
# Prompt construction and generation
# ============================================================

def build_legal_analysis_prompt(
    record: Dict[str, Any],
    citation: Dict[str, Any],
    angle: Dict[str, str],
) -> str:
    """Build prompt for generating a legal analysis paragraph."""
    case_summary = record.get("case_summary", "")
    cite = citation.get("cite", "No citation")
    year = citation.get("year", "")
    reason = citation.get("reason", "")

    citation_text = f"Citation: {cite}"

    if year:
        citation_text += f" ({year})"

    if reason:
        citation_text += f"\nHolding: {reason}"

    user_content = f"""You are writing a focused legal analysis paragraph for a brief based on this case.

CASE SUMMARY:
{case_summary}

CRITICAL CONSTRAINT:
You must analyze this case ONLY from the {angle["name"]} perspective.

{angle["guidance"]}

CITATION TO USE:
{citation_text}

YOUR TASK:
Write a 120-150 word legal analysis paragraph that:

1. Stays strictly within {angle["focus"]}
2. Uses the citation provided above and explains it in depth
3. Explains what the cited case held, why the holding matters, and how it relates to the {angle["name"]} issue
4. Builds the analysis around this single precedent
5. Uses proper citation format: [discussion], {cite} ({year})
6. Sounds like a concise legal brief paragraph

CRITICAL REQUIREMENTS:
- Use exactly the citation provided above
- Do not add additional citations
- Stay focused on {angle["focus"]}
- Write 120-150 words total
- Output only the paragraph text, no commentary."""

    return user_content


def get_max_tokens(model_name: str) -> int:
    """Set max token budget by model family."""
    lower_name = model_name.lower()

    if "gemini" in lower_name:
        return 2000

    return 600


def generate_analysis(
    prompt_text: str,
    model_name: str,
    temperature: float,
    max_retries: int,
    timeout: float,
) -> str:
    """Generate a legal analysis paragraph."""
    max_tokens = get_max_tokens(model_name)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            print(f"\nAttempt {attempt + 1}/{max_retries} failed: {str(e)[:200]}")

            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                return ""

    return ""


# ============================================================
# Processing
# ============================================================

def process_input_file(
    input_file: Path,
    model_name: str,
    output_dir: Path,
    start: int,
    end: Optional[int],
    rng: random.Random,
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
    output_file = output_dir / f"cat3_step1_{input_stem}.jsonl"

    processed_ids = load_processed_case_ids(output_file)

    to_process = [
        record
        for record in selected_records
        if normalize_source_case_id(record) not in processed_ids
    ]

    print(f"Already processed: {len(selected_records) - len(to_process)}")
    print(f"To process: {len(to_process)}")
    print(f"Will generate up to: {len(to_process) * len(LEGAL_ANGLES)} analyses")
    print(f"Output file: {output_file}")

    if not to_process:
        print("All done for this input file.")
        return

    total_generated = 0
    skipped = 0
    by_angle = {angle["name"]: 0 for angle in LEGAL_ANGLES}

    buffer = []

    for record in tqdm(to_process, desc=model_name.split("/")[-1]):
        citations = select_citations_with_strong_reasons(
            citations_list=record.get("citations_list", []),
            num_needed=len(LEGAL_ANGLES),
            rng=rng,
        )

        if len(citations) < len(LEGAL_ANGLES):
            skipped += 1
            continue

        source_case_id = normalize_source_case_id(record)

        for version, (citation, angle) in enumerate(zip(citations, LEGAL_ANGLES), 1):
            prompt = build_legal_analysis_prompt(record, citation, angle)

            paragraph = generate_analysis(
                prompt_text=prompt,
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
            )

            if not paragraph:
                continue

            output_data = {
                "id": f"{source_case_id}_cat3_v{version}",
                "source_case_id": source_case_id,
                "version": version,
                "legal_angle": angle["name"],
                "original_paragraph": paragraph,
                "citation_used": citation,
                "all_citations": [
                    c.get("cite", "")
                    for c in record.get("citations_list", [])
                    if isinstance(c, dict)
                ],
                "model": model_name.split("/")[-1],
            }

            buffer.append(output_data)
            total_generated += 1
            by_angle[angle["name"]] += 1

            if len(buffer) >= 100:
                append_jsonl(output_file, buffer)
                buffer = []

    if buffer:
        append_jsonl(output_file, buffer)

    print(f"Total generated: {total_generated}")
    print(f"Successful cases approximately: {total_generated // len(LEGAL_ANGLES)}")
    print(f"Skipped cases: {skipped}")
    print("Angle distribution:")
    for angle_name, count in by_angle.items():
        print(f"  - {angle_name}: {count}")
    print(f"Saved to: {output_file}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cat3 intermediate legal analyses."
    )

    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)

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

    rng = random.Random(args.seed)

    print("=" * 60)
    print("Category 3 Step 1: Generate Legal Analyses")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    for input_file, model_name in zip(input_files, models):
        process_input_file(
            input_file=input_file,
            model_name=model_name,
            output_dir=output_dir,
            start=args.start,
            end=args.end,
            rng=rng,
            temperature=args.temperature,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )

    print("\n" + "=" * 60)
    print("All completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()