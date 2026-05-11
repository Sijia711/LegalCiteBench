"""
Generate Category 4-2 true case verification questions.

This is Step 1 of Cat4-2 construction.

Input:
  Structured root-case summaries.

Output:
  Cat4-2 true verification questions. Each question contains a legal analysis
  ending with a request like:
  "Can I reference [case name], [citation] ([year]) for [principle]?"

Ground truth:
  The source root case metadata.

Each root case can produce up to 4 questions, one per legal angle:
  - jurisdictional
  - statutory
  - operational
  - delegation

Example:
python data/cat4/cat4_2_step1_generate_verification_questions.py \
  --input_files \
    outputs/case_summaries/openai_gpt-4o-mini_case_summaries.jsonl \
    outputs/case_summaries/google_gemini-2.5-flash_case_summaries.jsonl \
    outputs/case_summaries/anthropic_claude-3.5-haiku_case_summaries.jsonl \
  --models \
    openai/gpt-4o-mini \
    google/gemini-2.5-flash \
    anthropic/claude-3.5-haiku \
  --output_dir data/cat4/cat4-2/step1
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


load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)


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
            "Focus ONLY on: which court has jurisdiction, why one court rather than another, "
            "standards for determining proper venue, appellate vs. trial court authority. "
            "Do NOT discuss the substantive legal merits."
        ),
        "example_principle": "which court has jurisdiction over disputes involving an entity type",
    },
    {
        "name": "statutory",
        "focus": "statutory interpretation and construction of the governing law",
        "guidance": (
            "Focus ONLY on: specific statutory provisions cited, how the court interpreted "
            "statutory language, canons of construction applied, and legislative intent discussed."
        ),
        "example_principle": "how courts should interpret a specific statutory provision in context",
    },
    {
        "name": "operational",
        "focus": "operational scope, governance structure, and organizational authority",
        "guidance": (
            "Focus ONLY on: how the entity actually operates day-to-day, who governs it, "
            "funding sources, geographic scope, regulatory powers, and reporting requirements. "
            "Avoid discussing legal classification; focus on functional characteristics."
        ),
        "example_principle": "what operational characteristics define an entity's authority and governance",
    },
    {
        "name": "delegation",
        "focus": "delegation of authority and relationship with state agencies",
        "guidance": (
            "Focus ONLY on: whether authority was delegated and by whom, what powers were "
            "transferred, legal effect of delegation agreements, oversight relationships, and "
            "how inter-governmental cooperation affects legal status."
        ),
        "example_principle": "how delegation agreements affect an entity's legal status or authority",
    },
]


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def normalize_source_case_id(record: Dict[str, Any]) -> str:
    return str(record.get("id", "")).replace("root_case_", "")


def load_processed_keys(output_file: Path) -> set:
    """
    Load processed source_case_id + legal_angle keys.
    One root case generates multiple angles, so case id alone is not enough.
    """
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                key = f"{data.get('source_case_id', '')}||{data.get('legal_angle', '')}"
                processed.add(key)
            except Exception:
                pass

    return processed


def build_legal_analysis_prompt(record: Dict[str, Any], angle: Dict[str, str]) -> str:
    case_summary = record.get("case_summary", "")
    case_name = record.get("case_name", "Unknown Case")
    case_cite = record.get("case_cite", "No Citation")
    decision_date = record.get("decision_date", "")
    court = record.get("court", "")
    jurisdiction = record.get("jurisdiction", "")

    year = decision_date.split("-")[0] if decision_date else ""

    case_reference = f"{case_name}, {case_cite}"
    if year:
        case_reference += f" ({year})"

    user_content = f"""You are writing a focused legal analysis paragraph based on this case.

CASE INFORMATION:
Case Name: {case_name}
Citation: {case_cite}
Decision Date: {decision_date}
Court: {court}
Jurisdiction: {jurisdiction}

CASE SUMMARY:
{case_summary}

CRITICAL CONSTRAINT:
You must analyze this case ONLY from the {angle["name"]} perspective.

{angle["guidance"]}

YOUR TASK:
Write a 120-150 word legal analysis paragraph that:

1. Stays strictly within {angle["focus"]}
2. Identifies ONE specific legal principle this case establishes about {angle["name"]} issues
3. Explains:
   - what specific holding relates to {angle["name"]};
   - why this {angle["name"]}-specific holding matters;
   - what practitioners should know about this issue from the case
4. Ends with exactly this form:
   "Can I reference {case_reference} for [your identified {angle["name"]}-specific principle]?"
5. Makes the final question specific to this legal angle

Example of a good {angle["name"]} principle:
{angle["example_principle"]}

CRITICAL REQUIREMENTS:
- Write 120-150 words total
- Do not overlap heavily with other legal angles
- Do not add unrelated citations
- Output only the paragraph text, no commentary."""

    return user_content


def get_max_tokens(model_name: str) -> int:
    lower_name = model_name.lower()

    if "gemini" in lower_name:
        return 1200
    if "claude" in lower_name:
        return 1000
    return 800


def generate_analysis(
    prompt_text: str,
    model_name: str,
    temperature: float,
    max_retries: int,
    timeout: float,
) -> str:
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


def create_true_qa(
    record: Dict[str, Any],
    legal_angle: str,
    paragraph: str,
    model_name: str,
) -> Dict[str, Any]:
    source_case_id = normalize_source_case_id(record)

    return {
        "id": source_case_id,
        "source_case_id": source_case_id,
        "qa_style": "4-2-true",
        "legal_angle": legal_angle,
        "question": paragraph,
        "ground_truth": {
            "case_name": record.get("case_name", ""),
            "case_cite": record.get("case_cite", ""),
            "decision_date": record.get("decision_date", ""),
            "court": record.get("court", ""),
            "jurisdiction": record.get("jurisdiction", ""),
        },
        "model": model_name.split("/")[-1],
    }


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
    output_file = output_dir / f"cat4-2_true_{input_stem}.jsonl"

    processed_keys = load_processed_keys(output_file)

    to_generate = []
    for record in selected_records:
        source_case_id = normalize_source_case_id(record)
        for angle in LEGAL_ANGLES:
            key = f"{source_case_id}||{angle['name']}"
            if key not in processed_keys:
                to_generate.append((record, angle))

    print(f"Already generated: {len(selected_records) * len(LEGAL_ANGLES) - len(to_generate)}")
    print(f"To generate: {len(to_generate)}")
    print(f"Output file: {output_file}")

    if not to_generate:
        print("All done for this input file.")
        return

    total_generated = 0
    by_angle = {angle["name"]: 0 for angle in LEGAL_ANGLES}
    buffer = []

    for record, angle in tqdm(to_generate, desc=model_name.split("/")[-1]):
        prompt = build_legal_analysis_prompt(record, angle)

        paragraph = generate_analysis(
            prompt_text=prompt,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )

        if not paragraph:
            continue

        qa = create_true_qa(
            record=record,
            legal_angle=angle["name"],
            paragraph=paragraph,
            model_name=model_name,
        )

        buffer.append(qa)
        total_generated += 1
        by_angle[angle["name"]] += 1

        if len(buffer) >= 100:
            append_jsonl(output_file, buffer)
            buffer = []

    if buffer:
        append_jsonl(output_file, buffer)

    print(f"Total generated: {total_generated}")
    print("Angle distribution:")
    for angle_name, count in by_angle.items():
        print(f"  - {angle_name}: {count}")
    print(f"Saved to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cat4-2 true case verification questions."
    )

    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=0.7)
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

    print("=" * 60)
    print("Category 4-2 Step 1: Generate True Verification Questions")
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