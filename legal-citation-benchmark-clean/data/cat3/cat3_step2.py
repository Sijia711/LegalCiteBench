"""
Create Category 3 citation error detection QA pairs.

This is Step 2 of Cat3 construction.

Input:
  Step 1 legal analysis paragraphs.

Output:
  Two types of QA pairs:
  - 3-true: paragraph contains the original correct citation
  - 3-fake: paragraph contains a perturbed citation with one citation error

Fake citation error types:
  - page number error
  - volume number error
  - reporter series error

No API calls are needed in this step.

Example:
python data/cat3/cat3_step2_create_detection_pairs.py \
  --input_files \
    data/cat3/step1/cat3_step1_openai_gpt-4o-mini.jsonl \
    data/cat3/step1/cat3_step1_google_gemini-2.5-flash.jsonl \
    data/cat3/step1/cat3_step1_anthropic_claude-3.5-haiku.jsonl \
  --output_dir data/cat3/step2 \
  --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


FAKE_TYPES = ["page", "volume", "series"]


# ============================================================
# IO helpers
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Citation extraction and perturbation
# ============================================================

def extract_citations_from_paragraph(paragraph: str) -> List[Dict[str, Any]]:
    """Extract citation-like strings from a paragraph."""
    citations = []

    # Example: 466 U.S. 668 (1984)
    pattern_with_year = r"(\d+\s+[A-Za-z\.]+\d*[a-z]*\s+\d+)\s*\((\d{4})\)"

    for match in re.finditer(pattern_with_year, paragraph):
        citations.append(
            {
                "cite": match.group(1),
                "year": match.group(2),
                "full_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )

    # Example: 466 U.S. 668
    pattern_no_year = r"(?<!\()\b(\d+\s+[A-Za-z\.]+\d*[a-z]*\s+\d+)(?!\s*\()"

    for match in re.finditer(pattern_no_year, paragraph):
        cite = match.group(1)

        if not any(c["cite"] == cite for c in citations):
            citations.append(
                {
                    "cite": cite,
                    "year": None,
                    "full_match": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return citations


def modify_page_number(cite: str, rng: random.Random) -> str:
    """Modify the page number, usually the last numeric field."""
    match = re.match(r"(\d+\s+[A-Za-z\.]+\d*[a-z]*\s+)(\d+)", cite)

    if not match:
        return cite

    prefix, page = match.groups()
    offset = rng.choice([5, 10, -5, -10, 1, -1, 3, -3, 7])
    new_page = max(1, int(page) + offset)

    return f"{prefix}{new_page}"


def modify_volume(cite: str, rng: random.Random) -> str:
    """Modify the volume number, usually the first numeric field."""
    match = re.match(r"(\d+)(\s+[A-Za-z\.]+\d*[a-z]*\s+\d+)", cite)

    if not match:
        return cite

    volume, rest = match.groups()
    offset = rng.choice([1, -1, 2, -2, 5])
    new_volume = max(1, int(volume) + offset)

    return f"{new_volume}{rest}"


def modify_series(cite: str) -> str:
    """Modify reporter series, e.g., A.2d -> A.3d."""
    replacements = {
        "A.2d": "A.3d",
        "A.3d": "A.2d",
        "F.2d": "F.3d",
        "F.3d": "F.2d",
        "P.2d": "P.3d",
        "P.3d": "P.2d",
        "S.W.2d": "S.W.3d",
        "S.W.3d": "S.W.2d",
        "N.E.2d": "N.E.3d",
        "N.E.3d": "N.E.2d",
        "S.E.2d": "S.E.3d",
        "S.E.3d": "S.E.2d",
        "N.W.2d": "N.W.3d",
        "N.W.3d": "N.W.2d",
        "So.2d": "So.3d",
        "So.3d": "So.2d",
    }

    for old, new in replacements.items():
        if old in cite:
            return cite.replace(old, new)

    return cite


def normalize_cite(cite: str) -> str:
    """Normalize whitespace for citation matching."""
    return re.sub(r"\s+", " ", cite).strip()


def choose_main_citation(
    found_citations: List[Dict[str, Any]],
    target_cite: str,
) -> Optional[Dict[str, Any]]:
    """Find the intended citation in a generated paragraph."""
    target_norm = normalize_cite(target_cite)

    for citation in found_citations:
        if normalize_cite(citation["cite"]) == target_norm:
            return citation

    return found_citations[0] if found_citations else None


def create_fake_citation(
    record: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Create a perturbed citation inside one paragraph."""
    original_paragraph = record.get("original_paragraph", "")
    citation_used = record.get("citation_used", {})
    target_cite = citation_used.get("cite", "")

    found_citations = extract_citations_from_paragraph(original_paragraph)

    if not found_citations:
        return None, "no_citations_found"

    main_citation = choose_main_citation(found_citations, target_cite)

    if not main_citation:
        return None, "no_main_citation"

    original_cite = main_citation["cite"]

    fake_types = list(FAKE_TYPES)
    rng.shuffle(fake_types)

    for fake_type in fake_types:
        if fake_type == "page":
            fake_cite = modify_page_number(original_cite, rng)
        elif fake_type == "volume":
            fake_cite = modify_volume(original_cite, rng)
        else:
            fake_cite = modify_series(original_cite)

        if fake_cite != original_cite:
            modified_paragraph = original_paragraph.replace(original_cite, fake_cite, 1)

            return {
                "modified_paragraph": modified_paragraph,
                "original_cite": original_cite,
                "fake_cite": fake_cite,
                "fake_type": fake_type,
            }, None

    return None, "all_modifications_failed"


# ============================================================
# QA construction
# ============================================================

def build_detection_question(paragraph: str) -> str:
    """Build Cat3 detection prompt."""
    return f"""The following is a legal analysis paragraph. Please identify any incorrect citations and explain what is wrong.

LEGAL ANALYSIS:
{paragraph}

Task: Identify which citation is incorrect, specify what type of error it is (wrong page number, wrong volume number, or wrong reporter series), and provide the correct citation."""


def create_clean_qa_pair(record: Dict[str, Any]) -> Dict[str, Any]:
    """Create a 3-true QA pair with the original correct citation."""
    source_case_id = record["source_case_id"]
    legal_angle = record["legal_angle"]

    question = build_detection_question(record["original_paragraph"])

    return {
        "id": source_case_id,
        "qa_style": "3-true",
        "legal_angle": legal_angle,
        "question": question,
        "ground_truth": "There is no error in the citation.",
        "model": record["model"],
    }


def create_fake_qa_pair(
    record: Dict[str, Any],
    fake_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a 3-fake QA pair with one perturbed citation."""
    source_case_id = record["source_case_id"]
    legal_angle = record["legal_angle"]

    question = build_detection_question(fake_result["modified_paragraph"])

    return {
        "id": source_case_id,
        "qa_style": "3-fake",
        "legal_angle": legal_angle,
        "question": question,
        "ground_truth": (
            "The citation is incorrect. "
            f"The correct citation is: {fake_result['original_cite']}"
        ),
        "fake_type": fake_result["fake_type"],
        "fake_cite": fake_result["fake_cite"],
        "correct_cite": fake_result["original_cite"],
        "model": record["model"],
    }


# ============================================================
# Processing
# ============================================================

def infer_output_stem(input_file: Path) -> str:
    """Infer output suffix from step1 filename."""
    stem = input_file.stem

    if stem.startswith("cat3_step1_"):
        return stem.replace("cat3_step1_", "", 1)

    if stem.startswith("3_step1_"):
        return stem.replace("3_step1_", "", 1)

    return stem


def process_file(
    input_file: Path,
    output_dir: Path,
    rng: random.Random,
) -> Dict[str, int]:
    """Create clean and fake Cat3 examples from one step1 file."""
    print("\n" + "=" * 60)
    print(f"Processing: {input_file}")
    print("=" * 60)

    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} step1 records")

    output_stem = infer_output_stem(input_file)

    clean_output_file = output_dir / f"cat3_true_{output_stem}.jsonl"
    fake_output_file = output_dir / f"cat3_fake_{output_stem}.jsonl"

    clean_records = []
    fake_records = []

    stats = {
        "total": len(records),
        "clean_success": 0,
        "fake_success": 0,
        "fake_failed": 0,
        "no_citations_found": 0,
        "all_modifications_failed": 0,
        "other_errors": 0,
    }

    for record in tqdm(records, desc=f"Cat3 step2 - {output_stem}"):
        try:
            clean_qa = create_clean_qa_pair(record)
            clean_records.append(clean_qa)
            stats["clean_success"] += 1

            fake_result, error_type = create_fake_citation(record, rng)

            if fake_result is None:
                stats["fake_failed"] += 1

                if error_type in stats:
                    stats[error_type] += 1
                else:
                    stats["other_errors"] += 1

                continue

            fake_qa = create_fake_qa_pair(record, fake_result)
            fake_records.append(fake_qa)
            stats["fake_success"] += 1

        except Exception as e:
            stats["other_errors"] += 1
            print(f"\nError processing {record.get('id', 'unknown')}: {e}")

    write_jsonl(clean_output_file, clean_records)
    write_jsonl(fake_output_file, fake_records)

    print(f"\nResults for {input_file.name}:")
    print(f"  Clean success: {stats['clean_success']}")
    print(f"  Fake success:  {stats['fake_success']}")
    print(f"  Fake failed:   {stats['fake_failed']}")
    print(f"  Saved clean:   {clean_output_file}")
    print(f"  Saved fake:    {fake_output_file}")

    return stats


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Cat3 clean/fake citation detection QA pairs."
    )

    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("=" * 60)
    print("Category 3 Step 2: Create Citation Detection QA")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    total_stats = {
        "total": 0,
        "clean_success": 0,
        "fake_success": 0,
        "fake_failed": 0,
        "no_citations_found": 0,
        "all_modifications_failed": 0,
        "other_errors": 0,
    }

    for input_path in args.input_files:
        stats = process_file(
            input_file=Path(input_path),
            output_dir=output_dir,
            rng=rng,
        )

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for key, value in total_stats.items():
        print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()