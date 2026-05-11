"""
Generate Category 2 citation completion questions.

Input:
  Cat1 citation retrieval QA files.

Output:
  Cat2 QA pairs where the question already includes a subset of ground-truth
  citations and asks the model to provide the remaining important citations.

No API calls are needed. This is a deterministic text transformation given
a random seed.

Construction:
  - Start from a Cat1 QA pair with a full ground-truth citation list.
  - If the citation list has fewer than 3 citations, skip the example.
  - Randomly include 2 to min(4, total_citations - 1) citations in the question.
  - The remaining citations become the Cat2 ground truth.

Example:
python data/cat2/cat2.py \
  --input_files \
    data/cat1/cat1_openai_gpt-4o-mini.jsonl \
    data/cat1/cat1_google_gemini-2.5-flash.jsonl \
    data/cat1/cat1_anthropic_claude-3.5-haiku.jsonl \
  --output_dir data/cat2 \
  --seed 42
"""

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


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


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Append records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_keys(output_file: Path) -> set:
    """
    Load already processed keys.

    Cat1 can contain multiple questions for the same source case id, so using
    only `id` is not enough. We use id + question to avoid skipping distinct
    questions from the same root case.
    """
    if not output_file.exists():
        return set()

    processed = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(make_record_key(data))
            except Exception:
                pass

    return processed


def make_record_key(record: Dict[str, Any]) -> str:
    """Unique key for checkpointing."""
    return f"{record.get('id', '')}||{record.get('question', '')}"


# ============================================================
# Cat2 construction
# ============================================================

def format_included_citations(citations_to_include: List[str]) -> str:
    """Convert included citations into a natural sentence."""
    if len(citations_to_include) == 1:
        return f"I've already found {citations_to_include[0]}"

    if len(citations_to_include) == 2:
        return f"I've already found {citations_to_include[0]} and {citations_to_include[1]}"

    return (
        f"I've already found {', '.join(citations_to_include[:-1])}, "
        f"and {citations_to_include[-1]}"
    )


def modify_question_with_citations(
    question: str,
    citations_to_include: List[str],
) -> str:
    """
    Embed known citations into a Cat1 question and ask for the missing citations.
    """
    cites_text = format_included_citations(citations_to_include)

    replacements = {
        "What legal precedents and cases should I cite to answer this question?":
            f"{cites_text}. What other important cases am I missing that I should also cite?",
        "What cases should I cite to support this argument?":
            f"{cites_text}. What other important cases should I also cite to strengthen this argument?",
        "What legal authorities should I reference in this analysis?":
            f"{cites_text}. What other important legal authorities am I missing?",
    }

    modified = question

    for old_text, new_text in replacements.items():
        if old_text in modified:
            return modified.replace(old_text, new_text)

    # Fallback for questions whose ending differs from the template.
    return (
        question.strip()
        + f"\n\nNote: {cites_text}. What other important legal authorities should I also cite?"
    )


def create_category2_qa(
    qa_from_cat1: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create one Cat2 QA from one Cat1 QA.
    """
    all_citations = qa_from_cat1.get("ground_truth", [])

    if not isinstance(all_citations, list):
        return None, "ground_truth is not a list"

    all_citations = [c for c in all_citations if isinstance(c, str) and c.strip()]

    if len(all_citations) < 3:
        return None, f"Not enough citations: {len(all_citations)}"

    max_include = min(4, len(all_citations) - 1)
    num_to_include = rng.randint(2, max_include)

    included_citations = rng.sample(all_citations, num_to_include)
    remaining_citations = [
        citation for citation in all_citations
        if citation not in included_citations
    ]

    if not remaining_citations:
        return None, "No remaining citations"

    modified_question = modify_question_with_citations(
        question=qa_from_cat1["question"],
        citations_to_include=included_citations,
    )

    new_qa = {
        "id": qa_from_cat1.get("id"),
        "qa_style": "2",
        "question": modified_question,
        "ground_truth": remaining_citations,
        "included_citations": included_citations,
        "jurisdiction": qa_from_cat1.get("jurisdiction", "the relevant jurisdiction"),
        "model": qa_from_cat1.get("model"),
    }

    return new_qa, None


# ============================================================
# Processing
# ============================================================

def infer_output_name(input_file: Path) -> str:
    """
    Infer Cat2 output filename from Cat1 input filename.
    """
    stem = input_file.stem

    if stem.startswith("cat1_"):
        suffix = stem.replace("cat1_", "", 1)
    elif stem.startswith("1_"):
        suffix = stem.replace("1_", "", 1)
    else:
        suffix = stem

    return f"cat2_{suffix}.jsonl"


def process_file(
    input_file: Path,
    output_dir: Path,
    rng: random.Random,
) -> Dict[str, int]:
    """Process one Cat1 input file into one Cat2 output file."""

    print("\n" + "=" * 60)
    print(f"Processing: {input_file}")
    print("=" * 60)

    if not input_file.exists():
        print(f"ERROR: file not found: {input_file}")
        return {
            "input": 0,
            "output": 0,
            "skipped": 0,
            "errors": 0,
        }

    all_qa_cat1 = load_jsonl(input_file)
    print(f"Loaded {len(all_qa_cat1)} Cat1 QA pairs")

    output_file = output_dir / infer_output_name(input_file)
    processed_keys = load_processed_keys(output_file)

    to_process = [
        qa for qa in all_qa_cat1
        if make_record_key(qa) not in processed_keys
    ]

    print(f"Already processed: {len(all_qa_cat1) - len(to_process)}")
    print(f"To process: {len(to_process)}")
    print(f"Output file: {output_file}")

    success = 0
    skipped = 0
    errors = 0
    buffer = []

    for qa_cat1 in tqdm(to_process, desc=f"Cat2 - {input_file.stem}"):
        try:
            qa_cat2, error_msg = create_category2_qa(qa_cat1, rng)

            if qa_cat2 is None:
                skipped += 1
                if skipped <= 5:
                    print(f"\nSkipped {qa_cat1.get('id', 'unknown')}: {error_msg}")
                continue

            buffer.append(qa_cat2)
            success += 1

            if len(buffer) >= 100:
                append_jsonl(output_file, buffer)
                buffer = []

        except Exception as e:
            errors += 1
            print(f"\nERROR processing {qa_cat1.get('id', 'unknown')}: {e}")

            if errors <= 3:
                traceback.print_exc()

    if buffer:
        append_jsonl(output_file, buffer)

    print(f"\nResults for {input_file.name}:")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Saved to: {output_file}")

    return {
        "input": len(all_qa_cat1),
        "output": success,
        "skipped": skipped,
        "errors": errors,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cat2 citation completion QA pairs from Cat1."
    )

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Cat1 JSONL files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save Cat2 output files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting included citations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("=" * 60)
    print("Generate Category 2: Citation Completion")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    total_stats = {
        "input": 0,
        "output": 0,
        "skipped": 0,
        "errors": 0,
    }

    for input_path in args.input_files:
        stats = process_file(
            input_file=Path(input_path),
            output_dir=output_dir,
            rng=rng,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total input QA pairs:  {total_stats['input']}")
    print(f"Total output QA pairs: {total_stats['output']}")
    print(f"Total skipped:         {total_stats['skipped']}")
    print(f"Total errors:          {total_stats['errors']}")

    if total_stats["input"] > 0:
        success_rate = total_stats["output"] / total_stats["input"] * 100
        print(f"Success rate:          {success_rate:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()