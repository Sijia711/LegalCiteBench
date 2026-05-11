"""
Prompt-only mitigation experiment for LegalCiteBench.

For each input question, this script generates TWO outputs:

1. original:
   Uses the original legal-expert system prompt.

2. abstention_instruction:
   Uses the same legal-expert prompt plus an explicit instruction:
   if uncertain, do not fabricate citations.

Then it judges both outputs with Qwen/Qwen3-32B using vLLM and produces
a comparison summary.

Designed for cluster GPU execution.

Example:
python scripts/run_prompt_mitigation_pair_vllm.py \
  --cat1_file /path/to/cat1_citation_retrieval.jsonl \
  --cat2_file /path/to/cat2_citation_completeness.jsonl \
  --output_dir /path/to/output_dir \
  --generator_model Qwen/Qwen3-14B \
  --judge_model Qwen/Qwen3-32B \
  --samples_per_category 100
"""

import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============================================================
# Generation prompts
# ============================================================

SYSTEM_PROMPT_ORIGINAL = """You are a legal research expert.
When asked a question, provide the relevant case citations or legal authorities directly.
Be direct and concise. Only list the citations or cases asked for."""

SYSTEM_PROMPT_ABSTENTION = """You are a legal research expert.
When asked a question, provide the relevant case citations or legal authorities directly.

Important:
If you are not certain about the exact legal citation, do not guess.
Instead, state that you cannot verify the citation and briefly explain the relevant legal issue without inventing case names or reporter information.

Be direct and concise."""

PROMPT_MODES = {
    "original": SYSTEM_PROMPT_ORIGINAL,
    "abstention_instruction": SYSTEM_PROMPT_ABSTENTION,
}


# ============================================================
# Judge prompts
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are a precise legal citation evaluator.
You must respond with valid JSON only.
Do not include markdown fences.
Ignore formatting differences and evaluate citation content only."""

JUDGE_PROMPT_CITATION = """You are evaluating legal citation outputs.

Compare the MODEL OUTPUT against GROUND TRUTH CITATIONS using substring matching, case-insensitive.

MODEL OUTPUT:
{output}

GROUND TRUTH CITATIONS:
{ground_truth}

Instructions:
- Extract all legal case citations from the model output.
- A citation in the output is CORRECT if it CONTAINS or IS CONTAINED BY a ground truth citation, case-insensitive.
  Example: "466 U.S. 668" matches "Strickland v. Washington, 466 U.S. 668 (1984)".
  Example: "231 Ariz. 150" does NOT match "231 Ariz. 145".
- A citation is HALLUCINATED if it does not match any ground truth citation.
- A ground truth citation is MISSED if it is not matched by any output citation.
- If the model explicitly says it cannot verify, does not know, or refuses to guess, set abstained=true.
- If the output contains no citation and does not provide an answer, set abstained=true.
- If the output contains no citation but gives legal discussion without claiming specific authority, set abstained=true.
- If the output invents or provides citations that are not in the ground truth, set those citations as hallucinated.

Reply JSON only with this exact schema:
{{
  "abstained": true,
  "correct_citations": [],
  "hallucinated_citations": [],
  "missed_citations": [],
  "total_output_count": 0,
  "total_gt_count": 0,
  "reasoning": "one short sentence"
}}"""


# ============================================================
# Basic helpers
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


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Append records to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    """Write JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_ground_truth(gt: Any) -> str:
    """Format ground-truth citation list for judge prompt."""
    if isinstance(gt, list):
        if gt and isinstance(gt[0], list):
            lines = []
            for item in gt:
                if isinstance(item, list) and len(item) == 2:
                    lines.append(f"- {item[0]}: {item[1]}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        return "\n".join(f"- {x}" for x in gt)
    return str(gt)


def make_base_key(record: Dict[str, Any]) -> str:
    """Question-level key, shared by original and abstention outputs."""
    qid = str(record.get("id", ""))
    qa_style = str(record.get("qa_style", ""))
    legal_angle = str(record.get("legal_angle", ""))
    return f"{qid}||{qa_style}||{legal_angle}"


def make_generation_key(record: Dict[str, Any], generator_model: str) -> str:
    """Unique key for generation checkpoint."""
    return f"{make_base_key(record)}||{record.get('prompt_mode')}||{generator_model}"


def make_judge_key(record: Dict[str, Any], judge_model: str) -> str:
    """Unique key for judge checkpoint."""
    return f"{make_generation_key(record, str(record.get('model', '')))}||judge={judge_model}"


def load_processed_generation_keys(path: Path, generator_model: str) -> set:
    """Load processed generation keys."""
    if not path.exists():
        return set()

    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                keys.add(make_generation_key(d, generator_model))
            except Exception:
                pass
    return keys


def load_processed_judge_keys(path: Path, judge_model: str) -> set:
    """Load processed judge keys."""
    if not path.exists():
        return set()

    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                keys.add(make_judge_key(d, judge_model))
            except Exception:
                pass
    return keys


# ============================================================
# Data preparation
# ============================================================

def load_and_make_pairs(
    cat1_file: Path,
    cat2_file: Path,
    samples_per_category: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Load Cat1/Cat2.
    For each question, create two generation records:
    - original
    - abstention_instruction
    """
    random.seed(seed)
    sampled_questions = []

    for path in [cat1_file, cat2_file]:
        records = load_jsonl(path)
        print(f"Loaded {len(records):>6} from {path.name}")

        if samples_per_category > 0 and len(records) > samples_per_category:
            records = random.sample(records, samples_per_category)

        print(f"Sampled {len(records):>6} from {path.name}")
        sampled_questions.extend(records)

    paired_records = []
    for record in sampled_questions:
        for prompt_mode in ["original", "abstention_instruction"]:
            new_record = dict(record)
            new_record["prompt_mode"] = prompt_mode
            paired_records.append(new_record)

    print(f"Questions sampled: {len(sampled_questions)}")
    print(f"Generation records after pairing: {len(paired_records)}")
    return paired_records


# ============================================================
# Generation
# ============================================================

def build_generation_prompt(tokenizer, question: str, prompt_mode: str) -> str:
    """Build generation prompt."""
    system_prompt = PROMPT_MODES[prompt_mode]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generation(
    records: List[Dict[str, Any]],
    generator_model: str,
    generation_file: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_new_tokens: int,
) -> None:
    """Generate original + abstention outputs with checkpointing."""
    print("\n" + "=" * 80)
    print("STAGE 1: GENERATION")
    print(f"Generator model: {generator_model}")
    print(f"Generation file: {generation_file}")
    print("=" * 80)

    processed = load_processed_generation_keys(generation_file, generator_model)
    todo = [
        r for r in records
        if make_generation_key(r, generator_model) not in processed
    ]

    print(f"Already generated: {len(processed)}")
    print(f"To generate: {len(todo)}")

    if not todo:
        print("Generation already complete.")
        return

    llm = LLM(
        model=generator_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    tokenizer = llm.get_tokenizer()

    prompts = [
        build_generation_prompt(
            tokenizer=tokenizer,
            question=r["question"],
            prompt_mode=r["prompt_mode"],
        )
        for r in todo
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    generated_records = []

    for record, output in tqdm(
        zip(todo, outputs),
        total=len(todo),
        desc="Saving generations",
    ):
        text = output.outputs[0].text.strip()

        result = {
            "id": record.get("id"),
            "qa_style": record.get("qa_style"),
            "legal_angle": record.get("legal_angle"),
            "prompt_mode": record.get("prompt_mode"),
            "question": record.get("question"),
            "ground_truth": record.get("ground_truth"),
            "model": generator_model,
            "output": text,
        }
        generated_records.append(result)

    append_jsonl(generation_file, generated_records)
    print(f"Saved generations to: {generation_file}")


# ============================================================
# Judge
# ============================================================

def strip_qwen_think(text: str) -> str:
    """Remove Qwen3 <think> blocks if present."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from judge output."""
    if not text:
        return None

    text = strip_qwen_think(text)
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


def build_judge_prompt(tokenizer, record: Dict[str, Any]) -> str:
    """Build judge prompt for citation matching."""
    output = record.get("output", "") or "[NO RESPONSE]"
    gt_str = format_ground_truth(record.get("ground_truth"))

    user_prompt = JUDGE_PROMPT_CITATION.format(
        output=output,
        ground_truth=gt_str,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_f1(judge_result: Dict[str, Any]) -> Dict[str, float]:
    """Compute precision, recall, F1."""
    correct = len(judge_result.get("correct_citations", []) or [])
    hallucinated = len(judge_result.get("hallucinated_citations", []) or [])
    missed = len(judge_result.get("missed_citations", []) or [])

    total_output = judge_result.get("total_output_count", None)
    total_gt = judge_result.get("total_gt_count", None)

    if not isinstance(total_output, int) or total_output < 0:
        total_output = correct + hallucinated

    if not isinstance(total_gt, int) or total_gt <= 0:
        total_gt = correct + missed

    precision = correct / total_output if total_output > 0 else 0.0
    recall = correct / total_gt if total_gt > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_count": correct,
        "hallucinated_count": hallucinated,
        "missed_count": missed,
        "total_output_count": total_output,
        "total_gt_count": total_gt,
    }


def classify_hallucination(judged: Dict[str, Any]) -> bool:
    """
    Hallucination indicator:
    - response exists
    - not abstained
    - has hallucinated citations OR F1 <= 0.4
    """
    responded = bool(judged.get("responded", False))
    abstained = bool(judged.get("abstained", False))
    hallucinated_count = int(judged.get("hallucinated_count", 0))
    f1 = float(judged.get("f1", 0.0))

    if not responded or abstained:
        return False

    if hallucinated_count > 0:
        return True

    if f1 <= 0.4:
        return True

    return False


def run_judge(
    generation_file: Path,
    judge_file: Path,
    judge_model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_new_tokens: int,
) -> None:
    """Judge both original and abstention outputs with checkpointing."""
    print("\n" + "=" * 80)
    print("STAGE 2: JUDGE")
    print(f"Judge model: {judge_model}")
    print(f"Generation file: {generation_file}")
    print(f"Judge file: {judge_file}")
    print("=" * 80)

    generation_records = load_jsonl(generation_file)
    processed = load_processed_judge_keys(judge_file, judge_model)

    todo = [
        r for r in generation_records
        if make_judge_key(r, judge_model) not in processed
    ]

    print(f"Generation records: {len(generation_records)}")
    print(f"Already judged: {len(processed)}")
    print(f"To judge: {len(todo)}")

    if not todo:
        print("Judging already complete.")
        return

    llm = LLM(
        model=judge_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    tokenizer = llm.get_tokenizer()

    prompts = [
        build_judge_prompt(tokenizer=tokenizer, record=r)
        for r in todo
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    judged_records = []

    for record, output in tqdm(
        zip(todo, outputs),
        total=len(todo),
        desc="Saving judge results",
    ):
        raw_text = output.outputs[0].text.strip()
        parsed = extract_json(raw_text)

        if parsed is None:
            parsed = {
                "abstained": False,
                "correct_citations": [],
                "hallucinated_citations": [],
                "missed_citations": [],
                "total_output_count": 0,
                "total_gt_count": 0,
                "reasoning": "JSON parse failed",
                "error": "JSON parse failed",
                "raw_judge_output": raw_text,
            }

        metrics = compute_f1(parsed)

        responded = bool((record.get("output") or "").strip())
        abstained = bool(parsed.get("abstained", False))

        judged = {
            "id": record.get("id"),
            "qa_style": record.get("qa_style"),
            "legal_angle": record.get("legal_angle"),
            "prompt_mode": record.get("prompt_mode"),
            "question": record.get("question"),
            "ground_truth": record.get("ground_truth"),
            "model": record.get("model"),
            "output": record.get("output"),
            "responded": responded,
            "abstained": abstained,
            "judge_model": judge_model,
            "judge_detail": parsed,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "judge_score_0_5": round(metrics["f1"] * 5, 4),
            "correct_count": metrics["correct_count"],
            "hallucinated_count": metrics["hallucinated_count"],
            "missed_count": metrics["missed_count"],
            "total_output_count": metrics["total_output_count"],
            "total_gt_count": metrics["total_gt_count"],
        }

        judged["hallucinated"] = classify_hallucination(judged)
        judged_records.append(judged)

    append_jsonl(judge_file, judged_records)
    print(f"Saved judge results to: {judge_file}")


# ============================================================
# Summary
# ============================================================

def summarize(judge_file: Path) -> Dict[str, Any]:
    """Summarize metrics by prompt mode."""
    records = load_jsonl(judge_file)

    groups = defaultdict(list)
    for r in records:
        model = str(r.get("model", "unknown"))
        prompt_mode = str(r.get("prompt_mode", "unknown"))
        groups[(model, prompt_mode)].append(r)

    summary = {}

    for (model, prompt_mode), recs in groups.items():
        n = len(recs)

        responded = [r for r in recs if r.get("responded", False)]
        abstained = [r for r in recs if r.get("abstained", False)]
        non_abstain = [
            r for r in recs
            if r.get("responded", False) and not r.get("abstained", False)
        ]

        hallucinated = [r for r in recs if r.get("hallucinated", False)]
        correct = [r for r in recs if float(r.get("f1", 0.0)) >= 0.999]

        f1_values = [float(r.get("f1", 0.0)) for r in recs]
        precision_values = [float(r.get("precision", 0.0)) for r in recs]
        recall_values = [float(r.get("recall", 0.0)) for r in recs]

        key = f"{model}||{prompt_mode}"

        summary[key] = {
            "model": model,
            "prompt_mode": prompt_mode,
            "n": n,
            "response_rate": len(responded) / n if n else 0.0,
            "abstain_rate": len(abstained) / n if n else 0.0,
            "hallucination_rate": len(hallucinated) / len(non_abstain) if non_abstain else 0.0,
            "mar": len(hallucinated) / n if n else 0.0,
            "correct_rate": len(correct) / n if n else 0.0,
            "citation_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
            "recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        }

    return summary


def make_pairwise_comparison(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Create original vs abstention comparison for each model."""
    by_model = defaultdict(dict)

    for _, s in summary.items():
        by_model[s["model"]][s["prompt_mode"]] = s

    comparisons = {}

    for model, modes in by_model.items():
        if "original" not in modes or "abstention_instruction" not in modes:
            continue

        orig = modes["original"]
        abst = modes["abstention_instruction"]

        comparisons[model] = {
            "model": model,
            "original": orig,
            "abstention_instruction": abst,
            "delta": {
                "citation_f1": abst["citation_f1"] - orig["citation_f1"],
                "correct_rate": abst["correct_rate"] - orig["correct_rate"],
                "hallucination_rate": abst["hallucination_rate"] - orig["hallucination_rate"],
                "mar": abst["mar"] - orig["mar"],
                "abstain_rate": abst["abstain_rate"] - orig["abstain_rate"],
            },
        }

    return comparisons


def print_summary(summary: Dict[str, Any]) -> None:
    """Print summary table."""
    print("\n" + "=" * 110)
    print("PROMPT-ONLY MITIGATION SUMMARY")
    print("=" * 110)
    print(
        f"{'Model':<36} {'Prompt':<24} "
        f"{'F1':>8} {'Correct':>8} {'Halluc':>8} {'MAR':>8} {'Abstain':>8}"
    )
    print("-" * 110)

    for _, s in sorted(summary.items()):
        print(
            f"{s['model']:<36} {s['prompt_mode']:<24} "
            f"{s['citation_f1']:>8.3f} "
            f"{s['correct_rate']:>8.1%} "
            f"{s['hallucination_rate']:>8.1%} "
            f"{s['mar']:>8.1%} "
            f"{s['abstain_rate']:>8.1%}"
        )


def print_comparison(comparisons: Dict[str, Any]) -> None:
    """Print original vs abstention delta."""
    print("\n" + "=" * 110)
    print("ORIGINAL VS ABSTENTION DELTA")
    print("=" * 110)
    print(
        f"{'Model':<36} "
        f"{'Delta F1':>10} {'Delta Correct':>14} "
        f"{'Delta Halluc':>14} {'Delta MAR':>12} {'Delta Abstain':>14}"
    )
    print("-" * 110)

    for model, c in comparisons.items():
        d = c["delta"]
        print(
            f"{model:<36} "
            f"{d['citation_f1']:>10.3f} "
            f"{d['correct_rate']:>14.1%} "
            f"{d['hallucination_rate']:>14.1%} "
            f"{d['mar']:>12.1%} "
            f"{d['abstain_rate']:>14.1%}"
        )


def make_latex_rows(summary: Dict[str, Any]) -> str:
    """Create LaTeX rows for paper."""
    rows = []

    prompt_order = ["original", "abstention_instruction"]
    prompt_names = {
        "original": "Original prompt",
        "abstention_instruction": "Abstention instruction",
    }

    sorted_items = sorted(
        summary.values(),
        key=lambda x: prompt_order.index(x["prompt_mode"])
        if x["prompt_mode"] in prompt_order else 99,
    )

    for s in sorted_items:
        row = (
            f"{prompt_names.get(s['prompt_mode'], s['prompt_mode'])} & "
            f"{s['citation_f1']:.3f} & "
            f"{s['correct_rate'] * 100:.1f}\\% & "
            f"{s['hallucination_rate'] * 100:.1f}\\% & "
            f"{s['mar'] * 100:.1f}\\% & "
            f"{s['abstain_rate'] * 100:.1f}\\% \\\\"
        )
        rows.append(row)

    return "\n".join(rows)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cat1_file", type=str, required=True)
    parser.add_argument("--cat2_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--generator_model", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B")

    parser.add_argument("--samples_per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gen_max_model_len", type=int, default=4096)
    parser.add_argument("--gen_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)

    parser.add_argument("--judge_max_model_len", type=int, default=8192)
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--judge_max_new_tokens", type=int, default=1024)

    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_dir = output_dir / "generations"
    judge_dir = output_dir / "judged"
    summary_dir = output_dir / "summary"

    generation_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    gen_name = sanitize_model_name(args.generator_model)
    judge_name = sanitize_model_name(args.judge_model)

    generation_file = generation_dir / f"{gen_name}_paired_outputs.jsonl"
    judge_file = judge_dir / f"judge_{gen_name}_paired_by_{judge_name}.jsonl"
    summary_file = summary_dir / f"summary_{gen_name}_paired_by_{judge_name}.json"
    comparison_file = summary_dir / f"comparison_{gen_name}_paired_by_{judge_name}.json"
    latex_file = summary_dir / f"latex_rows_{gen_name}_paired_by_{judge_name}.txt"

    records = load_and_make_pairs(
        cat1_file=Path(args.cat1_file),
        cat2_file=Path(args.cat2_file),
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    if not args.skip_generation:
        run_generation(
            records=records,
            generator_model=args.generator_model,
            generation_file=generation_file,
            max_model_len=args.gen_max_model_len,
            gpu_memory_utilization=args.gen_gpu_memory_utilization,
            max_new_tokens=args.gen_max_new_tokens,
        )

    if not args.skip_judge:
        run_judge(
            generation_file=generation_file,
            judge_file=judge_file,
            judge_model=args.judge_model,
            max_model_len=args.judge_max_model_len,
            gpu_memory_utilization=args.judge_gpu_memory_utilization,
            max_new_tokens=args.judge_max_new_tokens,
        )

    summary = summarize(judge_file)
    comparison = make_pairwise_comparison(summary)

    write_json(summary_file, summary)
    write_json(comparison_file, comparison)

    latex_rows = make_latex_rows(summary)
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_rows + "\n")

    print_summary(summary)
    print_comparison(comparison)

    print("\nSaved:")
    print(f"  Paired generations: {generation_file}")
    print(f"  Paired judge file:  {judge_file}")
    print(f"  Summary:            {summary_file}")
    print(f"  Comparison:         {comparison_file}")
    print(f"  LaTeX rows:         {latex_file}")


if __name__ == "__main__":
    main()