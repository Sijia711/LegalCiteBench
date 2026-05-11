# LegalCiteBench

This repository contains construction and evaluation code for LegalCiteBench, a benchmark for evaluating legal citation reliability in large language models.

## Repository Structure

- data/: scripts for constructing benchmark categories.
- eval/: scripts for model generation and LLM-as-judge evaluation.
- analysis/: scripts for result analysis.
- configs/: configuration files.
- .env.example: example environment variables for OpenAI-compatible API access.

## Environment

Install basic dependencies:

pip install openai python-dotenv tqdm

For open-source model inference and local judging, install vLLM following the official vLLM instructions.

## API Configuration

Create a local .env file with:

OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE_URL=your_api_base_url_here

Do not commit .env or private credentials.

## Dataset Construction

The benchmark is constructed in stages:

1. Generate structured root-case summaries.
2. Construct Cat1 citation retrieval questions.
3. Construct Cat2 citation completion questions.
4. Construct Cat3 citation error detection examples.
5. Construct Cat4-1 case matching examples.
6. Construct Cat4-2 case verification and correction examples.

## Evaluation

Model generation and judging are separated into two checkpointed stages. Outputs are stored in JSONL format.
