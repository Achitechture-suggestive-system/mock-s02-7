from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evidence-first architecture context pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="JSON stakeholder input")
    parser.add_argument("--kb", type=Path, required=True, help="Architecture Knowledge Base root")
    parser.add_argument("--out", type=Path, required=True, help="Output bundle directory")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence units to retrieve per normalized query")
    parser.add_argument("--generator", choices=["mock", "ollama"], default="mock", help="IR generator")
    parser.add_argument("--model", default="qwen3:4b", help="Local Ollama model, used with --generator ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--llm-timeout", type=int, default=300, help="LLM request timeout in seconds")
    parser.add_argument("--semantic-model", help="Optional SentenceTransformers model, e.g. BAAI/bge-m3")
    parser.add_argument("--reranker-model", help="Optional SentenceTransformers cross-encoder model")
    parser.add_argument("--accept-mock-review", action="store_true", help="Demonstrate the gate only; never treat this as real human acceptance")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        args.input,
        args.kb,
        args.out,
        args.top_k,
        args.accept_mock_review,
        args.generator,
        args.model,
        args.ollama_url,
        args.llm_timeout,
        args.semantic_model,
        args.reranker_model,
    )
    print(f"Generated bundle: {result['output_dir']}")
    cases = result["retrieval"].get("retrieved_cases", [])
    if cases:
        top = cases[0]
        print(f"Top retrieval: {top['case_id']} (RRF {top['best_rrf_score']:.6f})")
    else:
        print("Top retrieval: none")
    print(f"Validation: {result['validation']['overall_status']}")
    print(f"Handoff gate: {result['gate']['gate_result']}")
