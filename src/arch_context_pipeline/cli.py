from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    run_pipeline,
)


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
    parser.add_argument(
        "--semantic-model",
        default=DEFAULT_SEMANTIC_MODEL,
        help=f"Dense embedding model for the default BM25F + semantic RRF path (default: {DEFAULT_SEMANTIC_MODEL})",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable dense semantic retrieval and leave RRF with the lexical rank list only",
    )
    parser.add_argument(
        "--semantic-trust-remote-code",
        action="store_true",
        help="Allow a trusted embedding model with custom Hugging Face modeling code",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-encoder reranker for the RRF candidate pool (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranking and return the RRF order",
    )
    parser.add_argument(
        "--reranker-trust-remote-code",
        action="store_true",
        help="Allow trusted Hugging Face rerankers with custom model code (required by GTE multilingual)",
    )
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
        None if args.no_semantic else args.semantic_model,
        None if args.no_reranker else args.reranker_model,
        args.reranker_trust_remote_code,
        args.semantic_trust_remote_code,
    )
    print(f"Generated bundle: {result['output_dir']}")
    cases = result["retrieval"].get("retrieved_cases", [])
    if cases:
        top = cases[0]
        basis = result["retrieval"]["method"]["final_ranking"]["basis"]
        score_key = "best_reranker_score" if basis == "reranker" else "best_rrf_score"
        print(f"Top retrieval: {top['case_id']} ({basis} {top[score_key]:.6f})")
    else:
        print("Top retrieval: none")
    print(f"Validation: {result['validation']['overall_status']}")
    print(f"Handoff gate: {result['gate']['gate_result']}")
