"""Run the same labelled fixture with the historical MMARCO reranker.

The BGE result is already persisted in benchmark-result.json. This script
keeps the embedding model, BM25F, RRF and candidate pool identical, and only
changes the cross-encoder to MMARCO so the two rerankers can be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate import (  # noqa: E402
    CASES,
    CachedEmbeddingProvider,
    EMBEDDING_MODEL,
    make_input,
    rank_of,
    summarize,
)
from arch_context_pipeline.pipeline import (  # noqa: E402
    RetrievalConfig,
    SentenceTransformerCrossEncoderReranker,
    SentenceTransformerEmbeddingProvider,
    retrieve,
    write_json,
)


MMARCO_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-root", type=Path, default=Path(r"C:\disk D\KnowledgeBase_SoftwareArchitect"))
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "mmarco-result.json")
    args = parser.parse_args()

    if len(CASES) != 60:
        raise RuntimeError(f"Expected 60 benchmark targets, got {len(CASES)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}", flush=True)
    embedding = CachedEmbeddingProvider(SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL, batch_size=32))
    print(f"Loading reranker model: {MMARCO_MODEL}", flush=True)
    reranker = SentenceTransformerCrossEncoderReranker(MMARCO_MODEL, batch_size=16)
    config = RetrievalConfig(rrf_k=60, candidate_multiplier=5)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, (target, domain, source_query, paraphrase) in enumerate(CASES, 1):
        for variant, query_text in (("source_vocab", source_query), ("paraphrase", paraphrase)):
            query_id = f"EVAL-{index:03d}-{variant}"
            result = retrieve(
                args.kb_root,
                make_input(query_id, query_text),
                top_k=10,
                config=config,
                semantic_provider=embedding,
                reranker=reranker,
            )
            query = next(item for item in result["queries"] if item["query_id"] == query_id)
            candidate_pool = query["candidate_pool"]
            top10 = [row["evidence_id"] for row in query["evidence"]]
            rows.append({
                "input_id": query_id,
                "domain": domain,
                "variant": variant,
                "query": query_text,
                "gold_evidence_id": target,
                "candidate_rank": rank_of(candidate_pool, target, "rrf_rank"),
                "rrf_rank": rank_of(candidate_pool, target, "rrf_rank"),
                "mmarco_rank": rank_of(candidate_pool, target, "reranker_rank"),
                "rrf_top10": [row["evidence_id"] for row in sorted(candidate_pool, key=lambda row: row["rrf_rank"])[:10]],
                "mmarco_top10": top10,
                "rrf_top1": sorted(candidate_pool, key=lambda row: row["rrf_rank"])[0]["evidence_id"],
                "mmarco_top1": top10[0] if top10 else None,
                "mmarco_rank_delta_vs_rrf": (
                    rank_of(candidate_pool, target, "reranker_rank") - rank_of(candidate_pool, target, "rrf_rank")
                    if rank_of(candidate_pool, target, "reranker_rank") is not None
                    else None
                ),
                "support_status": query["support_status"],
                "mmarco_target_row": next((row for row in candidate_pool if row["evidence_id"] == target), None),
            })
        print(f"[{index:02d}/{len(CASES)}] completed {domain}; elapsed={time.perf_counter() - started:.1f}s", flush=True)

    output = {
        "benchmark": {
            "name": "MMARCO reranker on the RRF versus BGE labelled fixture",
            "scope": "retrieval and reranking only; no Ollama generation",
            "inputs": len(rows),
            "gold_labels": "one target evidence unit per input, sourced from the production KB",
            "candidate_pool_size": 50,
            "top_k": 10,
            "rrf_k": 60,
            "shared_embedding_model": EMBEDDING_MODEL,
            "reranker_model": MMARCO_MODEL,
        },
        "aggregate": {
            "mmarco": summarize(rows, "semantic + lexical RRF + MMARCO cross-encoder", "mmarco_rank"),
            "candidate_pool_recall_at_50": round(sum(row["candidate_rank"] is not None for row in rows) / len(rows), 6),
            "top1_agreement_with_rrf": round(sum(row["rrf_top1"] == row["mmarco_top1"] for row in rows) / len(rows), 6),
            "mean_top3_overlap_with_rrf": round(
                sum(len(set(row["rrf_top10"][:3]) & set(row["mmarco_top10"][:3])) / 3 for row in rows) / len(rows),
                6,
            ),
            "rank_comparison_vs_rrf": {
                "improved": sum(row["mmarco_rank"] < row["rrf_rank"] for row in rows),
                "harmed": sum(row["mmarco_rank"] > row["rrf_rank"] for row in rows),
                "unchanged": sum(row["mmarco_rank"] == row["rrf_rank"] for row in rows),
            },
        },
        "rows": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
