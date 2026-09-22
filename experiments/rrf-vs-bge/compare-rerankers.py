"""Compare persisted BGE and MMARCO results on the same labelled inputs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BGE_PATH = ROOT / "benchmark-result.json"
MMARCO_PATH = ROOT / "mmarco-result.json"
OUTPUT_PATH = ROOT / "comparison-bge-vs-mmarco.json"


def reciprocal(rank: int | None, cutoff: int) -> float:
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def metrics(rows: list[dict[str, object]], rank_key: str, name: str) -> dict[str, object]:
    ranks = [int(row[rank_key]) for row in rows if row[rank_key] is not None]
    return {
        "name": name,
        "queries": len(rows),
        "candidate_recall_at_50": round(sum(row["candidate_rank"] is not None for row in rows) / len(rows), 6),
        "hit_rate": {
            f"@{cutoff}": round(sum(row[rank_key] is not None and int(row[rank_key]) <= cutoff for row in rows) / len(rows), 6)
            for cutoff in (1, 3, 5, 10, 50)
        },
        "mrr": {
            f"@{cutoff}": round(sum(reciprocal(int(row[rank_key]) if row[rank_key] is not None else None, cutoff) for row in rows) / len(rows), 6)
            for cutoff in (1, 3, 5, 10, 50)
        },
        "mean_rank_when_found_at_50": round(statistics.mean(ranks), 6),
        "median_rank_when_found_at_50": statistics.median(ranks),
    }


def subset_metrics(rows: list[dict[str, object]], variant: str) -> dict[str, object]:
    subset = [row for row in rows if row["variant"] == variant]
    return {
        "inputs": len(subset),
        "bge": metrics(subset, "bge_rank", "BGE"),
        "mmarco": metrics(subset, "mmarco_rank", "MMARCO"),
    }


def main() -> None:
    bge = json.loads(BGE_PATH.read_text(encoding="utf-8"))
    mmarco = json.loads(MMARCO_PATH.read_text(encoding="utf-8"))
    bge_rows = {row["input_id"]: row for row in bge["rows"]}
    mmarco_rows = {row["input_id"]: row for row in mmarco["rows"]}
    if set(bge_rows) != set(mmarco_rows):
        raise RuntimeError("BGE and MMARCO result sets do not contain the same input IDs")

    rows: list[dict[str, object]] = []
    for input_id in sorted(bge_rows):
        left = bge_rows[input_id]
        right = mmarco_rows[input_id]
        rows.append({
            "input_id": input_id,
            "domain": left["domain"],
            "variant": left["variant"],
            "query": left["query"],
            "gold_evidence_id": left["gold_evidence_id"],
            "candidate_rank": left["candidate_rank"],
            "bge_rank": left["bge_rank"],
            "mmarco_rank": right["mmarco_rank"],
            "bge_top1": left["bge_top1"],
            "mmarco_top1": right["mmarco_top1"],
            "bge_top10": left["bge_top10"],
            "mmarco_top10": right["mmarco_top10"],
            "bge_minus_mmarco_rank": int(left["bge_rank"]) - int(right["mmarco_rank"]),
        })

    bge_wins = [row for row in rows if row["bge_rank"] < row["mmarco_rank"]]
    mmarco_wins = [row for row in rows if row["mmarco_rank"] < row["bge_rank"]]
    ties = [row for row in rows if row["bge_rank"] == row["mmarco_rank"]]
    comparison = {
        "bge_wins": len(bge_wins),
        "mmarco_wins": len(mmarco_wins),
        "ties": len(ties),
        "top1_gold": {
            "bge": sum(row["bge_top1"] == row["gold_evidence_id"] for row in rows),
            "mmarco": sum(row["mmarco_top1"] == row["gold_evidence_id"] for row in rows),
        },
        "top1_agreement_between_models": round(sum(row["bge_top1"] == row["mmarco_top1"] for row in rows) / len(rows), 6),
        "mean_top3_overlap": round(sum(len(set(row["bge_top10"][:3]) & set(row["mmarco_top10"][:3])) / 3 for row in rows) / len(rows), 6),
        "mean_top10_overlap": round(sum(len(set(row["bge_top10"]) & set(row["mmarco_top10"])) / 10 for row in rows) / len(rows), 6),
        "mean_bge_minus_mmarco_rank": round(statistics.mean(row["bge_minus_mmarco_rank"] for row in rows), 6),
    }
    output = {
        "benchmark": {
            "name": "BGE-reranker-v2-m3 versus MMARCO paired comparison",
            "inputs": len(rows),
            "same_first_stage": True,
            "shared_embedding_model": bge["benchmark"]["shared_embedding_model"],
            "candidate_pool_size": bge["benchmark"]["candidate_pool_size"],
            "rrf_k": bge["benchmark"]["rrf_k"],
            "bge_model": bge["benchmark"]["bge_model"],
            "mmarco_model": mmarco["benchmark"]["reranker_model"],
        },
        "aggregate": {
            "bge": metrics(rows, "bge_rank", "BGE-reranker-v2-m3"),
            "mmarco": metrics(rows, "mmarco_rank", "MMARCO mMiniLMv2"),
            "comparison": comparison,
            "by_variant": {
                "source_vocab": subset_metrics(rows, "source_vocab"),
                "paraphrase": subset_metrics(rows, "paraphrase"),
            },
        },
        "rows": rows,
        "bge_wins": bge_wins,
        "mmarco_wins": mmarco_wins,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
