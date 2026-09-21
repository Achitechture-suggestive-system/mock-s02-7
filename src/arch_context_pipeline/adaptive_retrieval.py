"""Adaptive query-decomposition retrieval based on Petcu et al. (EACL 2026).

The existing pipeline produces one ranked evidence list per normalized
requirement. This module adds the paper's decision layer: each requirement is
an arm, each pull observes the next evidence unit on that arm's ranked list,
and a Bernoulli posterior controls the next pull under a fixed budget.

The module deliberately keeps relevance labels outside the retriever. Labels
may come from human judgments, an explicitly configured judge, clicks, or a
research-only fixture. Unlisted labels are only treated as zero when the
caller explicitly accepts the ``unjudged_as_zero`` policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pipeline import normalize_input, read_json, retrieve


@dataclass(frozen=True)
class AdaptiveBanditConfig:
    """Configuration for the adaptive evidence selector.

    ``thompson_bernoulli`` follows the paper's discrete-space Thompson
    sampling backbone. ``top_k_ucb`` is a deterministic research adapter for
    the paper's rank-aware top-k reward family; it is useful for offline
    experiments but does not replace a learned relevance model.
    """

    budget: int = 12
    policy: str = "thompson_bernoulli"
    seed: int = 17
    warm_start: bool = True
    top_k_window: int = 3
    exploration_constant: float = 0.25
    unjudged_policy: str = "unjudged_as_zero"

    def validate(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.policy not in {"thompson_bernoulli", "top_k_ucb"}:
            raise ValueError("policy must be thompson_bernoulli or top_k_ucb")
        if self.top_k_window < 1:
            raise ValueError("top_k_window must be positive")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must not be negative")
        if self.unjudged_policy not in {"unjudged_as_zero", "error"}:
            raise ValueError("unsupported unjudged_policy")


def _ranked_arms(retrieval: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert retrieval output into one ranked list per query arm."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for query in retrieval.get("queries", []):
        query_id = str(query["query_id"])
        if query_id in arms:
            raise ValueError(f"duplicate query arm: {query_id}")
        candidates = list(query.get("evidence", []))
        candidates.sort(key=lambda item: (int(item["rank"]), str(item["evidence_id"])))
        arms[query_id] = candidates
    if not arms:
        raise ValueError("retrieval output has no query arms")
    return arms


def _relevance(
    relevance_labels: Mapping[str, Mapping[str, float]],
    query_id: str,
    evidence_id: str,
    config: AdaptiveBanditConfig,
) -> tuple[float, str]:
    labels = relevance_labels.get(query_id, {})
    if evidence_id not in labels:
        if config.unjudged_policy == "error":
            raise ValueError(f"missing relevance label for {query_id}/{evidence_id}")
        return 0.0, "unjudged_as_zero"
    value = float(labels[evidence_id])
    if value not in {0.0, 1.0}:
        raise ValueError(f"Bernoulli relevance must be 0 or 1: {query_id}/{evidence_id}")
    return value, "judged"


def _select_thompson_arm(
    state: Mapping[str, Mapping[str, Any]],
    available: Sequence[str],
    rng: random.Random,
) -> tuple[str, dict[str, float]]:
    samples = {
        query_id: rng.betavariate(float(state[query_id]["alpha"]), float(state[query_id]["beta"]))
        for query_id in available
    }
    selected = max(available, key=lambda query_id: (samples[query_id], query_id))
    return selected, samples


def _select_top_k_ucb_arm(
    state: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    relevance_labels: Mapping[str, Mapping[str, float]],
    available: Sequence[str],
    config: AdaptiveBanditConfig,
) -> tuple[str, dict[str, float], dict[str, float]]:
    """Select an arm using a rank-aware top-k reward plus UCB exploration.

    This is an offline adapter: it estimates the next window using the
    supplied relevance labels. The paper's reported best policy also includes
    a diversity factor; this first backbone intentionally records diversity as
    not applied because the current retrieval output does not expose a
    calibrated embedding for every evidence unit.
    """
    total_pulls = max(1, sum(int(item["pulls"]) for item in state.values()))
    values: dict[str, float] = {}
    window_means: dict[str, float] = {}
    for query_id in available:
        start = int(state[query_id]["next_index"])
        window = list(arms[query_id][start : start + config.top_k_window])
        rewards = [
            _relevance(relevance_labels, query_id, str(item["evidence_id"]), config)[0]
            for item in window
        ]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        pulls = int(state[query_id]["pulls"])
        if pulls == 0:
            ucb = float("inf")
        else:
            ucb = config.exploration_constant * math.sqrt(math.log2(total_pulls + 1) / pulls)
        window_means[query_id] = mean
        values[query_id] = mean + ucb
    selected = max(available, key=lambda query_id: (values[query_id], query_id))
    return selected, values, window_means


def select_adaptive_evidence(
    retrieval: Mapping[str, Any],
    relevance_labels: Mapping[str, Mapping[str, float]] | None = None,
    *,
    config: AdaptiveBanditConfig | None = None,
) -> dict[str, Any]:
    """Select evidence adaptively from per-query ranked retrieval lists.

    The output is a reproducible research trace. It does not alter the
    lexical/semantic scores emitted by :func:`pipeline.retrieve`.
    """
    config = config or AdaptiveBanditConfig()
    config.validate()
    relevance_labels = relevance_labels or {}
    arms = _ranked_arms(retrieval)
    state: dict[str, dict[str, Any]] = {
        query_id: {
            "next_index": 0,
            "pulls": 0,
            "alpha": 1.0,
            "beta": 1.0,
            "observed_evidence_ids": [],
        }
        for query_id in arms
    }
    rng = random.Random(config.seed)
    trace: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    warmed_arms: set[str] = set()

    for step in range(1, config.budget + 1):
        available = [
            query_id for query_id in sorted(arms)
            if int(state[query_id]["next_index"]) < len(arms[query_id])
        ]
        if not available:
            break

        if config.warm_start and len(warmed_arms) < len(arms):
            query_id = next(item for item in available if item not in warmed_arms)
            policy_scores = {item: None for item in available}
            window_means: dict[str, float] = {}
            samples: dict[str, float] = {}
            warmed_arms.add(query_id)
            decision = "warm_start"
        elif config.policy == "thompson_bernoulli":
            query_id, samples = _select_thompson_arm(state, available, rng)
            policy_scores = samples
            window_means = {}
            decision = "thompson_sample"
        else:
            query_id, policy_scores, window_means = _select_top_k_ucb_arm(
                state, arms, relevance_labels, available, config
            )
            samples = {}
            decision = "top_k_ucb"

        arm_state = state[query_id]
        rank_index = int(arm_state["next_index"])
        candidate = dict(arms[query_id][rank_index])
        evidence_id = str(candidate["evidence_id"])
        observed_reward, label_status = _relevance(relevance_labels, query_id, evidence_id, config)
        prior_alpha = float(arm_state["alpha"])
        prior_beta = float(arm_state["beta"])
        arm_state["next_index"] = rank_index + 1
        arm_state["pulls"] = int(arm_state["pulls"]) + 1
        if observed_reward >= 0.5:
            arm_state["alpha"] = prior_alpha + 1.0
        else:
            arm_state["beta"] = prior_beta + 1.0
        arm_state["observed_evidence_ids"].append(evidence_id)

        selected.append({
            "selection_rank": step,
            "query_id": query_id,
            "retrieval_rank": int(candidate["rank"]),
            "evidence_id": evidence_id,
            "case_id": candidate.get("case_id"),
            "evidence_type": candidate.get("evidence_type"),
            "text": candidate.get("text"),
            "source_path": candidate.get("source_path"),
            "source_locator": candidate.get("source_locator"),
            "observed_relevance": observed_reward,
            "label_status": label_status,
        })
        trace.append({
            "step": step,
            "decision": decision,
            "query_id": query_id,
            "candidate_rank": int(candidate["rank"]),
            "evidence_id": evidence_id,
            "prior": {"alpha": prior_alpha, "beta": prior_beta},
            "posterior": {"alpha": arm_state["alpha"], "beta": arm_state["beta"]},
            "sampled_theta": samples.get(query_id),
            "policy_score": policy_scores.get(query_id),
            "top_k_window_mean": window_means.get(query_id),
            "observed_relevance": observed_reward,
            "label_status": label_status,
            "available_arm_count": len(available),
        })

    unique_selected: dict[str, dict[str, Any]] = {}
    for item in selected:
        evidence_id = str(item["evidence_id"])
        if evidence_id not in unique_selected:
            unique_selected[evidence_id] = dict(item)
            unique_selected[evidence_id]["selected_by_queries"] = [item["query_id"]]
        elif item["query_id"] not in unique_selected[evidence_id]["selected_by_queries"]:
            unique_selected[evidence_id]["selected_by_queries"].append(item["query_id"])

    if config.policy == "thompson_bernoulli":
        method_name = "thompson_sampling_bernoulli"
        paper_claim = "Paper Algorithm 1 backbone: Beta(1,1) posterior per query arm and one next document per pull."
    else:
        method_name = "bernoulli_top_k_ucb_research_adapter"
        paper_claim = "Research adapter: rank-aware top-k relevance window plus UCB exploration; diversity is not applied."

    return {
        "method": {
            "name": method_name,
            "paper": "Petcu et al., Query Decomposition for RAG: Balancing Exploration-Exploitation, EACL 2026",
            "paper_claim": paper_claim,
            "relevance_source": "caller-provided labels; this run may use an illustrative fixture",
            "diversity_applied": False,
            "score_interpretation": "posterior or policy score for evidence allocation; not relevance probability for an unseen case",
        },
        "config": asdict(config),
        "arm_count": len(arms),
        "arm_candidate_counts": {query_id: len(items) for query_id, items in arms.items()},
        "selected_pull_count": len(trace),
        "unique_selected_evidence_count": len(unique_selected),
        "selected_evidence": list(unique_selected.values()),
        "trace": trace,
        "final_posteriors": {
            query_id: {
                "alpha": state[query_id]["alpha"],
                "beta": state[query_id]["beta"],
                "posterior_mean": state[query_id]["alpha"] / (state[query_id]["alpha"] + state[query_id]["beta"]),
                "pulls": state[query_id]["pulls"],
                "next_rank": state[query_id]["next_index"] + 1,
            }
            for query_id in sorted(state)
        },
        "claim_boundary": (
            "Adaptive selection allocates a fixed retrieval budget across query arms. "
            "It does not prove that an evidence unit is relevant or that a generated architecture is correct."
        ),
    }


def adaptive_retrieve(
    kb_root: Path,
    requirements: dict[str, Any],
    *,
    retrieval_top_k: int = 20,
    relevance_labels: Mapping[str, Mapping[str, float]] | None = None,
    config: AdaptiveBanditConfig | None = None,
    semantic_provider: Any | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    """Run current retrieval first, then the adaptive bandit selection layer."""
    retrieval = retrieve(
        kb_root,
        requirements,
        top_k=retrieval_top_k,
        semantic_provider=semantic_provider,
        reranker=reranker,
    )
    return {
        "retrieval": retrieval,
        "adaptive_selection": select_adaptive_evidence(
            retrieval,
            relevance_labels,
            config=config,
        ),
    }


def _load_labels(path: Path | None) -> tuple[dict[str, dict[str, float]], str]:
    if path is None:
        return {}, "no labels supplied; unjudged items treated as zero only for demo"
    payload = read_json(path)
    return payload.get("queries", {}), str(payload.get("label_source", path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive query-decomposition retrieval.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--policy", choices=["thompson_bernoulli", "top_k_ucb"], default="thompson_bernoulli")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    raw = read_json(args.input)
    requirements = normalize_input(raw)
    labels, label_source = _load_labels(args.labels)
    config = AdaptiveBanditConfig(budget=args.budget, policy=args.policy, seed=args.seed)
    result = adaptive_retrieve(
        args.kb,
        requirements,
        retrieval_top_k=args.retrieval_top_k,
        relevance_labels=labels,
        config=config,
    )
    result["adaptive_selection"]["method"]["label_source"] = label_source
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "adaptive-selection.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated adaptive bundle: {output_path}")
    for item in result["adaptive_selection"]["trace"]:
        print(
            "step={step} decision={decision} query={query_id} rank={candidate_rank} "
            "evidence={evidence_id} reward={observed_relevance} posterior={posterior}".format(**item)
        )


if __name__ == "__main__":
    main()
