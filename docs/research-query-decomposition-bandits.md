# Adaptive query-decomposition retrieval backbone

This document adapts only the budget-allocation idea from Petcu et al.,
**Query Decomposition for RAG: Balancing Exploration-Exploitation**, EACL 2026,
to the evidence-first architecture pipeline in this repository. It does not
claim that the repository reproduces the paper's complete experimental setup.

Paper: <https://aclanthology.org/2026.eacl-long.322/>
Preprint: <https://arxiv.org/abs/2510.18633>

## What the paper contributes

The paper treats each decomposed sub-query as a multi-armed-bandit arm. Each
arm has a ranked document list. Pulling an arm retrieves the next document on
that list, observes a document-level relevance reward, updates that arm's
posterior, and allocates the next pull between exploitation and exploration.
The discrete algorithm uses Thompson sampling with Beta priors. The paper
also evaluates rank-aware top-k rewards, UCB exploration, diversity, and
correlated bandits for hierarchical decompositions. These claims refer to the
paper's §3.2–§3.3, Algorithm 1 and §5.1, not to this repository's demo result.

The paper's central assumption is important: relevance labels, or another
meaningful reward signal, are available, and the initial ranked list is a
reasonable estimate of relevance. If the ranker is poor, its noise propagates
into the bandit reward. This repository currently has no production relevance
judgment set, so the included clinic labels are explicitly illustrative.

## Mapping to this repository

| Paper concept | Repository mapping | Boundary |
| --- | --- | --- |
| Sub-query / arm | `CONTEXT-001`, `FR-*`, `NFR-*`, `C-*` from `build_retrieval_queries()` | Deterministic requirement decomposition is an engineering adaptation. |
| Ranked documents for an arm | `retrieval.queries[*].evidence` from BM25F/RRF | Initial ranking still depends on current lexical/semantic providers. |
| Pull one document | Advance the arm's `next_index` by one | Evidence is not removed from the source KB. |
| Bernoulli relevance | Caller-provided `0/1` labels | Demo labels are not ground truth. |
| Beta posterior | `alpha`, `beta`, posterior mean in `adaptive-selection.json` | Posterior estimates arm utility under the supplied labels. |
| Retrieval budget | `AdaptiveBanditConfig.budget` | Budget counts pulls, not unique documents. |
| Hierarchical bandit | Not enabled in this first backbone | Requires explicit parent-child query decomposition and inheritance. |
| Diversity reward | Not enabled | Current output does not provide calibrated evidence embeddings. |

## Equations implemented

For arm `i`, the discrete Thompson backbone starts with:

```text
alpha_i = 1
beta_i = 1
theta_i ~ Beta(alpha_i, beta_i)
arm_t = argmax_i theta_i
```

The next evidence unit on the selected arm is observed. For binary reward
`r_t`:

```text
alpha_i <- alpha_i + 1  if r_t = 1
beta_i  <- beta_i + 1   if r_t = 0
```

The posterior mean is:

```text
E[p_i | data] = alpha_i / (alpha_i + beta_i)
```

The optional `top_k_ucb` adapter estimates the next local window:

```text
window_mean_i = (1 / K) * sum_{j=n}^{n+K-1} relevance(i,j)

ucb_i = c * sqrt(2 * log2(T + 1) / pulls_i)

policy_value_i = window_mean_i + ucb_i
```

Unpulled arms receive infinite UCB in this adapter, guaranteeing exploration.
The implementation is intentionally tagged as a research adapter because the
paper's full strongest policy also includes diversity and uses the real
relevance setup.

## Exact runnable trial

The trial uses the existing clinic fixture and manually written illustrative
labels:

```powershell
cd 'C:\disk D\mock-s02-7'
$env:PYTHONPATH = 'src'
python -m arch_context_pipeline.adaptive_retrieval `
  --input examples/clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out out/clinic-local-qwen3 `
  --labels examples/clinic_bandit_judgments.json `
  --retrieval-top-k 20 `
  --budget 12 `
  --policy thompson_bernoulli `
  --seed 17
```

The output artifact is:

```text
out/clinic-local-qwen3/adaptive-selection.json
```

The command also prints a short decision trace to stdout, for example:

```text
step=1 decision=warm_start query=C-001 rank=1 evidence=case-000006:E001 reward=0.0
step=2 decision=warm_start query=CONTEXT-001 rank=1 evidence=case-000006:E001 reward=0.0
step=3 decision=warm_start query=FR-001 rank=1 evidence=case-000006:E001 reward=0.0
...
```

The exact sequence is deterministic for `seed=17`; the JSON audit artifact
preserves the sampled theta, prior, posterior, candidate rank, label status,
and evidence provenance. It is an adaptive-selection trace, not a general
application runtime log. The initial warm-start pulls are an engineering
safeguard to observe every query arm once; they are not a claim that the paper
requires a warm-start phase.

## Interpretation for the clinic fixture

The trial demonstrates budget allocation, not improved architecture quality.
The current KB is mostly English while the clinic request is Vietnamese, and
the semantic provider is not configured. Therefore an adaptive policy cannot
recover evidence that never entered an adequate ranked list. A future valid
evaluation needs:

1. a multilingual semantic retriever or translated retrieval projection;
2. a judgment set with graded or binary relevance per query/evidence unit;
3. a fixed retrieval budget and baselines: fixed top-k, round-robin,
   Thompson, top-k UCB, and diversity-aware selection;
4. retrieval metrics such as precision@k, recall@k, nDCG/alpha-nDCG, unique
   evidence coverage, and context-token cost;
5. downstream checks that every architecture claim is supported by its source
   evidence, without treating bandit posterior values as correctness scores.

The adaptive layer is therefore a budget allocator above the existing
evidence-first retriever, not a replacement for evidence validation or
architecture IR validation. `adaptive-selection.json` must be interpreted
together with `retrieval-evidence.json`; its posterior/policy scores are not
relevance probabilities for an unseen case.
