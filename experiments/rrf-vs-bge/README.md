# RRF versus BGE reranking benchmark

This folder contains the reproducible 120-input comparison between:

- semantic embedding + lexical BM25F rank fusion with RRF;
- the same top-50 RRF candidate pool reranked by `BAAI/bge-reranker-v2-m3`.

The benchmark uses 60 labelled evidence targets from the production knowledge
base. Each target has one source-vocabulary query and one paraphrase query.
Ollama generation is intentionally excluded so the result measures retrieval
and reranking only.

## Run

From the repository root, with the optional `sentence-transformers`
dependencies installed:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 .\experiments\rrf-vs-bge\evaluate.py
```

The script writes the complete result to `benchmark-result.json` in this
folder. Use `--output` to choose another path.

## Result summary

| Metric | RRF | RRF + BGE |
|---|---:|---:|
| Candidate recall@50 | 100.00% | 100.00% |
| Hit@1 | 81.67% | 82.50% |
| Hit@3 | 92.50% | 94.17% |
| Hit@10 | 96.67% | 100.00% |
| MRR@10 | 0.8753 | 0.8894 |

BGE improved 16 inputs, harmed 12, and left 92 unchanged. It helped more on
paraphrases; RRF was slightly safer on queries sharing source vocabulary.
Scores are ranking signals, not relevance probabilities.

## BGE versus MMARCO

The paired comparison uses the same 120 inputs, embedding model, BM25F, RRF
`k=60`, and top-50 candidate pool:

| Metric | BGE-reranker-v2-m3 | MMARCO mMiniLMv2 |
|---|---:|---:|
| Hit@1 | 82.50% | 79.17% |
| Hit@3 | 94.17% | 95.00% |
| Hit@5 | 99.17% | 97.50% |
| Hit@10 | 100.00% | 100.00% |
| MRR@10 | 0.8894 | 0.8767 |
| Mean target rank | 1.383 | 1.400 |

BGE wins the target rank on 15 inputs, MMARCO wins 11, and 94 are tied. BGE
is better on paraphrases (MRR@10 `0.7955` versus `0.7617`), while MMARCO is
slightly better on source-vocabulary queries (MRR@10 `0.9917` versus `0.9833`).
The full paired output is `comparison-bge-vs-mmarco.json`.

Recommendation for this multilingual architecture-evidence pipeline: keep BGE
as the quality-first default because it wins overall MRR and paraphrase
handling. Use MMARCO when CPU latency, memory, or throughput is more important;
the model card lists about `0.1B` parameters for MMARCO versus about `0.6B` for
BGE, and the observed 120-input run was substantially faster for MMARCO on this
machine. This is a local fixture decision, not a universal model ranking.
