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
