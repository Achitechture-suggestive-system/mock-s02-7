# Exact references used by the prototype

Mỗi dòng dưới đây ghi phần của nguồn được dùng và phần nào **không** được
suy ra. Link ưu tiên là publisher, ACL Anthology, NeurIPS hoặc author-hosted
paper. Không dùng citation này để claim benchmark của repository.

| Source | Exact part used | Repository use / non-claim |
|---|---|---|
| Patrick Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) · [Abstract](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | §2 Methods, §§2.1–2.3 | Conceptual separation between external retrieved memory and generation. Does not validate architecture topology. |
| Stephen Robertson & Hugo Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” Foundations and Trends in Information Retrieval 4(1–2), 2009. [DOI](https://doi.org/10.1561/1500000019) | §3.4 BM25; §3.6 multiple streams/BM25F; Eqs. 3.19–3.21 | Mathematical reference for field normalization, weighted pseudo-TF and saturation. Project defaults/fields are adaptations, not paper results. |
| Gordon V. Cormack, Charles L. A. Clarke & Stefan Büttcher, “Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods,” SIGIR 2009. [Paper PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) · [DOI](https://doi.org/10.1145/1571941.1572114) | §1 and RRF formula; experimental `k=60` | Reference for rank fusion. Code’s configurable `k`, deterministic tie-break and evidence-unit scope are engineering choices. |
| Xiaohua Chen et al., “M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation,” Findings of ACL 2024. [ACL Anthology](https://aclanthology.org/2024.findings-acl.137/) | Abstract/metadata | Motivates optional multilingual embedding adapter. Does not prove `BAAI/bge-m3` is optimal for this KB. |
| Marco Petcu et al., “Query Decomposition for RAG: Balancing Exploration-Exploitation,” EACL 2026, pp. 6857–6871. [ACL Anthology](https://aclanthology.org/2026.eacl-long.322/) · [PDF](https://aclanthology.org/2026.eacl-long.322.pdf) | §3.2, §3.3, Algorithm 1, §5.1 | Reference for research-only bandit allocation adapter. Fixture labels are not production relevance ground truth; current adapter is not the paper’s full evaluation. |
| Julian Scholak, Richard Schucher & Dzmitry Bahdanau, “PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models,” EMNLP 2021. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.779/) | Abstract and §2 | Analogy for formal output constraints. Repository does not implement token-level incremental parsing. |
| Akari Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,” ICLR 2024. [arXiv abstract](https://arxiv.org/abs/2310.11511) | Abstract | Related direction only. Repository does not train/use reflection tokens or a Self-RAG critic. |

## Documentation/runtime references

- [Ollama API `format` and generation options](https://github.com/ollama/ollama/blob/main/docs/api.md): source for the API contract used by the Ollama adapter.
- [Ollama Qwen3 model page](https://ollama.com/library/qwen3): model/package reference; it is not a paper result for this repository.
- ARCH-KB-PUML profile and vocabulary: `C:\disk D\KnowledgeBase_SoftwareArchitect`; local KB artifacts are the authoritative source for profile/version and evidence content in a run.

## Removed/avoided citation

An earlier draft cited “The Hidden Cost of Structure: Constrained Decoding in
LLMs” as `Kovatchev et al., RANLP 2025` with the identifier
`2025.ranlp-1.124`. That identifier does not support the stated bibliographic
claim, so the citation is intentionally removed. The README now describes the
implementation boundary directly: JSON Schema plus post-generation validation
is not PICARD-style token-level constrained decoding.
