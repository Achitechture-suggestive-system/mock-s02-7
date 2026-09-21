# Research boundary và citation ledger

Tài liệu này chỉ ghi những nguồn có liên hệ trực tiếp với code hiện tại và
phân biệt rõ ba lớp: điều paper định nghĩa, điều repository đã implement, và
điều repository chưa chứng minh. Không dùng citation trong file này để tuyên
bố kiến trúc sinh ra là đúng production.

## 1. Retrieval-Augmented Generation

Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks*, NeurIPS 2020, [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf),
[abstract](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).

Phần được dùng: §2 Methods, đặc biệt §2.1–§2.3, để phân biệt parametric
memory của generator với non-parametric memory bên ngoài và mô tả retriever →
generator. Đây là nền tảng khái niệm cho việc truyền evidence từ KB vào
context.

Giới hạn: RAG paper dùng thiết lập knowledge-intensive NLP với dense retriever
và generator được huấn luyện; repository này là pipeline architecture synthesis
với BM25F/RRF, optional local adapters và static validator. Citation này không
chứng minh component/deployment topology được suy ra đúng.

## 2. BM25F và RRF

### BM25F

Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
Beyond*, Foundations and Trends in Information Retrieval 4(1–2), 2009,
[DOI](https://doi.org/10.1561/1500000019).

Phần được dùng: §3.4 cho BM25 và §3.6 cho multiple streams/BM25F; các phép
toán field-weighted pseudo term frequency, field-length normalization và
saturation được đối chiếu với Eq. 3.19–3.21.

Repository implementation: `_bm25f_score()` dùng cùng dạng toán học nhưng
field names, default weights, index projection và query construction là
engineering adaptations. `k1=1.2`, `b=0.75` và neutral field weights không
được tuyên bố là tối ưu. RSJ-style IDF được chọn rõ trong code; negative IDF
không bị âm thầm clip để tránh giả vờ đang dùng một biến thể khác.

### Reciprocal Rank Fusion

Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and
Individual Rank Learning Methods*, SIGIR 2009,
[paper PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf),
[DOI](https://doi.org/10.1145/1571941.1572114).

Phần được dùng: §1, công thức `RRFscore(d) = sum_r 1 / (k + r(d))`. Paper báo
cáo thí nghiệm với `k=60`; repository giữ `k` configurable, dùng rank 1-based
và tie-break theo `evidence_id` để có output deterministic. RRF score là score
xếp hạng, không phải xác suất relevance hay confidence.

## 3. Semantic retrieval và structured output

### M3-Embedding

Chen et al., *M3-Embedding: Multi-Linguality, Multi-Functionality,
Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*, Findings
of ACL 2024, [ACL Anthology](https://aclanthology.org/2024.findings-acl.137/).

Phạm vi citation: abstract/metadata mô tả multilingual, cross-lingual và các
chức năng dense/sparse/multi-vector của M3-Embedding. Repository chỉ cung cấp
adapter embedding tùy chọn; chọn `BAAI/bge-m3` là model/configuration choice,
không phải kết luận tối ưu cho KB này.

### PICARD

Scholak, Schucher & Bahdanau, *PICARD: Parsing Incrementally for Constrained
Auto-Regressive Decoding from Language Models*, EMNLP 2021,
[ACL Anthology](https://aclanthology.org/2021.emnlp-main.779/).

Phần được dùng: abstract và §2 mô tả incremental parsing để reject token không
hợp lệ trong decoding. Pipeline này **không implement PICARD**: Ollama nhận JSON
Schema và code chạy post-generation validation trên toàn IR. Vì vậy PICARD chỉ
là analogy về formal contract, không phải implementation credit.

Ollama [API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
là nguồn cho việc dùng `stream=false` và `format` JSON/JSON Schema. JSON Schema
chỉ kiểm soát hình dạng/parseability; nó không chứng minh alias, topology,
evidence support hay semantic correctness.

## 4. Adaptive query decomposition

Petcu et al., *Query Decomposition for RAG: Balancing Exploration-Exploitation*,
EACL 2026, pp. 6857–6871,
[ACL Anthology](https://aclanthology.org/2026.eacl-long.322/),
[PDF](https://aclanthology.org/2026.eacl-long.322.pdf).

Phần được dùng:

- §3.2: mỗi sub-query là một bandit arm; pull kế tiếp document trên ranked
  list và cập nhật reward/posterior;
- §3.3: các assumption về relevance labels và chất lượng ranked list;
- Algorithm 1: Beta(1,1), Thompson sample, chọn arm, lấy document và update;
- §5.1: thí nghiệm các chính sách bandit/reward trong thiết lập của paper.

Repository adapter `adaptive_retrieval.py` chỉ dùng phần allocation backbone:
requirement-level lists, Bernoulli labels caller-provided, Thompson sampling và
research-only top-k UCB. Warm start, fixture labels, budget accounting và
absence of diversity/calibrated production judgments là engineering boundaries.
Adapter không được nối tự động vào main generation path và posterior không phải
điểm đúng của architecture.

## 5. Reflection và claim boundary

Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection*, ICLR 2024, [arXiv abstract](https://arxiv.org/abs/2310.11511).

Nguồn này được ghi như một hướng tham khảo về adaptive retrieval/reflection.
Repository **không** huấn luyện Self-RAG, không dùng reflection tokens và không
có critic model tương đương. Các file `matched_terms`, `unresolved_concepts`,
`validation.json` và human gate chỉ là governance artifacts của prototype.

## 6. Những điều không được claim từ các citation trên

- retrieved case top-1 là ground truth hoặc architecture oracle;
- `ready_for_review` là production readiness;
- RRF `k=60`, BM25F defaults hoặc `BAAI/bge-m3` là tối ưu phổ quát;
- JSON Schema là constrained decoding kiểu PICARD;
- adaptive posterior là xác suất evidence đúng cho case mới;
- diagram render được đồng nghĩa với semantic correctness, performance hoặc
  runtime deployability.

Các claim này cần dataset judgment độc lập, benchmark có leakage-safe split,
reviewer agreement và/hoặc Stage 2 adapters; hiện repository chưa có các bằng
chứng đó.
