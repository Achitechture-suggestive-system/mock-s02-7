# Retrieval math và implementation boundary

Phạm vi của tài liệu này là từ normalized stakeholder requirements tới ranked
Knowledge Base evidence. Nó không đánh giá architecture quality, runtime
correctness, performance hay production suitability.

## 1. Query và evidence unit hiện tại

`build_retrieval_queries()` tạo một query context-neutral và một query cho mỗi
functional requirement, non-functional requirement hoặc constraint được phép
retrieval. Query được xây deterministic từ text của input; pipeline không gọi
LLM để rewrite query và không thêm synonym/technology.

`build_evidence_units()` materialize evidence từ các field requirement/case
thật trong KB. Mỗi unit có `evidence_id`, `case_id`, `evidence_type`, literal
`text`, `source_path`, `source_locator`, `review_status` và `evidence_tier`.
Component/deployment PlantUML được giữ như paired case provenance; không tự
biến diagram thành một câu evidence mới.

Hệ quả: nhiều query có thể cùng chọn một `evidence_id`. Đây là quan hệ hợp lệ
và được lưu trong từng `queries[*].evidence`; không deduplicate mất dấu query.
Muốn tìm các evidence dùng chung, group theo `evidence_id` như hướng dẫn ở
[`log-artifacts.md`](log-artifacts.md).

## 2. BM25F

Repository đối chiếu BM25F với Robertson & Zaragoza (2009), §3.4 và §3.6,
Eq. 3.19–3.21: [DOI](https://doi.org/10.1561/1500000019).

Với query term `t`, evidence unit `d` và field `s`, code dùng:

```text
B_s(d) = (1 - b_s) + b_s * field_length_s(d) / average_field_length_s

pseudo_tf(t,d) = sum_s v_s * tf(t,s,d) / B_s(d)

BM25F(t,d) = ((k1 + 1) * pseudo_tf(t,d) / (k1 + pseudo_tf(t,d))) * IDF(t)
```

Score cuối là tổng theo các unique query terms. IDF được chọn rõ là RSJ-style:

```text
IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5))
```

Trong đó `N`, `df(t)`, field term frequency, field length, field average,
`k1`, `b_s` và `v_s` là input/index statistics của implementation. Negative IDF
không bị clip. Default `k1=1.2`, `b_s=0.75` và neutral weights là engineering
starting points, không phải hyperparameter đã được benchmark tối ưu.

Code location: `_bm25f_score()` và `_build_field_index()` trong
`src/arch_context_pipeline/pipeline.py`.

## 3. Dense semantic retrieval

Dense retrieval dùng một bi-encoder `f_theta` để mã hóa query và evidence
độc lập. Với query `q` và evidence `d`:

```text
z_q = f_theta(q)
z_d = f_theta(d)

hat(z) = z / ||z||_2

S_sem(q,d) = hat(z_q)^T hat(z_d)
            = (z_q · z_d) / (||z_q||_2 ||z_d||_2)
            = cosine(z_q, z_d)
```

`SentenceTransformerEmbeddingProvider` yêu cầu `normalize_embeddings=True`,
do đó `S_sem` được tính bằng dot product của hai vector đã L2-normalize. Nếu
provider tùy biến không khai báo vector đã normalize, code tính cosine với
độ dài vector ở runtime và trả `0` cho vector zero. Đây là similarity/ranking
score, không phải probability hoặc confidence.

Bi-encoder có thể bắt được paraphrase hoặc khác biệt ngôn ngữ mà lexical
matching bỏ sót, nhưng không nên thay thế BM25F một cách mù quáng. Repository
dùng dense retrieval như một rank list bổ sung. `BAAI/bge-m3` và
`Alibaba-NLP/gte-multilingual-base` là model choices; repository chỉ dùng dense
branch, không claim đã dùng toàn bộ sparse hoặc multi-vector functionality của
M3. Xem Reimers & Gurevych (2019), Karpukhin et al. (2020) và Chen et al.
(2024) trong [`references.md`](references.md).

## 4. Fusion và cross-encoder reranking

Khi dense provider được bật, RRF nhận **hai** rank list, không còn chỉ là biến
đổi rank lexical:

```text
RRF(d) = 1 / (k + r_lex(d)) + 1 / (k + r_sem(d))
```

Nếu một ranker không chứa `d`, chỉ các hạng thực sự có mặt được cộng. Công thức
tổng quát là:

```text
RRFscore(d) = sum_r 1 / (k + rank_r(d))
```

Rank là 1-based. `k=60` chỉ là default theo thí nghiệm paper; code giữ
configurable. Nếu semantic provider không được truyền vào, RRF chỉ có một
lexical list và đúng là chỉ tạo một score biến đổi từ rank lexical.

Reranker chạy sau RRF trên candidate pool nhỏ:

```text
C_q = top_M(RRF_q),  M = max(top_k, top_k * candidate_multiplier)
s_ce(q,d) = g_phi([CLS] q [SEP] d [SEP])
final_q = sort_desc({(d, s_ce(q,d)) : d in C_q})[:top_k]
```

`g_phi` là cross-encoder: query và evidence đi vào cùng một Transformer để
attention tương tác giữa hai chuỗi. Vì vậy nó thường tinh hơn bi-encoder
nhưng tốn chi phí theo từng cặp; đây là lý do chỉ rerank candidate pool sau
first-stage retrieval. Score của cross-encoder là logit/regression ranking
score; không tự động chuyển thành xác suất relevance. Một wrapper có thể áp
dụng một hàm đơn điệu như sigmoid để chuẩn hóa hiển thị, nhưng phép biến đổi
đó không tự tạo ra nhãn relevance và không nên dùng để so sánh số giữa hai
model. Khi reranker bật,
`retrieved_cases` và `queries[*].evidence[*].rank` dùng thứ tự reranker cuối,
còn `rrf_rank` vẫn được lưu để audit.

RRF/BM25F/semantic/reranker scores đều là ranking hoặc similarity scores,
không phải probability, confidence hay architecture correctness.

## 5. Retrieval evidence contract

`retrieval-evidence.json` lưu các trường để audit:

| Nhóm | Trường | Ý nghĩa |
|---|---|---|
| Query | `query_id`, `requirement_id`, `query_text`, `query_tokens` | query nào tạo ra candidate |
| Evidence | `evidence_id`, `case_id`, `evidence_type`, `text` | evidence literal nào được chọn |
| Candidate pool | `candidate_pool`, `candidate_pool_size` | toàn bộ top-M sau RRF, gồm cả item không lọt `top_k` cuối |
| Ranking | `lexical_rank/score`, `semantic_rank/score`, `rrf_rank/score`, optional `reranker_*` | đường đi của ranking |
| Explainability | `matched_terms` | query tokens giao với evidence tokens |
| Provenance | `source_path`, `source_locator`, `review_status`, `evidence_tier`, `provenance` | truy ngược và claim boundary |
| Exclusion | root `exclusions` | item bị loại và lý do, không silently drop |

`top-k` là số evidence lấy theo query, không phải relevance threshold. Không
được diễn giải `rank=1` hoặc `rrf_score` thành “đúng nhất” ngoài phạm vi ranking
của run đó.

## 6. Runtime flags và trạng thái được audit

Semantic retrieval và reranking vẫn là optional để mock path không phải tải model
ngoài. Chạy thật:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-semantic-reranked `
  --top-k 3 `
  --semantic-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Lần chạy đầu sẽ tải model từ Hugging Face. Cặp model này không cần remote code.
Nếu thay bằng model GTE/mGTE, cần bật hai cờ `--semantic-trust-remote-code` và
`--reranker-trust-remote-code` sau khi kiểm tra model/revision mà người dùng
đã tin cậy. Trong checkout hiện tại, GTE base đã gặp lỗi tương thích custom
rotary-position với `transformers 5.17`, nên run kiểm chứng bên dưới dùng
checkpoint chuẩn không cần remote code. Bundle phải cho thấy:

```yaml
semantic:
  status: enabled
  similarity: dot_product_of_l2_normalized_vectors
fusion:
  name: rrf
reranker:
  status: enabled
  candidate_pool: top-M RRF candidates
  score_transform: sigmoid | identity
final_ranking:
  basis: reranker
```

Nếu không truyền hai model flag, trạng thái `not_configured` là đúng và không
được ghi semantic/reranker score giả. Kết quả thực tế phải đọc từ
`retrieval-evidence.json`; model name, rank, score và candidate pool không được
suy ra chỉ từ README.

## 7. Đánh giá BGE-reranker-v2-m3 và MMARCO trong cùng toán pipeline

BGE-reranker-v2-m3 là một model cụ thể cho hàm `g_phi`, không phải một công
thức RRF mới. Với cùng candidate pool, hai run có dạng:

```text
C_q = top_M(RRF_q)
s_BGE(q,d)   = sigmoid(g_BGE(q,d))
s_MMARCO(q,d) = identity(g_MMARCO(q,d))
rank_model(q) = argsort_d in C_q descending s_model(q,d)
```

Theo model card BGE, reranker nhận trực tiếp cặp query-document và có thể trả
raw score hoặc map qua sigmoid về `[0,1]`; model card MMARCO mô tả cùng boundary
cross-encoder nhưng dùng multilingual MiniLMv2 và huấn luyện trên MMARCO, bản
dịch máy của MS MARCO sang 14 ngôn ngữ. Xem [BGE model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
và [MMARCO model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).

Trong runtime checkout này, `CrossEncoder.activation_fn` xác nhận BGE dùng
`Sigmoid`, còn MMARCO dùng `Identity`. Vì sigmoid đơn điệu, việc chuẩn hóa BGE
không thay đổi thứ hạng BGE; nó chỉ thay đổi biểu diễn score. Do đó các score
giữa hai model không cùng hệ đo và không được dùng để so sánh tuyệt đối.

Run đối chứng dùng `clinic_input.json`, dense model
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, RRF `k=60`,
`top_k=3`, candidate multiplier `5`, tức 15 ứng viên cho mỗi query:

| Metric | BGE | MMARCO | Cách đọc |
|---|---:|---:|---|
| Exact labeled top-1 | 4/5 | 5/5 | context + 4 requirement evidence có target; loại C-001 vì thiếu gold evidence |
| Labeled hit@3 | 5/5 | 5/5 | Cả hai giữ target trong top-3 |
| Top-1 agreement | 5/6 | 5/6 | Chỉ `FR-001` khác top-1 |
| Top-3 overlap | 2/3 mỗi query | 2/3 mỗi query | Hai model cùng candidate pool nhưng khác ưu tiên |
| Kendall tau trên 15 ứng viên | — | 0.438 so với BGE | Chỉ là mức đồng thuận của hai rank list, không phải accuracy |

Ở `FR-001`, MMARCO đưa `case-000008:E003` (functional requirement trực tiếp)
lên hạng 1; BGE đưa actor evidence `E002` lên hạng 1 và `E003` xuống hạng 2.
Đây là bằng chứng fixture cho thấy MMARCO tốt hơn ở exact top-1 của run này,
không phải bằng chứng BGE kém hơn trên domain nói chung. BGE lớn hơn (model card
ghi khoảng 0.6B tham số) trong khi MMARCO ghi khoảng 0.1B tham
số; đây là trade-off capacity/resource, chưa phải latency benchmark của máy
này.

Kết luận: nếu ưu tiên multilingual quality và đủ tài nguyên, BGE là ứng viên
đáng benchmark tiếp; nếu ưu tiên footprint và baseline đang được kiểm chứng,
MMARCO hiện thắng fixture exact top-1. Muốn chọn production cần gold judgments
độc lập và đo ít nhất Recall@candidate-pool, MRR@k hoặc nDCG@k, kèm latency và
memory. Không dùng raw reranker score để kết luận relevance hoặc architecture
correctness.

## 8. Equation-to-code map

| Operation | Code | Classification |
|---|---|---|
| Field normalization `B_s(d)` | `_bm25f_score` | Paper-derived form, project parameters |
| Weighted pseudo TF | `_bm25f_score` | Paper-derived form, project fields |
| BM25 saturation | `_bm25f_score` | Paper-derived form |
| RSJ IDF | `_build_field_index` / lexical scoring | Explicitly selected BM25 variant |
| RRF | `reciprocal_rank_fusion` | Paper-derived formula + project tie-break |
| L2 normalization/cosine | `SentenceTransformerEmbeddingProvider`, `retrieve` | SBERT/dense-retrieval similarity, model adapter |
| Dense query/document encoding | `SentenceTransformerEmbeddingProvider` | Model-dependent learned representation; not a project relevance label |
| Cross-encoder pair score | `SentenceTransformerCrossEncoderReranker`, `retrieve` | Neural reranking stage; candidate-pool engineering choice |
| Reranker output transform | `CrossEncoder.activation_fn`, `method.reranker.score_transform` | Runtime adapter metadata; monotone score display transform, not a relevance probability |
| Query decomposition | `build_retrieval_queries` | Engineering adaptation |
| Evidence IDs/provenance | `build_evidence_units` | Engineering audit contract |

Không có equation nào trong tài liệu này được dùng để claim architecture
correctness, confidence hoặc global quality score.
