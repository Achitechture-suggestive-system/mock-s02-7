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

## 3. Semantic branch và rank fusion

Semantic provider là optional. Nếu provider khai báo normalized vectors, score
là dot product giữa vectors đã L2-normalize, tương đương cosine similarity.
Nếu không có provider, output giữ `semantic.status=not_configured`; không bịa
semantic score.

Rank lists được hợp nhất bằng công thức RRF trong §1 của Cormack et al. (2009),
[paper PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf):

```text
RRFscore(d) = sum_r 1 / (k + rank_r(d))
```

Rank là 1-based. `k=60` chỉ là default theo thí nghiệm paper; code giữ
configurable. Stable ordering là score descending rồi `evidence_id` ascending.
RRF/BM25F/semantic scores đều là ranking or similarity scores, không phải
probability hay confidence.

## 4. Retrieval evidence contract

`retrieval-evidence.json` lưu các trường để audit:

| Nhóm | Trường | Ý nghĩa |
|---|---|---|
| Query | `query_id`, `requirement_id`, `query_text`, `query_tokens` | query nào tạo ra candidate |
| Evidence | `evidence_id`, `case_id`, `evidence_type`, `text` | evidence literal nào được chọn |
| Ranking | `lexical_rank/score`, `semantic_rank/score`, `rrf_rank/score`, optional `reranker_*` | đường đi của ranking |
| Explainability | `matched_terms` | query tokens giao với evidence tokens |
| Provenance | `source_path`, `source_locator`, `review_status`, `evidence_tier`, `provenance` | truy ngược và claim boundary |
| Exclusion | root `exclusions` | item bị loại và lý do, không silently drop |

`top-k` là số evidence lấy theo query, không phải relevance threshold. Không
được diễn giải `rank=1` hoặc `rrf_score` thành “đúng nhất” ngoài phạm vi ranking
của run đó.

## 5. Optional reranking

Reranker nếu được cấu hình chỉ chạy trên candidate pool của RRF. Kết quả được
lưu ở `reranker_score`/`reranker_rank`; nếu không chạy, RRF vẫn là final
ranking. Model name, candidate size và enable/disable state phải đọc từ bundle,
không suy ra từ README.

## 6. Equation-to-code map

| Operation | Code | Classification |
|---|---|---|
| Field normalization `B_s(d)` | `_bm25f_score` | Paper-derived form, project parameters |
| Weighted pseudo TF | `_bm25f_score` | Paper-derived form, project fields |
| BM25 saturation | `_bm25f_score` | Paper-derived form |
| RSJ IDF | `_build_field_index` / lexical scoring | Explicitly selected BM25 variant |
| RRF | `reciprocal_rank_fusion` | Paper-derived formula + project tie-break |
| Query decomposition | `build_retrieval_queries` | Engineering adaptation |
| Evidence IDs/provenance | `build_evidence_units` | Engineering audit contract |

Không có equation nào trong tài liệu này được dùng để claim architecture
correctness, confidence hoặc global quality score.
