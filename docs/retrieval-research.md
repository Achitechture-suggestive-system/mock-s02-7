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

Trong CLI, model mặc định hiện tại là
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; có thể thay
bằng `BAAI/bge-m3` qua `--semantic-model`. Việc thay model chỉ thay
`f_theta` — hàm biến query/evidence thành vector — còn phép chấm vẫn là
cosine của vector đã L2-normalize. Sau đó code sắp xếp giảm dần thành
`semantic_rank`; RRF dùng rank này cùng `lexical_rank` theo công thức RRF,
không dùng raw cosine như một xác suất.

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

CLI chính bật semantic retrieval và BGE reranking theo mặc định. API thấp tầng
vẫn cho phép bỏ provider để chạy test/mock deterministic. Chạy thật:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-semantic-reranked `
  --top-k 3 `
  --semantic-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --reranker-model BAAI/bge-reranker-v2-m3
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

Nếu truyền `--no-semantic` hoặc `--no-reranker`, trạng thái tương ứng là
`not_configured`; không được ghi score giả. Kết quả thực tế phải đọc từ
`retrieval-evidence.json`; model name, rank, score và candidate pool không được
suy ra chỉ từ README.

## 7. Equation-to-code map

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
