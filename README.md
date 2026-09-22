# Evidence-first architecture context pipeline

Prototype hiện tại nhận một stakeholder query tiếng Anh có cấu trúc, truy vấn
Knowledge Base kiến trúc, đóng gói evidence có provenance cho local LLM, sinh
IR trung gian, rồi render và kiểm tra cặp component/deployment view.

Đây là pipeline tạo **candidate architecture**. `ready_for_review` chỉ có nghĩa
là các kiểm tra tĩnh của prototype đã pass; nó không chứng minh semantic
correctness, runtime behavior, performance hay production readiness.

## Pipeline hiện tại

```text
English query JSON
  -> normalize requirements
  -> one retrieval query per eligible requirement
  -> evidence-unit retrieval: BM25F + dense semantic + RRF
  -> optional cross-encoder reranking of the RRF candidate pool
  -> context package with source provenance
  -> mock or Ollama JSON IR
  -> paired component/deployment PlantUML
  -> static cross-view validation
  -> human-review and Stage 2 handoff gate
```

Các module chính:

- `src/arch_context_pipeline/pipeline.py`: normalize, build evidence units,
  retrieve, build context, generate IR, render và validate.
- `src/arch_context_pipeline/adaptive_retrieval.py`: research adapter phân bổ
  budget giữa các requirement-level ranked lists; không thay thế retriever và
  không nằm trong đường chạy chính.
- `schemas/`: mock contracts cho retrieval evidence, architecture IR và
  validation.
- `tests/`: regression tests cho retrieval, IR và cross-view invariants.
- `pipeline_diagram.html`: sơ đồ các stage S01–S10 và hai lane Stage 2.

## Input: một query tiếng Anh chi tiết

File mẫu là [`examples/clinic_input.json`](examples/clinic_input.json). Phần
`raw_text` là một query tiếng Anh dài, đồng thời JSON tách riêng:

- `problem_statement` và `system_summary`;
- `actors`;
- `functional_requirements`, `non_functional_requirements`, `constraints`;
- `architecture_hints` dùng canonical vocabulary key;
- `domain_terms` giữ các thuật ngữ cần bảo toàn.

Mỗi requirement đủ điều kiện (`rag_eligible=true` hoặc mặc định được phép)
trở thành một query riêng. Vì vậy một query đầu vào có thể tạo các query
`CONTEXT-*`, `FR-*`, `NFR-*` và `C-*`, thay vì gom mọi câu thành một chuỗi duy
nhất.

Mục tiêu của input tiếng Anh chi tiết là để token trong stakeholder query
giao với token thật trong KB, ví dụ `electronic health record`, `practice
management`, `FHIR`, `SMART-on-FHIR`, `authorized applications`, `scoped
access`. Runtime hiện tại:

1. bảo toàn Unicode và token hóa theo từ/cụm đã có trong evidence;
2. tính lexical score trên từng evidence unit, không chỉ trên cả case;
3. ghi `matched_terms` để biết token nào thực sự giao nhau;
4. cho phép nhiều requirement/query cùng trỏ đến một `evidence_id` nếu evidence
   đó là bằng chứng chung;
5. giữ nguyên `case_id`, source path và source locator để truy ngược về KB.

Không tự thêm synonym, technology hoặc deployment topology vào query. Hint
kiến trúc được giữ cho context/generation contract và bị loại khỏi primary
retrieval text để không làm evidence tự củng cố chính hint của người dùng.

## Chạy pipeline

Pipeline dùng Python standard library cho mock path. Trên Windows PowerShell:

```powershell
Set-Location 'C:\disk D\mock-s02-7'
$env:PYTHONPATH = "$PWD\src"

py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-english `
  --top-k 3
```

Kết quả stdout chỉ là summary:

```text
Generated bundle: ...\out\clinic-english
Top retrieval: case-... (RRF 0.xxxxxx)
Validation: ready_for_review
Handoff gate: blocked
```

`mock` là generator deterministic, không gọi model. Muốn dùng Ollama local:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-local-qwen3 `
  --top-k 3 `
  --generator ollama `
  --model qwen3:4b `
  --ollama-url 'http://127.0.0.1:11434' `
  --llm-timeout 600
```

Nhánh Ollama gọi `POST /api/generate` với JSON Schema. Schema chỉ giúp output
parse được và có đúng khung field; validator sau generation vẫn cần thiết.
Pipeline không âm thầm fallback từ Ollama sang mock.

### Chạy Ollama bằng GPU và chia phần còn thiếu sang CPU

Ollama không có tham số để đặt cứng tỷ lệ GPU/CPU theo phần trăm. Scheduler tự
đặt các layer vừa VRAM lên GPU và phần còn lại vào RAM; `ollama ps` sẽ hiển thị
ví dụ `48%/52% CPU/GPU`. Script cấu hình trong repo là
[`scripts/start-ollama-gpu-cpu.ps1`](scripts/start-ollama-gpu-cpu.ps1).

Trên máy Windows hiện tại, cần cập nhật NVIDIA driver lên tối thiểu 550 và
cập nhật Ollama trước. Driver đang được phát hiện là `546.30`, còn log Ollama
ghi `CUDA error: device kernel image is invalid`, nên chỉ đổi cổng hoặc tăng
timeout không sửa được lỗi này.

Sau khi quit Ollama ở system tray, mở một PowerShell riêng và chạy:

```powershell
Set-Location 'C:\disk D\mock-s02-7'
.\scripts\start-ollama-gpu-cpu.ps1 -Port 11436 -GpuId auto -ContextLength 32768
```

Không truyền `-GpuId 0` trên Windows này: để `auto` giúp Ollama tự discovery
CUDA. `CUDA_VISIBLE_DEVICES=0` hiện làm bản Ollama đang cài bỏ qua GPU và chỉ
khởi động backend CPU. Script cũng không ép `OLLAMA_LLM_LIBRARY=cuda`; Ollama
cần tự chọn thư viện tương thích đang cài, hiện là `cuda_v13`.

Ở PowerShell khác, kiểm tra server và phân bổ:

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11436'
Invoke-RestMethod 'http://127.0.0.1:11436/api/version'
ollama ps
nvidia-smi
```

Nếu `ollama ps` hiện `100% GPU` thì toàn bộ model vừa VRAM; nếu hiện dạng
`CPU/GPU` thì model đã được chia giữa RAM và VRAM. Pipeline hiện gửi
`num_ctx=32768` trực tiếp trong request, vì vậy giá trị `OLLAMA_CONTEXT_LENGTH`
chỉ là mặc định cho server và không ghi đè tham số request này.

Khi server đã qua probe, chạy pipeline bằng đúng cổng đó:

```powershell
$env:PYTHONPATH = "$PWD\src"
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-ollama-qwen3-gpu-cpu `
  --top-k 1 `
  --generator ollama `
  --model qwen3:4b `
  --ollama-url 'http://127.0.0.1:11436' `
  --llm-timeout 1200
```

Log server Windows nằm ở `%LOCALAPPDATA%\Ollama\server.log`. Nếu probe còn
trả `device kernel image is invalid`, chưa nên chạy pipeline; kiểm tra lại
driver, phiên bản Ollama và log GPU trước.

Hoặc chạy một lệnh qua script đã cấu hình sẵn Ollama generator, semantic
embedding và BGE reranker:

```powershell
.\scripts\run-ollama-pipeline.ps1 `
  -OllamaUrl 'http://127.0.0.1:11436' `
  -Model 'qwen3:4b' `
  -PullModel
```

Script chỉ gọi pipeline sau khi probe `GET /api/version` thành công. Nếu server
đang dùng cổng mặc định, truyền `-OllamaUrl 'http://127.0.0.1:11434'`. `-PullModel`
là tùy chọn; bỏ nó nếu model đã có sẵn trong Ollama.

Chạy test:

```powershell
$env:PYTHONPATH = "$PWD\src"
py -3.13 -m unittest discover -s tests -v
py -3.13 -m compileall -q src tests
```

## Dense semantic retrieval và reranking

CLI chính mặc định chạy đủ hybrid retrieval: BM25F + dense semantic rank được
hợp nhất bằng RRF, sau đó rerank candidate pool bằng
`BAAI/bge-reranker-v2-m3`. Vì vậy lần đầu chạy cần có `sentence-transformers`
và tải model từ Hugging Face:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-semantic-reranked `
  --top-k 3 `
  --semantic-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --reranker-model BAAI/bge-reranker-v2-m3
```

Các model trên không cần remote code và là cấu hình đã được chạy kiểm chứng
trong checkout này. Nếu thay bằng dòng GTE/mGTE, cần thêm
`--semantic-trust-remote-code` và `--reranker-trust-remote-code` sau khi đã
kiểm tra model/revision tương ứng.

Muốn chạy mock/lexical-only mà không tải model, dùng rõ ràng:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-mock `
  --no-semantic `
  --no-reranker
```

Compatibility note: một lần thử `Alibaba-NLP/gte-multilingual-base` với
`transformers 5.17` hiện tại dừng trong custom rotary-position implementation;
do đó GTE chưa được đánh dấu là run đã kiểm chứng ở checkout này. Kết quả bên
dưới dùng checkpoint chuẩn không cần remote code.

Trạng thái cần đọc trong `retrieval-evidence.json`, không suy ra từ command:

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

Toán semantic được dùng là:

```text
z_q = f_theta(q),  z_d = f_theta(d)
hat(z) = z / ||z||_2
S_sem(q,d) = hat(z_q)^T hat(z_d)
            = cosine(z_q, z_d)
```

Query và evidence được encode độc lập bằng bi-encoder. Khi đã normalize L2,
cosine similarity bằng dot product. Dense ranking được dùng cùng lexical
ranking trong RRF:

```text
RRF(d) = 1 / (k + r_lex(d)) + 1 / (k + r_sem(d))
```

Sau đó cross-encoder nhận từng cặp `(query, evidence)` trong candidate pool:

```text
s_ce(q,d) = g_phi([CLS] q [SEP] d [SEP])
final = sort_desc(s_ce)[:top_k]
```

`semantic_score`, `rrf_score` và `reranker_score` đều là score để xếp hạng,
không phải xác suất relevance hay bằng chứng architecture correctness. Khi
reranker bật, `rank` là thứ tự cuối từ reranker; `rrf_rank` vẫn được lưu để
audit. Chi tiết công thức và paper nằm ở
[`docs/retrieval-research.md`](docs/retrieval-research.md) và
[`docs/references.md`](docs/references.md).

### BGE-reranker-v2-m3 là reranker mặc định

Trong pipeline, BGE không thay đổi BM25F, dense embedding hoặc công thức RRF.
Nó chỉ thay thế hàm chấm điểm cross-encoder `g_phi` trên cùng candidate pool:

```text
C_q = top_M(RRF_q),  M = max(top_k, top_k * candidate_multiplier)
s_model(q,d) = a_model(g_phi(q,d))
final_model(q) = sort_desc({(d, s_model(q,d)) : d in C_q})[:top_k]
```

Với BGE, `a_model` trong runtime SentenceTransformers hiện tại là `sigmoid`;
MMARCO chỉ được giữ làm baseline lịch sử và dùng `identity`. Sigmoid là
hàm đơn điệu nên giữ nguyên thứ tự trong BGE, chỉ đổi thang điểm. Vì vậy không
được so sánh trực tiếp `0.99` của BGE với `10.7` của MMARCO; phải so thứ hạng
hoặc dùng metric có nhãn.

| Tiêu chí | BGE-reranker-v2-m3 mặc định | MMARCO baseline lịch sử |
|---|---|---|
| Vai trò | Cross-encoder multilingual | Cross-encoder multilingual |
| Nền tảng/model card | BGE-M3, multilingual, khoảng 0.6B tham số | multilingual MiniLMv2, khoảng 0.1B tham số |
| Huấn luyện công bố | Reranker multilingual của BGE | MMARCO, bản dịch máy MS MARCO qua 14 ngôn ngữ |
| Score trong run này | Sigmoid, quan sát trong `[0.000481, 0.999980]` | Identity/raw, quan sát trong `[-7.633160, 10.786008]` |
| Candidate pool | Cùng 15 evidence/query | Cùng 15 evidence/query |

Kết quả kiểm chứng trên `examples/clinic_input.json`, với cùng semantic model,
BM25F, RRF `k=60`, `top_k=3` và candidate multiplier `5`:

| Metric trên fixture có nhãn | BGE | MMARCO |
|---|---:|---:|
| Exact labeled top-1 | 4/5 | 5/5 |
| Labeled hit@3 | 5/5 | 5/5 |
| Top-1 agreement giữa hai model | 5/6 | 5/6 |
| Top-3 set overlap giữa hai model | 2/3 ở cả 6 query | 2/3 ở cả 6 query |
| Kendall tau trên candidate pool 15 item | — | trung bình 0.438 giữa hai rank list |

`C-001` không được tính vào exact labeled metric vì KB hiện tại không có
evidence constraint tương ứng để làm gold label. Ở `FR-001`, MMARCO đưa trực
tiếp evidence `case-000008:E003` lên hạng 1, còn BGE đặt nó ở hạng 2 sau actor
description `E002`; đây là khác biệt có ý nghĩa nhất của run nhỏ này. Cả hai
model đều đạt labeled hit@3, nên fixture này chưa đủ để kết luận model nào tốt
hơn một cách tổng quát.

Kết luận engineering: pipeline hiện dùng BGE làm mặc định; MMARCO chỉ là
baseline lịch sử để tái lập kết quả cũ. Quyết định production cần tập query/evidence
được đánh nhãn độc lập và báo cáo MRR/nDCG/Recall cùng latency, memory và
candidate-pool recall. Các bundle thực nghiệm là
[`BGE`](out/clinic-bge-reranked/retrieval-evidence.json) và
[`MMARCO`](out/clinic-semantic-reranked/retrieval-evidence.json).

### RRF so với reranker model: ablation đúng cách

Không nên gọi RRF là một reranker model. RRF là thuật toán fusion không học
tham số; nó hợp nhất rank BM25F và rank semantic để tạo candidate pool hoặc
final ranking khi chưa bật cross-encoder. So sánh công bằng phải giữ nguyên
input/KB và đo bốn cấu hình:

```text
BM25F
semantic + RRF
semantic + RRF + MMARCO
semantic + RRF + BGE
```

Trên cùng `clinic_input.json`, đánh nhãn target cho CONTEXT và bốn
functional/non-functional requirements, kết quả là:

| Cấu hình | Candidate-pool recall@15 | Hit@1 | Hit@3 | MRR@3 |
|---|---:|---:|---:|---:|
| BM25F | 5/5 | 5/5 | 5/5 | 1.000 |
| Semantic + RRF | 5/5 | 5/5 | 5/5 | 1.000 |
| Semantic + RRF + MMARCO | 5/5 | 5/5 | 5/5 | 1.000 |
| Semantic + RRF + BGE | 5/5 | 4/5 | 5/5 | 0.900 |

Diễn giải đúng của fixture này: RRF đã đủ tốt để giữ target và đặt target
đúng ở top-1; MMARCO không cải thiện thêm top-1 nhưng cũng không làm hỏng thứ
tự; BGE làm hỏng một top-1 ở `FR-001`. Đây là kết quả của fixture nhỏ, không
phải bằng chứng RRF luôn tốt hơn neural reranker. Muốn kết luận production
cần nhiều query có judgment độc lập, rồi đo Recall@M trước rerank và MRR/nDCG
sau rerank.

### Benchmark mở rộng: RRF so với BGE trên 120 input

Để tránh kết luận từ fixture 5 query, script
[`experiments/rrf-vs-bge/evaluate.py`](experiments/rrf-vs-bge/evaluate.py) chạy 60
target evidence thật trong production KB, mỗi target có hai input: một bản giữ
từ vựng nguồn và một bản paraphrase. Cả hai cấu hình dùng cùng BM25F, cùng
`paraphrase-multilingual-MiniLM-L12-v2`, `rrf_k=60`, candidate pool top-50 và
`top_k=10`. Khác biệt duy nhất là RRF được trả về trực tiếp hay top-50 được
đưa qua `BAAI/bge-reranker-v2-m3`.

| Metric trên 120 input | Semantic + RRF | Semantic + RRF + BGE | Chênh lệch |
|---|---:|---:|---:|
| Candidate recall@50 | 100.00% | 100.00% | 0 |
| Hit@1 | 81.67% | 82.50% | +0.83 điểm % |
| Hit@3 | 92.50% | 94.17% | +1.67 điểm % |
| Hit@5 | 96.67% | 99.17% | +2.50 điểm % |
| Hit@10 | 96.67% | 100.00% | +3.33 điểm % |
| MRR@10 | 0.8753 | 0.8894 | +0.0141 |
| Mean rank khi target nằm trong top-50 | 2.175 | 1.383 | tốt hơn 0.792 hạng |

BGE cải thiện 16 input, làm target tụt 12 input và giữ nguyên 92 input. Hai
phương án chỉ cùng top-1 ở 77.50% input; overlap top-3 trung bình là 55.83%.
Điều này cho thấy BGE thực sự đang sắp xếp lại candidate pool, không chỉ đổi
thang điểm. Trên riêng 60 paraphrase, MRR@10 tăng từ `0.7506` lên `0.7955`
và Hit@1 tăng từ `63.33%` lên `68.33%`; trên 60 input giữ từ vựng nguồn,
RRF đạt MRR `1.0000` còn BGE giảm còn `0.9833`. Vì vậy BGE có lợi rõ nhất khi
query diễn đạt khác evidence, còn RRF mạnh và ổn định với từ khóa trực tiếp.

Các thay đổi đáng chú ý:

- BGE kéo `case-000009:E003` offline learning từ hạng 26 lên 5,
  `case-000015:E005` GPX từ 46 lên 4, `case-000016:E004` geolocation từ 27
  lên 1 và `case-000030:E010` store-and-forward từ 4 lên 1.
- BGE làm tụt một số target vốn đã đúng ở RRF: public-health
  `case-000020:E003` từ 1 xuống 4, research repository `case-000033:E004`
  từ 1 xuống 3 và `case-000033:E007` từ 3 xuống 9.
- Candidate recall bằng nhau ở 100%, nên trong benchmark này BGE không cứu
  được target bị loại khỏi candidate pool; tác dụng của nó chỉ là precision
  của thứ tự cuối. Đây là khác biệt quan trọng giữa reranking và retrieval.

Artifact đầy đủ, gồm 120 query, gold evidence, top-10 của mỗi phương án,
rank/score của target và các ca tăng/giảm hạng, nằm ở
[`experiments/rrf-vs-bge/benchmark-result.json`](experiments/rrf-vs-bge/benchmark-result.json).

Kết luận engineering có điều kiện: với bộ nhãn hiện tại, BGE thắng nhẹ về
MRR và recall ở top-k, đặc biệt trên paraphrase; RRF đủ tốt và ít tốn compute
hơn khi truy vấn đã chia sẻ nhiều từ khóa với KB. Không nên suy ra BGE luôn
tốt hơn: 12/120 ca bị tụt, và nhãn hiện tại là một gold evidence đơn cho mỗi
query, chưa phải đánh giá graded relevance hoặc chất lượng architecture IR.

### BGE-reranker-v2-m3 so với MMARCO trên cùng 120 input

Đây là phép so sánh hai cross-encoder trên cùng top-50 RRF candidate pool;
embedding, BM25F và `rrf_k=60` không đổi:

| Metric | BGE-reranker-v2-m3 | MMARCO mMiniLMv2 |
|---|---:|---:|
| Hit@1 | 82.50% | 79.17% |
| Hit@3 | 94.17% | 95.00% |
| Hit@5 | 99.17% | 97.50% |
| Hit@10 | 100.00% | 100.00% |
| MRR@10 | 0.8894 | 0.8767 |
| Mean target rank | 1.383 | 1.400 |

BGE thắng thứ hạng gold ở 15 input, MMARCO thắng 11 và 94 input hòa. BGE
thắng rõ hơn với paraphrase (`MRR@10 0.7955` so với `0.7617`); MMARCO nhỉnh
hơn với query giữ nguyên từ vựng nguồn (`0.9917` so với `0.9833`). Cả hai có
candidate recall@50 là 100%, nên khác biệt nằm ở rerank cuối, không phải khả
năng lấy candidate.

Khuyến nghị: giữ BGE làm mặc định cho mục tiêu chất lượng và query đa ngôn ngữ;
dùng MMARCO khi ưu tiên CPU latency, memory hoặc throughput. Model card ghi
MMARCO khoảng `0.1B` tham số còn BGE khoảng `0.6B`; trong run 120 input local,
MMARCO hoàn tất khoảng 89 giây, còn run BGE trước đó khoảng 946 giây gồm load
model và retrieval. Đây là đo trên checkout/máy hiện tại, không phải claim
latency phổ quát. Kết quả chi tiết nằm trong
[`experiments/rrf-vs-bge/comparison-bge-vs-mmarco.json`](experiments/rrf-vs-bge/comparison-bge-vs-mmarco.json).

## Bundle output và cách đọc “log” cho đúng

Pipeline hiện **không có logging subsystem hoặc log parser riêng**. Thứ được
ghi ra là một bundle các JSON/Markdown/PlantUML audit artifacts. Vì vậy “trích
log” trong project này phải hiểu là đọc các artifact đã persist sau run; không
được gọi `stdout` summary hoặc `retrieval-evidence.json` là toàn bộ runtime log.

Bảng đầy đủ về field và lệnh trích xuất nằm ở
[`docs/log-artifacts.md`](docs/log-artifacts.md). Các file quan trọng:

| File | Nó chứng minh được | Nó không chứng minh được |
|---|---|---|
| `input.json` | raw input đã dùng cho run | input đúng domain |
| `requirements.json` | normalized facts/requirements | requirement semantic đúng |
| `retrieval-evidence.json` | query, evidence literal, rank/score, matched terms, content-term audit và provenance | top-1 là đáp án kiến trúc |
| `architecture-context.json` | context package gửi cho generator | LLM đã hiểu đúng context |
| `llm-prompt.md` | prompt/contract dạng đọc được | model đã chạy thành công |
| `generation-run.json` | generator, model, run id, output hashes và claim boundary | chất lượng model |
| `local-llm-response.json` | raw HTTP payload nếu chạy Ollama | file có ở mock run |
| `architecture-ir.json` | candidate IR được render | topology đúng production |
| `validation.json` | static checks và unresolved concepts | semantic correctness/performance |
| `design-review.json` | trạng thái review artifact | architect đã duyệt nếu status pending |
| `handoff-gate.json` | lý do gate blocked/allowed | Stage 2 đã thực thi |
| `manifest.json` | nhóm file và bundle status | tính đúng đắn của nội dung |
| `experiment-config.json`, `stage2-status.json` | kế hoạch/trạng thái Stage 2 | benchmark, PCM hay solver đã chạy |

Đặc biệt, `generation-run.json` là execution metadata của pipeline; nó không
phải stdout/stderr đầy đủ. `local-llm-response.json` chỉ được ghi khi Ollama
được gọi. Với mock run, `generator=mock` và `structured_output=false` là bằng
chứng generator deterministic đã được chọn.

Ví dụ trích retrieval audit bằng PowerShell:

```powershell
$bundle = '.\out\clinic-english'
$retrieval = Get-Content "$bundle\retrieval-evidence.json" -Raw | ConvertFrom-Json

foreach ($query in $retrieval.queries) {
  foreach ($evidence in $query.evidence) {
    [pscustomobject]@{
      QueryId        = $query.query_id
      RequirementId  = $query.requirement_id
      QueryText      = $query.query_text
      EvidenceId     = $evidence.evidence_id
      CaseId         = $evidence.case_id
      Rank           = $evidence.rank
      RrfRank        = $evidence.rrf_rank
      RrfScore       = $evidence.rrf_score
      MatchedTerms   = ($evidence.matched_terms -join ', ')
      Source         = "$($evidence.source_path)#$($evidence.source_locator)"
      ReviewStatus   = $evidence.review_status
      EvidenceTier   = $evidence.evidence_tier
    }
  }
} | Format-Table -AutoSize
```

Muốn kiểm tra một evidence được nhiều query dùng chung:

```powershell
$rows = foreach ($query in $retrieval.queries) {
  foreach ($evidence in $query.evidence) {
    [pscustomobject]@{
      QueryId = $query.query_id
      RequirementId = $query.requirement_id
      EvidenceId = $evidence.evidence_id
      MatchedTerms = ($evidence.matched_terms -join ', ')
      Source = "$($evidence.source_path)#$($evidence.source_locator)"
    }
  }
}

$rows | Group-Object EvidenceId | Where-Object Count -gt 1 |
  Select-Object Name, Count, Group
```

Khi audit một row, phải đối chiếu ít nhất `query_text` → `matched_terms` →
`evidence_id` → `source_path/source_locator` → file thật trong KB. `rrf_score`
chỉ là score xếp hạng, không phải probability, confidence hoặc percentage
relevance.

### Audit input ngoài miền KB

Để kiểm tra pipeline có nhận diện query lạ hay chỉ ép trả top-k, chạy fixture:

```powershell
py -3.13 -m arch_context_pipeline `
  --input .\examples\out_of_domain_cryobot_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\ood-cryobot-audit `
  --top-k 3
```

Đọc trường `retrieval_audit` trong
`out/ood-cryobot-audit/retrieval-evidence.json`:

```powershell
$audit = Get-Content '.\out\ood-cryobot-audit\retrieval-evidence.json' -Raw |
  ConvertFrom-Json

$audit.retrieval_audit | Select-Object overall_status, weak_query_ids
$audit.retrieval_audit.queries |
  Select-Object query_id, support_status, content_coverage,
    known_content_terms, oov_content_terms, top_evidence_id
```

`out_of_domain_candidate` là cảnh báo heuristic dựa trên content-term overlap
và OOV terms, không phải một OOD classifier đã được benchmark. Pipeline vẫn
trả top-k để giữ audit trail, nhưng thêm `CHK-008=fail`, chuyển validation
sang `invalid` và giữ handoff gate `blocked`; không được đọc các case đó như
evidence phù hợp.

## Retrieval và nguồn công thức

Lexical branch dùng field-normalized BM25F trên evidence units. Cụ thể, code
áp dụng pseudo term frequency, field-length normalization và saturation theo
Robertson & Zaragoza, §3.6, Eq. 3.19–3.21; các giá trị field/default là lựa
chọn engineering của project, không phải hyperparameter tối ưu được chứng
minh. Dense branch dùng bi-encoder cosine similarity; cross-encoder chỉ đọc
candidate pool nhỏ sau RRF. Xem [docs/retrieval-research.md](docs/retrieval-research.md)
và [docs/references.md](docs/references.md).

Nếu bật semantic provider, score cosine của embedding là một rank list bổ sung.
Việc chọn `BAAI/bge-m3` là adapter/model choice; paper M3-Embedding không chứng
minh model đó tối ưu cho Knowledge Base này. Hai nhánh được hợp nhất bằng
Reciprocal Rank Fusion:

```text
RRFscore(d) = sum_r 1 / (k + rank_r(d))
```

Đây là công thức trong §1 của Cormack, Clarke & Büttcher (SIGIR 2009). `k=60`
là default theo thí nghiệm của paper; code giữ configurable và không tuyên bố
đó là giá trị tối ưu phổ quát.

Nếu semantic provider không được bật, RRF chỉ nhận lexical rank list và về bản
chất chỉ biến đổi rank lexical. Nếu reranker được bật, RRF tạo candidate pool
và cross-encoder quyết định thứ tự evidence cuối.

### Citation và giới hạn diễn giải

- RRF và công thức `sum_r 1/(k + rank_r(d))` được trích từ Cormack, Clarke &
  Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and Individual Rank
  Learning Methods*, SIGIR 2009 ([DOI](https://doi.org/10.1145/1571941.1572114)).
  `k=60` là cấu hình thí nghiệm được báo cáo trong bài, không phải hằng số tối
  ưu phổ quát cho KB này.
- Model card BGE có liên kết tới Li et al., *Making Large Language Models A
  Better Foundation For Dense Retrieval* ([arXiv:2312.15503](https://arxiv.org/abs/2312.15503));
  đây là bài LLaRA về dense retrieval, không phải bài mô tả trực tiếp
  `BAAI/bge-reranker-v2-m3`. BGE-M3 được đối chiếu riêng với Chen et al.,
  *M3-Embedding* ([ACL Findings 2024](https://aclanthology.org/2024.findings-acl.137/));
  paper này nói về embedding family, không biến thành kết quả reranker của repo.
- Với đúng checkpoint `BAAI/bge-reranker-v2-m3`, nguồn runtime chính là
  [model card chính thức](https://huggingface.co/BAAI/bge-reranker-v2-m3): reranker
  nhận query-document pair và trả similarity score; sigmoid chỉ là phép chuẩn
  hóa hiển thị. Không gán các kết quả benchmark của paper BGE/M3 thành kết quả
  của repository này.
- Cross-encoder reranking sau first-stage retrieval được mô tả theo Nogueira &
  Cho, *Passage Re-ranking with BERT* ([arXiv:1901.04085](https://arxiv.org/abs/1901.04085)).

Adaptive query-decomposition nằm ngoài main path:

```powershell
py -3.13 -m arch_context_pipeline.adaptive_retrieval `
  --input .\examples\clinic_input.json `
  --kb 'C:\disk D\KnowledgeBase_SoftwareArchitect' `
  --out .\out\clinic-adaptive `
  --labels .\examples\clinic_bandit_judgments.json `
  --retrieval-top-k 20 `
  --budget 12 `
  --policy thompson_bernoulli `
  --seed 17
```

Adapter này được đối chiếu với Petcu et al., §3.2–§3.3 và Algorithm 1: mỗi
requirement-level ranked list là một arm, mỗi pull lấy evidence kế tiếp và
nhận reward từ label caller-provided. Fixture label là minh họa, không phải
production ground truth. Chi tiết ở
[`docs/research-query-decomposition-bandits.md`](docs/research-query-decomposition-bandits.md).

## Context, IR và paired views

`architecture-context.json` gồm normalized input, ranked evidence/provenance,
profile/vocabulary hashes và generation contract. Generator phải trả
`architecture-ir.json` trước khi renderer tạo:

- `component.puml`: logical/component view;
- `deployment.puml`: deployment view;
- `deployment_instances[*].instance_of`: liên kết về logical alias;
- `deployment_relations[*].logical_relation_id`: liên kết về relation trong IR.

Static validator kiểm tra alias, relation endpoint, relation id, allocation,
environment frame và cross-view references. `unknown`/`unresolved_concepts`
được giữ lại khi input/evidence không đủ; không ép chọn technology chỉ vì
precedent đứng hạng cao có technology đó.

## Trạng thái hiện tại và giới hạn

Trong bundle mock đã kiểm tra tại `out/clinic-english`, retrieval có các query
requirement-level và evidence `case-000008:E001` là evidence top của context
query; validation là `ready_for_review`, còn handoff gate là `blocked`. Đây là
kết quả của đúng input/KB checkout tại thời điểm chạy, không phải expected
benchmark hay claim tổng quát.

Prototype chưa thực hiện:

- human owner acceptance thật;
- PlantUML renderer execution;
- code generation, build/test adapter;
- Palladio PCM model, solver hoặc simulation;
- production relevance judgments và benchmark so sánh.

`experiment-config.json` chỉ mô tả khung thí nghiệm. `stage2-status.json`
ghi rõ adapter chưa implement; không được đọc các file này như performance log.

## Tài liệu tham chiếu

- [Research boundary và citation ledger](RESEARCH.md)
- [Retrieval math và implementation boundary](docs/retrieval-research.md)
- [Log/audit artifact extraction](docs/log-artifacts.md)
- [Exact paper references](docs/references.md)
- Architecture KB: `C:\disk D\KnowledgeBase_SoftwareArchitect` nếu checkout
  KB tồn tại ở đường dẫn này
