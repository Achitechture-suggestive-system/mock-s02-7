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
  -> evidence-unit retrieval: BM25F + optional semantic + RRF
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

Chạy test:

```powershell
$env:PYTHONPATH = "$PWD\src"
py -3.13 -m unittest discover -s tests -v
py -3.13 -m compileall -q src tests
```

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
minh. Xem [docs/retrieval-research.md](docs/retrieval-research.md) và
[docs/references.md](docs/references.md).

Nếu bật semantic provider, score cosine của embedding là một nhánh phụ. Việc
chọn `BAAI/bge-m3` là adapter/model choice; paper M3-Embedding không chứng minh
model đó tối ưu cho Knowledge Base này. Hai nhánh được hợp nhất bằng
Reciprocal Rank Fusion:

```text
RRFscore(d) = sum_r 1 / (k + rank_r(d))
```

Đây là công thức trong §1 của Cormack, Clarke & Büttcher (SIGIR 2009). `k=60`
là default theo thí nghiệm của paper; code giữ configurable và không tuyên bố
đó là giá trị tối ưu phổ quát.

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
