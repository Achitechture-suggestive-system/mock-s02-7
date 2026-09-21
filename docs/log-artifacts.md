# Log và audit artifacts

## Phân biệt đúng loại dữ liệu

Pipeline hiện tại không có logging subsystem, event sink hoặc parser để đọc
stdout/stderr. `run_pipeline()` ghi một bundle file-based sau khi các stage
chạy xong. Vì vậy:

- stdout của CLI chỉ có 4 dòng summary;
- `generation-run.json` là execution metadata và output hashes;
- `retrieval-evidence.json` là retrieval audit artifact;
- `local-llm-response.json` là raw Ollama response, chỉ có ở Ollama run;
- không có file nào tự chứng minh semantic correctness hoặc production
  behavior.

## Artifact ledger

| Artifact | Nội dung cần đọc | Claim boundary |
|---|---|---|
| `input.json` | input raw được lưu lại | không đánh giá input |
| `requirements.json` | normalized actors/requirements/NFR/constraints | normalized không đồng nghĩa semantic đúng |
| `retrieval-evidence.json` | per-query evidence, score, rank, matched terms, source locator, exclusions | ranking/provenance, không phải ground truth |
| `architecture-context.json` | normalized input + evidence + profile + generation contract | context đã đóng gói, không chứng minh LLM hiểu |
| `llm-prompt.md` | prompt hoàn chỉnh dạng text | không chứng minh request đã gửi |
| `architecture-ir.json` | candidate IR nguồn cho hai renderer | không chứng minh topology runtime |
| `component.puml`, `deployment.puml` | hai view được render từ IR | prototype không tự claim đã render bằng PlantUML binary |
| `validation.json` | static checks, warnings, unresolved concepts | không phải semantic/performance test |
| `design-review.json` | review status và claim boundary | `pending_human_review` chưa phải approval |
| `handoff-gate.json` | các điều kiện gate và blocked reasons | Stage 2 chưa chạy nếu adapter chưa sẵn sàng |
| `generation-run.json` | run id, generator/model, timestamps, hashes, claim boundary | không phải full application log |
| `local-llm-response.json` | payload JSON trả từ Ollama | chỉ xuất hiện khi có Ollama call |
| `manifest.json` | bundle grouping/status/file count | không validate nội dung |
| `experiment-config.json` | kế hoạch Stage 2 | không phải benchmark result |
| `stage2-status.json` | adapter availability/status | không phải PCM/simulation log |

## Trích retrieval trace

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

Để kiểm tra nhiều token/query cùng truy đến một evidence:

```powershell
$rows = foreach ($query in $retrieval.queries) {
  foreach ($evidence in $query.evidence) {
    [pscustomobject]@{
      QueryId       = $query.query_id
      RequirementId = $query.requirement_id
      EvidenceId    = $evidence.evidence_id
      MatchedTerms  = ($evidence.matched_terms -join ', ')
      Source        = "$($evidence.source_path)#$($evidence.source_locator)"
    }
  }
}

$rows |
  Group-Object EvidenceId |
  Where-Object Count -gt 1 |
  Select-Object Name, Count, Group
```

Đây là cách đọc đúng: `matched_terms` cho biết token overlap được ghi nhận;
`EvidenceId` cho biết unit KB; `Source` cho biết locator để mở file nguồn.
Không dùng `matched_terms` một mình để kết luận relevance: cần đọc literal
`text`, query đầy đủ và provenance của case.

## Trích generation metadata

```powershell
$run = Get-Content "$bundle\generation-run.json" -Raw | ConvertFrom-Json
$run | Select-Object run_id, created_at, generator, model, retrieval_index, profile_version, claim_boundary

$validation = Get-Content "$bundle\validation.json" -Raw | ConvertFrom-Json
$validation | Select-Object overall_status, unresolved_concepts, claim_boundary

$gate = Get-Content "$bundle\handoff-gate.json" -Raw | ConvertFrom-Json
$gate | Select-Object gate_result, blocked_reasons, claim_boundary
```

Nếu bundle là Ollama run, đọc raw response:

```powershell
$rawPath = Join-Path $bundle 'local-llm-response.json'
if (Test-Path -LiteralPath $rawPath) {
  $raw = Get-Content $rawPath -Raw | ConvertFrom-Json
  $raw | Select-Object model, created_at, done, done_reason, total_duration, prompt_eval_count, eval_count
}
```

Nếu file không tồn tại trong mock run, đó là trạng thái đúng; không được ghi
“Ollama không trả log” hoặc suy đoán endpoint đã được gọi.

## Quy trình audit tối thiểu

1. Đọc `manifest.json` để xác định bundle status và file groups.
2. Đọc `retrieval-evidence.json`; chọn query cụ thể, không chỉ nhìn top case.
3. Đối chiếu `query_text`, `matched_terms`, `text`, `source_path` và
   `source_locator` với checkout KB.
4. Đọc `architecture-context.json` để biết evidence nào thật sự được truyền
   xuống generator.
5. Đọc `generation-run.json` để phân biệt `mock` với `ollama` và biết claim
   boundary/hash.
6. Đọc `architecture-ir.json` rồi `validation.json`; unresolved concepts và
   warning không được xóa để làm report đẹp.
7. Đọc `handoff-gate.json`; `blocked` là kết quả expected khi chưa có owner
   acceptance hoặc Stage 2 adapters.
