[CmdletBinding()]
param(
    [string]$InputPath = (Join-Path (Get-Location) 'examples\clinic_input.json'),
    [string]$KbRoot = 'C:\disk D\KnowledgeBase_SoftwareArchitect',
    [string]$OutputDir = (Join-Path (Get-Location) 'out\clinic-ollama-bge-qwen3'),
    [int]$TopK = 3,
    [string]$Model = 'qwen3:4b',
    [string]$OllamaUrl = 'http://127.0.0.1:11436',
    [int]$LlmTimeout = 1200,
    [string]$SemanticModel = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    [string]$RerankerModel = 'BAAI/bge-reranker-v2-m3',
    [switch]$PullModel,
    [switch]$SemanticTrustRemoteCode,
    [switch]$RerankerTrustRemoteCode
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath([string]$PathValue) {
    return [System.IO.Path]::GetFullPath($PathValue)
}

$inputFull = Resolve-FullPath $InputPath
$kbFull = Resolve-FullPath $KbRoot
$outputFull = Resolve-FullPath $OutputDir
$ollamaBase = $OllamaUrl.TrimEnd('/')

if (-not (Test-Path -LiteralPath $inputFull -PathType Leaf)) {
    throw "Input file not found: $inputFull"
}
if (-not (Test-Path -LiteralPath $kbFull -PathType Container)) {
    throw "Knowledge Base directory not found: $kbFull"
}

$py = (Get-Command py.exe -ErrorAction Stop).Source
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repoRoot 'src'

Write-Host "Checking Ollama: $ollamaBase"
try {
    $version = Invoke-RestMethod -Uri "$ollamaBase/api/version" -Method Get
    Write-Host "Ollama version: $($version.version)"
} catch {
    throw "Cannot reach Ollama at $ollamaBase. Start scripts/start-ollama-gpu-cpu.ps1 first. Details: $($_.Exception.Message)"
}

if ($PullModel) {
    $ollama = (Get-Command ollama.exe -ErrorAction Stop).Source
    $oldOllamaHost = $env:OLLAMA_HOST
    try {
        $env:OLLAMA_HOST = ([System.Uri]$ollamaBase).Authority
        Write-Host "Ensuring Ollama model is available: $Model"
        & $ollama pull $Model
        if ($LASTEXITCODE -ne 0) {
            throw "ollama pull failed with exit code $LASTEXITCODE"
        }
    } finally {
        if ($null -eq $oldOllamaHost) {
            Remove-Item Env:OLLAMA_HOST -ErrorAction SilentlyContinue
        } else {
            $env:OLLAMA_HOST = $oldOllamaHost
        }
    }
}

$arguments = @(
    '-3.13',
    '-m', 'arch_context_pipeline',
    '--input', $inputFull,
    '--kb', $kbFull,
    '--out', $outputFull,
    '--top-k', "$TopK",
    '--generator', 'ollama',
    '--model', $Model,
    '--ollama-url', $ollamaBase,
    '--llm-timeout', "$LlmTimeout",
    '--semantic-model', $SemanticModel,
    '--reranker-model', $RerankerModel
)

if ($SemanticTrustRemoteCode) {
    $arguments += '--semantic-trust-remote-code'
}
if ($RerankerTrustRemoteCode) {
    $arguments += '--reranker-trust-remote-code'
}

Write-Host "Running evidence pipeline with Ollama model: $Model"
& $py @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}

$evidencePath = Join-Path $outputFull 'retrieval-evidence.json'
if (Test-Path -LiteralPath $evidencePath -PathType Leaf) {
    $evidence = Get-Content -LiteralPath $evidencePath -Raw | ConvertFrom-Json
    Write-Host "Semantic status: $($evidence.method.semantic.status)"
    Write-Host "Semantic model: $($evidence.method.semantic.model)"
    Write-Host "Reranker status: $($evidence.method.reranker.status)"
    Write-Host "Final ranking basis: $($evidence.method.final_ranking.basis)"
}

Write-Host "Bundle: $outputFull"
