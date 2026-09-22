[CmdletBinding()]
param(
    [int]$Port = 11434,
    [string]$GpuId = "auto",
    [int]$ContextLength = 32768,
    [int]$NumParallel = 1,
    [int]$MaxLoadedModels = 1
)

$ErrorActionPreference = "Stop"

$ollama = (Get-Command ollama.exe -ErrorAction Stop).Source

# Ollama's scheduler automatically places as many layers as fit in VRAM and
# leaves the remainder in system RAM. Do not set CUDA_VISIBLE_DEVICES=-1:
# that value explicitly forces CPU-only execution.
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_CONTEXT_LENGTH = "$ContextLength"
$env:OLLAMA_NUM_PARALLEL = "$NumParallel"
$env:OLLAMA_MAX_LOADED_MODELS = "$MaxLoadedModels"
$env:OLLAMA_KEEP_ALIVE = "10m"

Write-Host "Starting Ollama CUDA server at $($env:OLLAMA_HOST)"
if ($GpuId -eq "auto") {
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Write-Host "CUDA_VISIBLE_DEVICES=unset (Ollama auto-detect)"
} else {
    $env:CUDA_VISIBLE_DEVICES = $GpuId
    Write-Host "CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"
}

# Let the installed Ollama build choose its matching CUDA library (for this
# machine that is cuda_v13). Forcing the generic value `cuda` can make this
# Windows build fall back to the CPU backend.
Remove-Item Env:OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue
Write-Host "OLLAMA_LLM_LIBRARY=unset (Ollama backend auto-detect)"
Write-Host "OLLAMA_CONTEXT_LENGTH=$($env:OLLAMA_CONTEXT_LENGTH)"
Write-Host "The model may be split automatically between GPU VRAM and system RAM."
Write-Host "Press Ctrl+C to stop this server."

& $ollama serve
