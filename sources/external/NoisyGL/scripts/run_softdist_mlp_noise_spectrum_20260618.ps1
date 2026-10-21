param(
    [string]$Seeds = "0",
    [string]$OutputDir = "runs\softdist_mlp_noise_spectrum_20260618"
)

$ErrorActionPreference = "Stop"

$python = $env:NOISYGL_ROCM_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python)) {
    throw "NOISYGL_ROCM_PYTHON must point to the ROCm Python executable."
}

$embeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2"
$methods = "mlp_ce,frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$regimes = @(
    @{ Name = "clean0p0"; Type = "clean"; Rate = "0.0" },
    @{ Name = "uniform0p2"; Type = "uniform"; Rate = "0.2" },
    @{ Name = "uniform0p5"; Type = "uniform"; Rate = "0.5" },
    @{ Name = "uniform0p6"; Type = "uniform"; Rate = "0.6" },
    @{ Name = "pair0p4"; Type = "pair"; Rate = "0.4" },
    @{ Name = "instance0p4"; Type = "instance"; Rate = "0.4" }
)

$seedToken = $Seeds.Replace(",", "_")
foreach ($regime in $regimes) {
    $name = $regime.Name
    $noiseType = $regime.Type
    $noiseRate = $regime.Rate
    $output = Join-Path $OutputDir "${name}_seeds_${seedToken}.csv"
    Write-Output "STEP_START regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer `
        --ssl_methods grace `
        --base_methods $methods `
        --seeds $Seeds `
        --noise_type $noiseType `
        --noise_rate $noiseRate `
        --device cuda:0 `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "Noise-spectrum step failed: $name (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
