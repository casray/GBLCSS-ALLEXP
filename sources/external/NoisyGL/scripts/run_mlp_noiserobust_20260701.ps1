param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp_noiserobust_20260701"
)

$ErrorActionPreference = "Stop"

$python = $env:NOISYGL_ROCM_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python)) {
    $python = "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "ROCm Python executable not found: $python"
}

$embeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2"
$methods = "mlp_ce,mlp_small_loss,mlp_coteaching,mlp_jocor,mlp_elr,mlp_sce,frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$regimes = @(
    @{ Name = "uniform0p2"; Type = "uniform"; Rate = "0.2" },
    @{ Name = "uniform0p4"; Type = "uniform"; Rate = "0.4" },
    @{ Name = "uniform0p6"; Type = "uniform"; Rate = "0.6" },
    @{ Name = "pair0p4"; Type = "pair"; Rate = "0.4" }
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
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP noise-robust step failed: $name (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
