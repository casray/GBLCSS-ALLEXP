param(
    [string]$Seeds = "0",
    [string]$Datasets = "cora,citeseer",
    [string]$OutputDir = "runs\mlp_weighting_revalidate_20260701",
    [switch]$SummarizeOnly
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
$methods = "frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_ellipsoid_beta05,frozengrace_gb_mlp_softdist_gmmresp_beta05,frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SummarizeOnly) {
    $output = Join-Path $OutputDir ("weighting_uniform0p4_seeds_" + $Seeds.Replace(",", "_") + ".csv")
    Write-Output "WEIGHTING_STEP_START datasets=$Datasets seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets $Datasets `
        --ssl_methods grace `
        --base_methods $methods `
        --seeds $Seeds `
        --noise_type uniform `
        --noise_rate 0.4 `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP weighting revalidation step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "WEIGHTING_STEP_DONE time=$([DateTime]::Now.ToString('s'))"
}

Write-Output "SUMMARY_STEP_START time=$([DateTime]::Now.ToString('s'))"
& $python scripts\summarize_mlp_weighting_revalidate.py `
    --run_dir $OutputDir `
    --datasets $Datasets `
    --seeds $Seeds `
    --require_complete
if ($LASTEXITCODE -ne 0) {
    throw "MLP weighting revalidation summary failed (exit=$LASTEXITCODE)"
}
Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
