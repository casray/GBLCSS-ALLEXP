param(
    [string]$Seeds = "0",
    [string]$Datasets = "cora,citeseer,pubmed",
    [string]$OutputDir = "runs\mlp_beta_hardness_20260701",
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
$methods = "frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta1,frozengrace_gb_mlp_softdist_beta1_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SummarizeOnly) {
    $jobs = @(
        @{ Name = "uniform0p2"; Rate = "0.2" },
        @{ Name = "uniform0p4"; Rate = "0.4" }
    )
    foreach ($job in $jobs) {
        $output = Join-Path $OutputDir ("beta_" + $job.Name + "_seeds_" + $Seeds.Replace(",", "_") + ".csv")
        Write-Output "BETA_STEP_START name=$($job.Name) datasets=$Datasets seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets $Datasets `
            --ssl_methods grace `
            --base_methods $methods `
            --seeds $Seeds `
            --noise_type uniform `
            --noise_rate $job.Rate `
            --noise_scope train `
            --device cuda:0 `
            --split_type percent `
            --train_percent 0.2 `
            --val_percent 0.1 `
            --test_percent 0.7 `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP beta hardness step failed: $($job.Name) (exit=$LASTEXITCODE)"
        }
        Write-Output "BETA_STEP_DONE name=$($job.Name) time=$([DateTime]::Now.ToString('s'))"
    }
}

Write-Output "SUMMARY_STEP_START time=$([DateTime]::Now.ToString('s'))"
& $python scripts\summarize_mlp_beta_hardness.py `
    --run_dir $OutputDir `
    --datasets $Datasets `
    --seeds $Seeds `
    --require_complete
if ($LASTEXITCODE -ne 0) {
    throw "MLP beta hardness summary failed (exit=$LASTEXITCODE)"
}
Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
