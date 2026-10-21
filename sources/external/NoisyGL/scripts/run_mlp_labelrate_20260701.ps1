param(
    [string]$Seeds = "0",
    [string]$Datasets = "cora,citeseer,pubmed",
    [string]$OutputDir = "runs\mlp_labelrate_20260701",
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
$methods = "mlp_ce,mlp_small_loss,mlp_jocor,frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ratios = @(
    @{ Alias = "0p05"; Train = "0.05"; Val = "0.10"; Test = "0.85" },
    @{ Alias = "0p10"; Train = "0.10"; Val = "0.10"; Test = "0.80" },
    @{ Alias = "0p20"; Train = "0.20"; Val = "0.10"; Test = "0.70" },
    @{ Alias = "0p35"; Train = "0.35"; Val = "0.10"; Test = "0.55" },
    @{ Alias = "0p50"; Train = "0.50"; Val = "0.10"; Test = "0.40" },
    @{ Alias = "0p65"; Train = "0.65"; Val = "0.10"; Test = "0.25" }
)

if (-not $SummarizeOnly) {
    $seedToken = $Seeds.Replace(",", "_")
    foreach ($ratio in $ratios) {
        $alias = $ratio.Alias
        $output = Join-Path $OutputDir "labelrate_train${alias}_seeds_${seedToken}.csv"
        Write-Output "LABELRATE_STEP_START train=$($ratio.Train) val=$($ratio.Val) test=$($ratio.Test) datasets=$Datasets seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
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
            --train_percent $ratio.Train `
            --val_percent $ratio.Val `
            --test_percent $ratio.Test `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP label-rate step failed: train=$($ratio.Train) (exit=$LASTEXITCODE)"
        }
        Write-Output "LABELRATE_STEP_DONE train=$($ratio.Train) time=$([DateTime]::Now.ToString('s'))"
    }
}

Write-Output "SUMMARY_STEP_START time=$([DateTime]::Now.ToString('s'))"
& $python scripts\summarize_mlp_labelrate.py `
    --run_dir $OutputDir `
    --datasets $Datasets `
    --seeds $Seeds `
    --require_complete
if ($LASTEXITCODE -ne 0) {
    throw "MLP label-rate summary failed (exit=$LASTEXITCODE)"
}
Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
