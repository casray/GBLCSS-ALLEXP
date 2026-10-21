param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp6_fairness_labelrate_20260701",
    [switch]$SkipFairness,
    [switch]$SkipLabelrate,
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
$fairnessMethods = "mlp_xh_ce,mlp_small_loss_xh,mlp_jocor_xh"
$fairnessLowMethods = "mlp_small_loss_xh,mlp_jocor_xh"
$labelrateMethods = "frozengrace_gb_mlp_softdist_beta05,mlp_small_loss,mlp_jocor,mlp_fine,frozengrace_gb_mlp_softdist_beta05_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$seedToken = $Seeds.Replace(",", "_")

if (-not $SummarizeOnly -and -not $SkipFairness) {
    $output = Join-Path $OutputDir "fairness_uniform0p4_train0p20_seeds_${seedToken}.csv"
    Write-Output "STEP_START table=fairness train=0.20 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer,pubmed `
        --ssl_methods grace `
        --base_methods $fairnessMethods `
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
        throw "MLP-6 fairness train0.20 step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE table=fairness train=0.20 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"

    $lowOutput = Join-Path $OutputDir "fairness_uniform0p4_train0p05_seeds_${seedToken}.csv"
    Write-Output "STEP_START table=fairness_low train=0.05 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer `
        --ssl_methods grace `
        --base_methods $fairnessLowMethods `
        --seeds $Seeds `
        --noise_type uniform `
        --noise_rate 0.4 `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.05 `
        --val_percent 0.1 `
        --test_percent 0.85 `
        --output $lowOutput
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-6 fairness train0.05 step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE table=fairness_low train=0.05 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

if (-not $SummarizeOnly -and -not $SkipLabelrate) {
    $labelrates = @(
        @{ Name = "0p10"; Train = "0.10"; Test = "0.80" },
        @{ Name = "0p20"; Train = "0.20"; Test = "0.70" },
        @{ Name = "0p35"; Train = "0.35"; Test = "0.55" },
        @{ Name = "0p50"; Train = "0.50"; Test = "0.40" }
    )

    foreach ($rate in $labelrates) {
        $name = $rate.Name
        $trainPercent = $rate.Train
        $testPercent = $rate.Test
        $output = Join-Path $OutputDir "labelrate_train${name}_seeds_${seedToken}.csv"
        Write-Output "STEP_START table=labelrate train=$trainPercent seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets cora,citeseer `
            --ssl_methods grace `
            --base_methods $labelrateMethods `
            --seeds $Seeds `
            --noise_type uniform `
            --noise_rate 0.4 `
            --noise_scope train `
            --device cuda:0 `
            --split_type percent `
            --train_percent $trainPercent `
            --val_percent 0.1 `
            --test_percent $testPercent `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP-6 label-rate step failed: train=$trainPercent (exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE table=labelrate train=$trainPercent seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

& $python scripts\summarize_mlp6_fairness_labelrate.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-6 summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
