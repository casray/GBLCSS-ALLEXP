param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp_maintable_20260701",
    [switch]$SkipMain,
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
$mainMethods = "mlp_ce,mlp_small_loss,mlp_coteaching,mlp_jocor,mlp_elr,mlp_sce,mlp_fine,mlp_dividemix_lite,frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled"
$labelrateMethods = "mlp_small_loss,mlp_jocor,mlp_fine,frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$seedToken = $Seeds.Replace(",", "_")

if (-not $SummarizeOnly -and -not $SkipMain) {
    $regimes = @(
        @{ Name = "uniform0p2"; Type = "uniform"; Rate = "0.2" },
        @{ Name = "uniform0p4"; Type = "uniform"; Rate = "0.4" },
        @{ Name = "uniform0p6"; Type = "uniform"; Rate = "0.6" },
        @{ Name = "pair0p4"; Type = "pair"; Rate = "0.4" },
        @{ Name = "instance0p4"; Type = "instance"; Rate = "0.4" }
    )

    foreach ($regime in $regimes) {
        $name = $regime.Name
        $noiseType = $regime.Type
        $noiseRate = $regime.Rate
        $output = Join-Path $OutputDir "main_${name}_seeds_${seedToken}.csv"
        Write-Output "STEP_START table=main regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets cora,citeseer,pubmed `
            --ssl_methods grace `
            --base_methods $mainMethods `
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
            throw "MLP main-table step failed: $name (exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE table=main regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

if (-not $SummarizeOnly -and -not $SkipLabelrate) {
    $labelrates = @(
        @{ Name = "0p05"; Train = "0.05"; Test = "0.85" },
        @{ Name = "0p65"; Train = "0.65"; Test = "0.25" }
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
            throw "MLP label-rate step failed: train=$trainPercent (exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE table=labelrate train=$trainPercent seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

& $python scripts\summarize_mlp_maintable.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP main-table summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
