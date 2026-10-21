param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp_dualrobust_20260701",
    [switch]$SkipLabel,
    [switch]$SkipStructure,
    [switch]$SkipAblation,
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
$methods = "mlp_ce,mlp_small_loss,mlp_coteaching,mlp_jocor,mlp_elr,mlp_sce,frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_shuffled"
$ablationMethods = "mlp_ce,featuregb_mlp_softdist,frozengrace_gb_mlp_hard,frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_shuffled"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SummarizeOnly -and -not $SkipLabel) {
    $labelJobs = @(
        @{ Name = "instance0p4_cora_citeseer"; Datasets = "cora,citeseer"; Type = "instance"; Rate = "0.4" },
        @{ Name = "uniform0p2_coauthorcs"; Datasets = "coauthorcs"; Type = "uniform"; Rate = "0.2" },
        @{ Name = "uniform0p4_coauthorcs"; Datasets = "coauthorcs"; Type = "uniform"; Rate = "0.4" },
        @{ Name = "uniform0p6_coauthorcs"; Datasets = "coauthorcs"; Type = "uniform"; Rate = "0.6" },
        @{ Name = "pair0p4_coauthorcs"; Datasets = "coauthorcs"; Type = "pair"; Rate = "0.4" },
        @{ Name = "instance0p4_coauthorcs"; Datasets = "coauthorcs"; Type = "instance"; Rate = "0.4" }
    )
    foreach ($job in $labelJobs) {
        $output = Join-Path $OutputDir ("label_" + $job.Name + "_seeds_" + $Seeds.Replace(",", "_") + ".csv")
        Write-Output "LABEL_STEP_START name=$($job.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets $job.Datasets `
            --ssl_methods grace `
            --base_methods $methods `
            --seeds $Seeds `
            --noise_type $job.Type `
            --noise_rate $job.Rate `
            --noise_scope train `
            --device cuda:0 `
            --split_type percent `
            --train_percent 0.2 `
            --val_percent 0.1 `
            --test_percent 0.7 `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP dual-robust label step failed: $($job.Name) (exit=$LASTEXITCODE)"
        }
        Write-Output "LABEL_STEP_DONE name=$($job.Name) time=$([DateTime]::Now.ToString('s'))"
    }
}

if (-not $SummarizeOnly -and -not $SkipStructure) {
    $structureOutput = Join-Path $OutputDir ("structure_clean_seeds_" + $Seeds.Replace(",", "_") + ".csv")
    Write-Output "STRUCTURE_STEP_START seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_mlp_structure_robustness.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer,coauthorcs `
        --methods gcn,gin,frozengrace_gb_mlp_softdist `
        --seeds $Seeds `
        --label_noise_type clean `
        --label_noise_rate 0.0 `
        --noise_scope train `
        --structure_modes edge_drop,isolated `
        --structure_ps 0.25,0.5,0.75 `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output $structureOutput
    if ($LASTEXITCODE -ne 0) {
        throw "MLP dual-robust structure step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "STRUCTURE_STEP_DONE time=$([DateTime]::Now.ToString('s'))"
}

if (-not $SummarizeOnly -and -not $SkipAblation) {
    $ablationOutput = Join-Path $OutputDir ("ablation_uniform0p4_seeds_" + $Seeds.Replace(",", "_") + ".csv")
    Write-Output "ABLATION_STEP_START seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer,coauthorcs `
        --ssl_methods grace `
        --base_methods $ablationMethods `
        --seeds $Seeds `
        --noise_type uniform `
        --noise_rate 0.4 `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output $ablationOutput
    if ($LASTEXITCODE -ne 0) {
        throw "MLP dual-robust ablation step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "ABLATION_STEP_DONE time=$([DateTime]::Now.ToString('s'))"
}

Write-Output "SUMMARY_STEP_START time=$([DateTime]::Now.ToString('s'))"
& $python scripts\summarize_mlp_dualrobust.py `
    --run_dir $OutputDir `
    --legacy_label_dir runs\mlp_noiserobust_20260701 `
    --require_complete
if ($LASTEXITCODE -ne 0) {
    throw "MLP dual-robust summary failed (exit=$LASTEXITCODE)"
}
Write-Output "ALL_STEPS_DONE seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
