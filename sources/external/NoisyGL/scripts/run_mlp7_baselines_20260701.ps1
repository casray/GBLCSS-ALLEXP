param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp7_baselines_20260701",
    [switch]$SkipCT,
    [switch]$SkipCheckpointScan,
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

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$seedToken = $Seeds.Replace(",", "_")

if (-not $SummarizeOnly -and -not $SkipCT) {
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
        $output = Join-Path $OutputDir "main_ct_${name}_seeds_${seedToken}.csv"
        Write-Output "STEP_START table=main_ct regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_standard_downstream.py `
            --datasets cora,citeseer,pubmed `
            --methods mlp_ct `
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
            throw "MLP-7 CT step failed: $name (exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE table=main_ct regime=$name seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

if (-not $SummarizeOnly -and -not $SkipCheckpointScan) {
    $checkpointOutput = Join-Path $OutputDir "grace_encoder_checkpoint_scan.csv"
    & $python scripts\check_grace_encoder_checkpoints.py --output $checkpointOutput
    if ($LASTEXITCODE -ne 0) {
        throw "GRACE encoder checkpoint scan failed (exit=$LASTEXITCODE)"
    }
}

& $python scripts\summarize_mlp7_baselines.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-7 summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
