param(
    [ValidateSet("All", "A", "B", "Summarize")]
    [string]$Stage = "All",
    [string]$OutputDir = "runs\mlp13_ballparam_sensitivity_20260703",
    [string]$EmbeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2",
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"

$python = $env:NOISYGL_ROCM_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python)) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $python = Get-ChildItem -LiteralPath $desktop -Directory -Filter ".venv-rocm" -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "Scripts\python.exe" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "ROCm Python executable not found. Set NOISYGL_ROCM_PYTHON or keep .venv-rocm under Desktop."
}

$method = "frozengrace_gb_mlp_softdist_beta05"
$datasets = @("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
$maxBallsValues = @(32, 64, 128)
$minBallSizeValues = @(16, 32, 64)
$splitImprovementValues = @(0.01, 0.05, 0.1)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Convert-SplitToken {
    param([double]$Value)
    return ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:g}", $Value)).Replace(".", "p")
}

function Invoke-Summarizer {
    & $python scripts\summarize_mlp13_ballparam_sensitivity.py --run_dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-13 summarization failed (exit=$LASTEXITCODE)"
    }
}

function Invoke-Preflight {
    foreach ($dataset in $datasets) {
        $configPath = Join-Path "config\$method" "${method}_${dataset}.yaml"
        if (-not (Test-Path -LiteralPath $configPath)) {
            throw "Missing method config for dataset=$dataset path=$configPath"
        }
    }

    $datasetCsv = [string]::Join(",", $datasets)
    $preflightCode = @"
from pathlib import Path
from scripts.linear_probe_embeddings import embedding_path_for

root = Path(r'''$EmbeddingRoot''')
datasets = '$datasetCsv'.split(',')
missing = []
for dataset in datasets:
    path = embedding_path_for(root, dataset, 'grace')
    if not path.exists():
        missing.append((dataset, str(path)))
    else:
        print(f'EMBEDDING_OK dataset={dataset} path={path}')
if missing:
    for dataset, path in missing:
        print(f'EMBEDDING_MISSING dataset={dataset} path={path}')
    raise SystemExit(2)
"@
    & $python -c $preflightCode
    if ($LASTEXITCODE -ne 0) {
        throw "Embedding preflight failed (exit=$LASTEXITCODE). Do not silently shrink the dataset matrix."
    }
}

function Invoke-Downstream {
    param(
        [string]$StageName,
        [int]$MaxBalls,
        [int]$MinBallSize,
        [double]$SplitImprovement,
        [string]$Seeds
    )

    $siToken = Convert-SplitToken -Value $SplitImprovement
    $seedToken = $Seeds.Replace(",", "_")
    $output = Join-Path $OutputDir "${StageName}_mb${MaxBalls}_min${MinBallSize}_si${siToken}_seeds_${seedToken}.csv"
    if ((Test-Path -LiteralPath $output) -and -not $ForceRerun) {
        Write-Output "STEP_SKIP phase=MLP13 stage=$StageName max_balls=$MaxBalls min_ball_size=$MinBallSize split_improvement=$SplitImprovement seeds=$Seeds output=$output"
        return
    }

    Write-Output "STEP_START phase=MLP13 stage=$StageName max_balls=$MaxBalls min_ball_size=$MinBallSize split_improvement=$SplitImprovement seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $EmbeddingRoot `
        --datasets ([string]::Join(",", $datasets)) `
        --ssl_methods grace `
        --base_methods $method `
        --seeds $Seeds `
        --noise_type uniform `
        --noise_rate 0.4 `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --gb_max_balls $MaxBalls `
        --gb_min_ball_size $MinBallSize `
        --gb_split_improvement $SplitImprovement `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-13 downstream failed stage=$StageName max_balls=$MaxBalls min_ball_size=$MinBallSize split_improvement=$SplitImprovement seeds=$Seeds (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE phase=MLP13 stage=$StageName max_balls=$MaxBalls min_ball_size=$MinBallSize split_improvement=$SplitImprovement seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

function Invoke-StageA {
    foreach ($maxBalls in $maxBallsValues) {
        foreach ($minBallSize in $minBallSizeValues) {
            foreach ($splitImprovement in $splitImprovementValues) {
                Invoke-Downstream -StageName "stageA" -MaxBalls $maxBalls -MinBallSize $minBallSize -SplitImprovement $splitImprovement -Seeds "0"
            }
        }
    }
    Invoke-Summarizer
}

function Invoke-StageB {
    $selectedPath = Join-Path $OutputDir "mlp13_stageA_selected_configs.json"
    if (-not (Test-Path -LiteralPath $selectedPath)) {
        Invoke-Summarizer
    }
    if (-not (Test-Path -LiteralPath $selectedPath)) {
        throw "Selected config file not found after summarization: $selectedPath"
    }
    $selected = Get-Content -LiteralPath $selectedPath -Raw | ConvertFrom-Json
    foreach ($config in $selected) {
        Invoke-Downstream `
            -StageName "stageB" `
            -MaxBalls ([int]$config.gb_max_balls) `
            -MinBallSize ([int]$config.gb_min_ball_size) `
            -SplitImprovement ([double]$config.gb_split_improvement) `
            -Seeds "1,2,3,4"
    }
    Invoke-Summarizer
}

Invoke-Preflight

if ($Stage -eq "A") {
    Invoke-StageA
} elseif ($Stage -eq "B") {
    Invoke-StageB
} elseif ($Stage -eq "Summarize") {
    Invoke-Summarizer
} else {
    Invoke-StageA
    Invoke-StageB
}

Write-Output "ALL_STEPS_DONE phase=MLP13 stage=$Stage output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
