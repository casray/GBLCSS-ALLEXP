param(
    [ValidateSet("All", "Smoke", "Full", "Summarize")]
    [string]$Stage = "All",
    [string]$OutputDir = "runs\mlp14_graphfree_family_20260703",
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

$datasets = @("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
$methods = @("gsdn", "graphmlp")

$regimes = @(
    @{ Name = "uniform0p2"; NoiseType = "uniform"; NoiseRate = "0.2" },
    @{ Name = "uniform0p4"; NoiseType = "uniform"; NoiseRate = "0.4" },
    @{ Name = "uniform0p6"; NoiseType = "uniform"; NoiseRate = "0.6" },
    @{ Name = "pair0p4"; NoiseType = "pair"; NoiseRate = "0.4" },
    @{ Name = "instance0p4"; NoiseType = "instance"; NoiseRate = "0.4" }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-Summarizer {
    & $python scripts\summarize_mlp14_graphfree_family.py --run_dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-14 summarization failed (exit=$LASTEXITCODE)"
    }
}

function Invoke-Preflight {
    foreach ($dataset in $datasets) {
        foreach ($method in $methods) {
            $configPath = Join-Path "config\$method" "${method}_${dataset}.yaml"
            if (-not (Test-Path -LiteralPath $configPath)) {
                throw "Missing method config for method=$method dataset=$dataset path=$configPath"
            }
        }
    }
}

function Invoke-StandardRun {
    param(
        [string]$Label,
        [string]$NoiseType,
        [string]$NoiseRate,
        [string]$Seeds
    )

    $seedToken = $Seeds.Replace(",", "_")
    $output = Join-Path $OutputDir "${Label}_seeds_${seedToken}.csv"
    if ((Test-Path -LiteralPath $output) -and -not $ForceRerun) {
        Write-Output "STEP_SKIP phase=MLP14 label=$Label seeds=$Seeds output=$output"
        return
    }

    Write-Output "STEP_START phase=MLP14 label=$Label noise_type=$NoiseType noise_rate=$NoiseRate seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_standard_downstream.py `
        --datasets ([string]::Join(",", $datasets)) `
        --methods ([string]::Join(",", $methods)) `
        --seeds $Seeds `
        --noise_type $NoiseType `
        --noise_rate $NoiseRate `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-14 run failed label=$Label seeds=$Seeds (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE phase=MLP14 label=$Label seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

function Invoke-Smoke {
    Invoke-StandardRun -Label "smoke_uniform0p4" -NoiseType "uniform" -NoiseRate "0.4" -Seeds "0"
    Invoke-StandardRun -Label "smoke_uniform0p6" -NoiseType "uniform" -NoiseRate "0.6" -Seeds "0"
    Invoke-Summarizer
}

function Test-SmokeGate {
    $gatePath = Join-Path $OutputDir "mlp14_smoke_gate.csv"
    if (-not (Test-Path -LiteralPath $gatePath)) {
        Invoke-Summarizer
    }
    if (-not (Test-Path -LiteralPath $gatePath)) {
        throw "Smoke gate file not found: $gatePath"
    }

    $rows = Import-Csv -LiteralPath $gatePath
    $failed = @($rows | Where-Object { [int]$_.passes -ne 1 })
    if ($failed.Count -gt 0) {
        Write-Output "SMOKE_GATE_NOT_PASS count=$($failed.Count)"
        $failed | Format-Table method,dataset,noise_type,noise_rate,test,mlp_ce_test,delta_vs_mlp_ce,status_ok,ce_gate,passes -AutoSize
        return $false
    }
    Write-Output "SMOKE_GATE_PASS count=$($rows.Count)"
    return $true
}

function Invoke-Full {
    if (-not (Test-SmokeGate)) {
        throw "Smoke gate did not pass. Full stage is blocked by preregistered gate."
    }

    foreach ($regime in $regimes) {
        $name = [string]$regime.Name
        $noiseType = [string]$regime.NoiseType
        $noiseRate = [string]$regime.NoiseRate
        if ($name -eq "uniform0p4" -or $name -eq "uniform0p6") {
            Invoke-StandardRun -Label "full_${name}" -NoiseType $noiseType -NoiseRate $noiseRate -Seeds "1,2,3,4"
        } else {
            Invoke-StandardRun -Label "full_${name}" -NoiseType $noiseType -NoiseRate $noiseRate -Seeds "0,1,2,3,4"
        }
    }
    Invoke-Summarizer
}

Invoke-Preflight

if ($Stage -eq "Smoke") {
    Invoke-Smoke
} elseif ($Stage -eq "Full") {
    Invoke-Full
} elseif ($Stage -eq "Summarize") {
    Invoke-Summarizer
} else {
    Invoke-Smoke
    Invoke-Full
}

Write-Output "ALL_STEPS_DONE phase=MLP14 stage=$Stage output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
