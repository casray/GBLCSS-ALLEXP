param(
    [string]$OutputDir = "runs\softdist_mlp_efficiency_20260618"
)

$ErrorActionPreference = "Stop"

$python = $env:NOISYGL_ROCM_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python)) {
    throw "NOISYGL_ROCM_PYTHON must point to the ROCm Python executable."
}

$embeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2"
$methods = "mlp_ce,softdist_mlp,gcn,softdist_gcn,gcn_xh_ce,softdist_xh,nrgnn"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($dataset in @("cora", "citeseer")) {
    foreach ($seed in 0..4) {
        $output = Join-Path $OutputDir "${dataset}_seed${seed}.csv"
        Write-Output "STEP_START dataset=$dataset seed=$seed time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\measure_efficiency.py `
            --dataset $dataset `
            --methods $methods `
            --seed $seed `
            --embedding_root $embeddingRoot `
            --ssl_method grace `
            --noise_type uniform `
            --noise_rate 0.4 `
            --device cuda:0 `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "Efficiency process failed: dataset=$dataset seed=$seed exit=$LASTEXITCODE"
        }
        $rows = @(Import-Csv -LiteralPath $output)
        $failed = @($rows | Where-Object { $_.status -ne "ok" })
        if ($rows.Count -ne 7 -or $failed.Count -ne 0) {
            throw "Efficiency output invalid: dataset=$dataset seed=$seed rows=$($rows.Count) failed=$($failed.Count)"
        }
        Write-Output "STEP_DONE dataset=$dataset seed=$seed time=$([DateTime]::Now.ToString('s'))"
    }
}

Write-Output "ALL_STEPS_DONE time=$([DateTime]::Now.ToString('s'))"
