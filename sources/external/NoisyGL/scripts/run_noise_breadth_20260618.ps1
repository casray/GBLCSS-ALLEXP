$ErrorActionPreference = "Stop"

$python = $env:NOISYGL_ROCM_PYTHON
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python)) {
    throw "NOISYGL_ROCM_PYTHON must point to the ROCm Python executable."
}
$embeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2"
$outputDir = "runs\noise_breadth_20260618"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Output "STEP_START name=$Name time=$([DateTime]::Now.ToString('s'))"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE name=$Name time=$([DateTime]::Now.ToString('s'))"
}

Invoke-Step "coauthorcs_pair0p4_standard_seeds1to4" {
    & $python scripts\run_standard_downstream.py `
        --datasets coauthorcs `
        --methods gcn,nrgnn `
        --seeds 1,2,3,4 `
        --noise_type pair `
        --noise_rate 0.4 `
        --device cuda:0 `
        --output "$outputDir\coauthorcs_pair0p4_standard_seeds1to4.csv"
}

Invoke-Step "coauthorcs_pair0p4_xh_seeds1to4" {
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets coauthorcs `
        --ssl_methods grace `
        --base_methods gcn_xh_ce,frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled `
        --seeds 1,2,3,4 `
        --noise_type pair `
        --noise_rate 0.4 `
        --device cuda:0 `
        --output "$outputDir\coauthorcs_pair0p4_xh_seeds1to4.csv"
}

Invoke-Step "coauthorcs_instance0p4_standard_seeds1to4" {
    & $python scripts\run_standard_downstream.py `
        --datasets coauthorcs `
        --methods gcn,nrgnn `
        --seeds 1,2,3,4 `
        --noise_type instance `
        --noise_rate 0.4 `
        --device cuda:0 `
        --output "$outputDir\coauthorcs_instance0p4_standard_seeds1to4.csv"
}

Invoke-Step "coauthorcs_instance0p4_xh_seeds1to4" {
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets coauthorcs `
        --ssl_methods grace `
        --base_methods gcn_xh_ce,frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled `
        --seeds 1,2,3,4 `
        --noise_type instance `
        --noise_rate 0.4 `
        --device cuda:0 `
        --output "$outputDir\coauthorcs_instance0p4_xh_seeds1to4.csv"
}

foreach ($rateToken in @("0p2", "0p6")) {
    $rate = if ($rateToken -eq "0p2") { "0.2" } else { "0.6" }

    Invoke-Step "coauthorcs_uniform${rateToken}_standard" {
        & $python scripts\run_standard_downstream.py `
            --datasets coauthorcs `
            --methods gcn,nrgnn `
            --seeds 0,1,2,3,4 `
            --noise_type uniform `
            --noise_rate $rate `
            --device cuda:0 `
            --output "$outputDir\coauthorcs_uniform${rateToken}_standard.csv"
    }

    Invoke-Step "coauthorcs_uniform${rateToken}_xh" {
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets coauthorcs `
            --ssl_methods grace `
            --base_methods gcn_xh_ce,frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled `
            --seeds 0,1,2,3,4 `
            --noise_type uniform `
            --noise_rate $rate `
            --device cuda:0 `
            --output "$outputDir\coauthorcs_uniform${rateToken}_xh.csv"
    }
}

Invoke-Step "cora_citeseer_uniform0p5_standard" {
    & $python scripts\run_standard_downstream.py `
        --datasets cora,citeseer `
        --methods gcn,nrgnn `
        --seeds 0,1,2,3,4 `
        --noise_type uniform `
        --noise_rate 0.5 `
        --device cuda:0 `
        --output "$outputDir\cora_citeseer_uniform0p5_standard.csv"
}

Invoke-Step "cora_citeseer_uniform0p5_xh" {
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer `
        --ssl_methods grace `
        --base_methods gcn_xh_ce,frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled `
        --seeds 0,1,2,3,4 `
        --noise_type uniform `
        --noise_rate 0.5 `
        --device cuda:0 `
        --output "$outputDir\cora_citeseer_uniform0p5_xh.csv"
}

Write-Output "ALL_STEPS_DONE time=$([DateTime]::Now.ToString('s'))"
