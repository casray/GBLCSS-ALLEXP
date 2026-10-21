param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp11_k_sensitivity_20260702",
    [string]$EmbeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2",
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
$kValues = @(3, 5, 10, 20, 50, 100)

if (-not $SummarizeOnly) {
    foreach ($k in $kValues) {
        $output = Join-Path $OutputDir "stageA_knn_k${k}_uniform0p4_seeds_${seedToken}.csv"
        Write-Output "STEP_START phase=MLP11 stage=A method=knn_mlp_softdist_beta05 k=$k seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $EmbeddingRoot `
            --datasets "cora,citeseer,pubmed,coauthorcs" `
            --ssl_methods grace `
            --base_methods "knn_mlp_softdist_beta05" `
            --seeds $Seeds `
            --noise_type uniform `
            --noise_rate 0.4 `
            --noise_scope train `
            --device cuda:0 `
            --split_type percent `
            --train_percent 0.2 `
            --val_percent 0.1 `
            --test_percent 0.7 `
            --config_overrides "{`"gb.matched_k`": $k}" `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP-11 Stage A failed for k=$k (exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE phase=MLP11 stage=A method=knn_mlp_softdist_beta05 k=$k seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

& $python scripts\summarize_mlp11_k_sensitivity.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-11 summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE phase=MLP11 seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
