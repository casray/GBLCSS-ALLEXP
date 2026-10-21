param(
    [string]$Seeds = "1,2,3,4",
    [string]$OutputDir = "runs\mlp9_denoised_teacher_20260702",
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
$methods = "mlp_distill_gcn,mlp_distill_gcn_xh,mlp_distill_softdist_xh,mlp_distill_knn_xh,mlp_distill_softdist_xh_shuffled"
$seedToken = $Seeds.Replace(",", "_")
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SummarizeOnly) {
    foreach ($noiseRate in @(0.4, 0.6)) {
        $noiseToken = ("{0:0.0}" -f $noiseRate).Replace(".", "p")
        $output = Join-Path $OutputDir "mlp9_uniform${noiseToken}_seeds_${seedToken}.csv"
        Write-Output "STEP_START phase=MLP9 noise=uniform$noiseRate methods=$methods seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets cora,citeseer,pubmed `
            --ssl_methods grace `
            --base_methods $methods `
            --seeds $Seeds `
            --noise_type uniform `
            --noise_rate $noiseRate `
            --noise_scope train `
            --device cuda:0 `
            --split_type percent `
            --train_percent 0.2 `
            --val_percent 0.1 `
            --test_percent 0.7 `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "MLP-9 run failed (noise=$noiseRate exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE phase=MLP9 noise=uniform$noiseRate seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

& $python scripts\summarize_mlp9_denoised_teacher.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-9 summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE phase=MLP9 seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
