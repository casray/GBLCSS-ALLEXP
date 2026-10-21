param(
    [string]$Seeds = "0,1,2,3,4",
    [string]$P1Seeds = "0",
    [string]$OutputDir = "runs\mlp8_replacement_expansion_20260702",
    [switch]$SkipP0,
    [switch]$SkipP1,
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
$p0Methods = "knn_mlp_softdist_beta05,kmeans_mlp_softdist_beta05"
$p1ExpansionMethods = "frozengrace_gb_mlp_softdist_beta05_expand,frozengrace_gb_mlp_softdist_beta05_expand_shuffled,knn_mlp_softdist_beta05_expand"
$p1DistillMethods = "mlp_distill_gcn"
$seedToken = $Seeds.Replace(",", "_")
$p1SeedToken = $P1Seeds.Replace(",", "_")
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SummarizeOnly -and -not $SkipP0) {
    $output = Join-Path $OutputDir "p0_replacement_uniform0p4_seeds_${seedToken}.csv"
    Write-Output "STEP_START phase=P0 replacement=knn_kmeans noise=uniform0.4 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $embeddingRoot `
        --datasets cora,citeseer,pubmed `
        --ssl_methods grace `
        --base_methods $p0Methods `
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
        throw "MLP-8 P0 replacement step failed (exit=$LASTEXITCODE)"
    }
    Write-Output "STEP_DONE phase=P0 replacement=knn_kmeans seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
}

if (-not $SummarizeOnly -and -not $SkipP1) {
    foreach ($noiseRate in @(0.4, 0.6)) {
        $noiseToken = ("{0:0.0}" -f $noiseRate).Replace(".", "p")
        foreach ($lambdaText in @("0.1", "0.3", "1.0")) {
            $lambdaValue = [double]$lambdaText
            $lambdaToken = $lambdaText.Replace(".", "p")
            $output = Join-Path $OutputDir "p1_expansion_uniform${noiseToken}_lambda${lambdaToken}_seeds_${p1SeedToken}.csv"
            Write-Output "STEP_START phase=P1 expansion noise=uniform$noiseRate lambda=$lambdaText seeds=$P1Seeds time=$([DateTime]::Now.ToString('s'))"
            & $python scripts\run_ssl_swap_downstream.py `
                --embedding_root $embeddingRoot `
                --datasets cora,citeseer,pubmed `
                --ssl_methods grace `
                --base_methods $p1ExpansionMethods `
                --seeds $P1Seeds `
                --noise_type uniform `
                --noise_rate $noiseRate `
                --noise_scope train `
                --device cuda:0 `
                --split_type percent `
                --train_percent 0.2 `
                --val_percent 0.1 `
                --test_percent 0.7 `
                --gb_lambda_u $lambdaValue `
                --output $output
            if ($LASTEXITCODE -ne 0) {
                throw "MLP-8 P1 expansion step failed (noise=$noiseRate lambda=$lambdaText exit=$LASTEXITCODE)"
            }
            Write-Output "STEP_DONE phase=P1 expansion noise=uniform$noiseRate lambda=$lambdaText seeds=$P1Seeds time=$([DateTime]::Now.ToString('s'))"
        }

        $output = Join-Path $OutputDir "p1_glnn_uniform${noiseToken}_seeds_${p1SeedToken}.csv"
        Write-Output "STEP_START phase=P1 glnn noise=uniform$noiseRate seeds=$P1Seeds time=$([DateTime]::Now.ToString('s'))"
        & $python scripts\run_ssl_swap_downstream.py `
            --embedding_root $embeddingRoot `
            --datasets cora,citeseer,pubmed `
            --ssl_methods grace `
            --base_methods $p1DistillMethods `
            --seeds $P1Seeds `
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
            throw "MLP-8 P1 GLNN step failed (noise=$noiseRate exit=$LASTEXITCODE)"
        }
        Write-Output "STEP_DONE phase=P1 glnn noise=uniform$noiseRate seeds=$P1Seeds time=$([DateTime]::Now.ToString('s'))"
    }
}

& $python scripts\summarize_mlp8_replacement_expansion.py --run_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-8 summarization failed (exit=$LASTEXITCODE)"
}

Write-Output "ALL_STEPS_DONE phase=MLP8 seeds=$Seeds output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
