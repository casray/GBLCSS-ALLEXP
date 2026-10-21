param(
    [ValidateSet("Smoke", "Full", "RevisedA", "RevisedB", "Summarize")]
    [string]$Stage = "Smoke",
    [string]$Seeds = "0",
    [string]$OutputDir = "runs\mlp10_baseline_refresh_20260702",
    [string]$EmbeddingRoot = "..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2"
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
$regimes = @(
    @{ Name = "uniform0p2"; Type = "uniform"; Rate = "0.2" },
    @{ Name = "uniform0p4"; Type = "uniform"; Rate = "0.4" },
    @{ Name = "uniform0p6"; Type = "uniform"; Rate = "0.6" },
    @{ Name = "pair0p4"; Type = "pair"; Rate = "0.4" },
    @{ Name = "instance0p4"; Type = "instance"; Rate = "0.4" }
)

function Run-SSL {
    param(
        [string]$Datasets,
        [string]$Methods,
        [string]$NoiseType,
        [string]$NoiseRate,
        [string]$SeedsValue,
        [string]$Output,
        [string]$Overrides = "{}"
    )
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $EmbeddingRoot `
        --datasets $Datasets `
        --ssl_methods grace `
        --base_methods $Methods `
        --seeds $SeedsValue `
        --noise_type $NoiseType `
        --noise_rate $NoiseRate `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --config_overrides $Overrides `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-10 run failed: output=$Output exit=$LASTEXITCODE"
    }
}

function Summarize-MLP10 {
    & $python scripts\summarize_mlp10_baseline_refresh.py --run_dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-10 summarization failed: exit=$LASTEXITCODE"
    }
}

if ($Stage -eq "Smoke") {
    foreach ($regime in $regimes) {
        foreach ($tau in @("0.5", "1.0", "2.0")) {
            $tauToken = $tau.Replace(".", "p")
            $output = Join-Path $OutputDir "a1_logitclip_tau${tauToken}_$($regime.Name)_seed0.csv"
            Write-Output "STEP_START stage=Smoke part=A1 method=mlp_logitclip tau=$tau regime=$($regime.Name) time=$([DateTime]::Now.ToString('s'))"
            Run-SSL -Datasets "cora,citeseer,pubmed" -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
        }
        $output = Join-Path $OutputDir "a1_anl_$($regime.Name)_seed0.csv"
        Write-Output "STEP_START stage=Smoke part=A1 method=mlp_anl regime=$($regime.Name) time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "cora,citeseer,pubmed" -Methods "mlp_anl" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output
    }
    foreach ($regime in $regimes) {
        foreach ($tau in @("0.5", "1.0", "2.0")) {
            $tauToken = $tau.Replace(".", "p")
            $output = Join-Path $OutputDir "a2_coauthorcs_logitclip_tau${tauToken}_$($regime.Name)_seed0.csv"
            Write-Output "STEP_START stage=Smoke part=A2 method=mlp_logitclip tau=$tau regime=$($regime.Name) time=$([DateTime]::Now.ToString('s'))"
            Run-SSL -Datasets "coauthorcs" -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
        }
        $output = Join-Path $OutputDir "a2_coauthorcs_core_$($regime.Name)_seed0.csv"
        $methods = "frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled,mlp_fine,mlp_dividemix_lite,mlp_ct,mlp_anl"
        Write-Output "STEP_START stage=Smoke part=A2 methods=core regime=$($regime.Name) time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output
    }
    Summarize-MLP10
}
elseif ($Stage -eq "Full") {
    $selectionPath = Join-Path $OutputDir "mlp10_selected_logitclip_tau.csv"
    if (-not (Test-Path -LiteralPath $selectionPath)) {
        throw "Missing selected tau file. Run Smoke first: $selectionPath"
    }
    $tauRows = Import-Csv -LiteralPath $selectionPath
    foreach ($regime in $regimes) {
        $output = Join-Path $OutputDir "b1_anl_$($regime.Name)_seeds_${seedToken}.csv"
        Write-Output "STEP_START stage=Full part=B1 method=mlp_anl regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "cora,citeseer,pubmed" -Methods "mlp_anl" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output
        foreach ($dataset in @("cora", "citeseer", "pubmed")) {
            $tau = ($tauRows | Where-Object { $_.dataset -eq $dataset }).selected_tau
            if ([string]::IsNullOrWhiteSpace($tau)) { throw "Missing selected tau for $dataset" }
            $tauToken = $tau.Replace(".", "p")
            $output = Join-Path $OutputDir "b1_logitclip_${dataset}_tau${tauToken}_$($regime.Name)_seeds_${seedToken}.csv"
            Write-Output "STEP_START stage=Full part=B1 method=mlp_logitclip dataset=$dataset tau=$tau regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
            Run-SSL -Datasets $dataset -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
        }
        $tau = ($tauRows | Where-Object { $_.dataset -eq "coauthorcs" }).selected_tau
        if ([string]::IsNullOrWhiteSpace($tau)) { throw "Missing selected tau for coauthorcs" }
        $tauToken = $tau.Replace(".", "p")
        $output = Join-Path $OutputDir "b2_coauthorcs_logitclip_tau${tauToken}_$($regime.Name)_seeds_${seedToken}.csv"
        Write-Output "STEP_START stage=Full part=B2 method=mlp_logitclip tau=$tau regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"

        $output = Join-Path $OutputDir "b2_coauthorcs_core_$($regime.Name)_seeds_${seedToken}.csv"
        $methods = "frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled,mlp_fine,mlp_dividemix_lite,mlp_ct,mlp_anl"
        Write-Output "STEP_START stage=Full part=B2 methods=core regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output
    }
    Summarize-MLP10
}
elseif ($Stage -eq "RevisedA") {
    $selectionPath = Join-Path $OutputDir "mlp10_selected_logitclip_tau.csv"
    if (-not (Test-Path -LiteralPath $selectionPath)) {
        throw "Missing selected tau file. Run Smoke first: $selectionPath"
    }
    $tauRows = Import-Csv -LiteralPath $selectionPath

    $output = Join-Path $OutputDir "a4_anl_citeseer_uniform0p4_seeds_${seedToken}.csv"
    Write-Output "STEP_START stage=RevisedA part=A4 method=mlp_anl dataset=citeseer regime=uniform0p4 seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
    Run-SSL -Datasets "citeseer" -Methods "mlp_anl" -NoiseType "uniform" -NoiseRate "0.4" -SeedsValue $Seeds -Output $output

    foreach ($regime in $regimes) {
        foreach ($dataset in @("cora", "citeseer", "pubmed")) {
            $tau = ($tauRows | Where-Object { $_.dataset -eq $dataset }).selected_tau
            if ([string]::IsNullOrWhiteSpace($tau)) { throw "Missing selected tau for $dataset" }
            $tauToken = $tau.Replace(".", "p")
            $output = Join-Path $OutputDir "b1a_logitclip_${dataset}_tau${tauToken}_$($regime.Name)_seeds_${seedToken}.csv"
            Write-Output "STEP_START stage=RevisedA part=B1a method=mlp_logitclip dataset=$dataset tau=$tau regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
            Run-SSL -Datasets $dataset -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
        }

        $tau = ($tauRows | Where-Object { $_.dataset -eq "coauthorcs" }).selected_tau
        if ([string]::IsNullOrWhiteSpace($tau)) { throw "Missing selected tau for coauthorcs" }
        $tauToken = $tau.Replace(".", "p")
        $output = Join-Path $OutputDir "b2a_coauthorcs_logitclip_tau${tauToken}_$($regime.Name)_seeds_${seedToken}.csv"
        Write-Output "STEP_START stage=RevisedA part=B2a method=mlp_logitclip tau=$tau regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"

        $output = Join-Path $OutputDir "b2a_coauthorcs_core_no_anl_$($regime.Name)_seeds_${seedToken}.csv"
        $methods = "frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled,mlp_fine,mlp_dividemix_lite,mlp_ct"
        Write-Output "STEP_START stage=RevisedA part=B2a methods=core_no_anl regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output
    }
    Summarize-MLP10
}
elseif ($Stage -eq "RevisedB") {
    foreach ($regime in $regimes) {
        $datasets = "cora,pubmed"
        if (-not ($regime.Type -eq "uniform" -and $regime.Rate -eq "0.4")) {
            $datasets = "cora,citeseer,pubmed"
        }
        $output = Join-Path $OutputDir "b1b_anl_$($regime.Name)_seeds_${seedToken}.csv"
        Write-Output "STEP_START stage=RevisedB part=B1b method=mlp_anl datasets=$datasets regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets $datasets -Methods "mlp_anl" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output

        $output = Join-Path $OutputDir "b2b_coauthorcs_anl_$($regime.Name)_seeds_${seedToken}.csv"
        Write-Output "STEP_START stage=RevisedB part=B2b method=mlp_anl regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Datasets "coauthorcs" -Methods "mlp_anl" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output
    }
    Summarize-MLP10
}
else {
    Summarize-MLP10
}
