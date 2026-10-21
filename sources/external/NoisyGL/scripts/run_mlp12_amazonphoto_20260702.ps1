param(
    [ValidateSet("D0", "Smoke", "Full", "Summarize")]
    [string]$Stage = "D0",
    [string]$Seeds = "0,1,2,3,4",
    [string]$OutputDir = "runs\mlp12_amazonphoto_20260702",
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
        [string]$Methods,
        [string]$NoiseType,
        [string]$NoiseRate,
        [string]$SeedsValue,
        [string]$Output,
        [string]$Overrides = "{}"
    )
    & $python scripts\run_ssl_swap_downstream.py `
        --embedding_root $EmbeddingRoot `
        --datasets amazonpho `
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
        throw "MLP-12 run failed: output=$Output exit=$LASTEXITCODE"
    }
}

function Summarize-MLP12 {
    & $python scripts\summarize_mlp12_amazonphoto.py --run_dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-12 summarization failed: exit=$LASTEXITCODE"
    }
}

if ($Stage -eq "D0") {
    $d0Dir = Join-Path $OutputDir "d0"
    New-Item -ItemType Directory -Force -Path $d0Dir | Out-Null
    Write-Output "STEP_START phase=MLP12 stage=D0 dataset=amazonpho time=$([DateTime]::Now.ToString('s'))"
    & $python scripts\diagnose_biggraph_orthogonality.py `
        --embedding_root $EmbeddingRoot `
        --datasets amazonpho `
        --seeds 0 `
        --ssl_method grace `
        --method_config gcn_xh_ce `
        --noise_type uniform `
        --noise_rate 0.4 `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --output_dir $d0Dir
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-12 D0 failed: exit=$LASTEXITCODE"
    }
    Summarize-MLP12
}
elseif ($Stage -eq "Smoke") {
    foreach ($regime in $regimes) {
        $output = Join-Path $OutputDir "a1_smoke_$($regime.Name)_seed0.csv"
        $methods = "frozengrace_gb_mlp_softdist_beta05,mlp_ce,frozengrace_gb_mlp_softdist_beta05_shuffled"
        Write-Output "STEP_START phase=MLP12 stage=A1 methods=smoke regime=$($regime.Name) time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output
    }
    Summarize-MLP12
}
elseif ($Stage -eq "Full") {
    foreach ($regime in $regimes) {
        foreach ($tau in @("0.5", "1.0", "2.0")) {
            $tauToken = $tau.Replace(".", "p")
            $output = Join-Path $OutputDir "b1_logitclip_tau${tauToken}_$($regime.Name)_seed0.csv"
            Write-Output "STEP_START phase=MLP12 stage=B1 method=mlp_logitclip tau=$tau regime=$($regime.Name) seed=0 time=$([DateTime]::Now.ToString('s'))"
            Run-SSL -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "0" -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
        }
    }
    Summarize-MLP12
    $tauPath = Join-Path $OutputDir "mlp12_selected_logitclip_tau.csv"
    if (-not (Test-Path -LiteralPath $tauPath)) {
        throw "Missing selected tau file: $tauPath"
    }
    $tauRows = Import-Csv -LiteralPath $tauPath
    $tau = ($tauRows | Where-Object { $_.dataset -eq "amazonpho" }).selected_tau
    if ([string]::IsNullOrWhiteSpace($tau)) {
        throw "Missing selected tau for amazonpho"
    }
    $tauToken = $tau.Replace(".", "p")

    foreach ($regime in $regimes) {
        $output = Join-Path $OutputDir "b1_seed1_4_smoke_methods_$($regime.Name).csv"
        $methods = "frozengrace_gb_mlp_softdist_beta05,mlp_ce,frozengrace_gb_mlp_softdist_beta05_shuffled"
        Write-Output "STEP_START phase=MLP12 stage=B1 methods=smoke_methods regime=$($regime.Name) seeds=1,2,3,4 time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "1,2,3,4" -Output $output

        $output = Join-Path $OutputDir "b1_robust_methods_$($regime.Name)_seeds_${seedToken}.csv"
        $methods = "mlp_small_loss,mlp_coteaching,mlp_jocor,mlp_elr,mlp_sce,mlp_fine,mlp_dividemix_lite,mlp_ct,mlp_anl"
        Write-Output "STEP_START phase=MLP12 stage=B1 methods=robust regime=$($regime.Name) seeds=$Seeds time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Methods $methods -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue $Seeds -Output $output

        $output = Join-Path $OutputDir "b1_logitclip_tau${tauToken}_$($regime.Name)_seeds_1_2_3_4.csv"
        Write-Output "STEP_START phase=MLP12 stage=B1 method=mlp_logitclip tau=$tau regime=$($regime.Name) seeds=1,2,3,4 time=$([DateTime]::Now.ToString('s'))"
        Run-SSL -Methods "mlp_logitclip" -NoiseType $regime.Type -NoiseRate $regime.Rate -SeedsValue "1,2,3,4" -Output $output -Overrides "{`"model.logit_clip_tau`": $tau}"
    }
    Summarize-MLP12
}
else {
    Summarize-MLP12
}

Write-Output "ALL_STEPS_DONE phase=MLP12 stage=$Stage output=$OutputDir time=$([DateTime]::Now.ToString('s'))"
