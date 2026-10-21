param(
    [ValidateSet('geometry', 'smoke', 'five', 'all')]
    [string]$Phase = 'all'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $RepoRoot)
$Python = $env:MLP24_PYTHON
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw 'Set MLP24_PYTHON to the existing ROCm Python executable before running this script.'
}
$EmbeddingRoot = Join-Path $ProjectRoot 'gb_graph_aware_minival_6_1\checkpoints\embeddings_v2'
$RunDir = Join-Path $RepoRoot 'runs\mlp24_scaleadaptive_balls_20260712'
$Datasets = 'cora,citeseer,corafull,coauthorcs,amazonpho,citeseerfull,coauthorph,pubmed,amazoncom,coraml,attribwiki'

New-Item -ItemType Directory -Force $RunDir | Out-Null
Set-Location $RepoRoot

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Invoke-Downstream {
    param(
        [string]$Methods,
        [string]$Seeds,
        [string]$Output,
        [string]$Policy = ''
    )
    $arguments = @(
        'scripts\run_ssl_swap_downstream.py',
        '--embedding_root', $EmbeddingRoot,
        '--datasets', $Datasets,
        '--ssl_methods', 'grace',
        '--base_methods', $Methods,
        '--seeds', $Seeds,
        '--noise_type', 'uniform',
        '--noise_rate', '0.4',
        '--noise_scope', 'train',
        '--device', 'cuda:0',
        '--split_type', 'percent',
        '--train_percent', '0.2',
        '--val_percent', '0.1',
        '--test_percent', '0.7',
        '--output', $Output
    )
    if ($Policy) {
        $arguments += @('--gb_max_balls_policy', $Policy)
    }
    Invoke-CheckedPython $arguments
}

function Invoke-Summary {
    Invoke-CheckedPython @(
        'scripts\summarize_mlp24_scaleadaptive_balls_20260712.py',
        '--run_dir', $RunDir
    )
}

function Assert-Gate {
    param([string]$Stage)
    $gate = Import-Csv (Join-Path $RunDir 'mlp24_gate_decisions.csv') |
        Where-Object { $_.stage -eq $Stage } |
        Select-Object -Last 1
    if (-not $gate -or $gate.pass -ne '1') {
        $reason = if ($gate) { $gate.reasons } else { 'missing gate row' }
        throw "MLP-24 stage '$Stage' did not pass: $reason"
    }
}

if ($Phase -in @('geometry', 'all')) {
    Invoke-CheckedPython @(
        '-u', 'scripts\diagnose_mlp24_scaleadaptive_balls_20260712.py',
        '--embedding_root', $EmbeddingRoot,
        '--datasets', $Datasets,
        '--seed', '0',
        '--noise_type', 'uniform',
        '--noise_rate', '0.4',
        '--noise_scope', 'train',
        '--split_type', 'percent',
        '--train_percent', '0.2',
        '--val_percent', '0.1',
        '--test_percent', '0.7',
        '--output', (Join-Path $RunDir 'mlp24_geometry.csv')
    )
    Invoke-Summary
    if ($Phase -eq 'geometry') { exit 0 }
    Assert-Gate 'geometry'
}

if ($Phase -in @('smoke', 'all')) {
    Invoke-Downstream 'frozengrace_gb_mlp_softdist_beta05' '0' (Join-Path $RunDir 'smoke_fixed_seed0.csv') 'fixed'
    Invoke-Downstream 'frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled' '0' (Join-Path $RunDir 'smoke_adaptive_seed0.csv') 'sqrt_nodes'
    Invoke-Downstream 'mlp_jocor' '0' (Join-Path $RunDir 'smoke_jocor_seed0.csv')
    Invoke-Summary
    if ($Phase -eq 'smoke') { exit 0 }
    Assert-Gate 'seed0_smoke'
}

if ($Phase -in @('five', 'all')) {
    Invoke-Downstream 'frozengrace_gb_mlp_softdist_beta05' '1,2,3,4' (Join-Path $RunDir 'five_fixed_seeds_1_4.csv') 'fixed'
    Invoke-Downstream 'frozengrace_gb_mlp_softdist_beta05,frozengrace_gb_mlp_softdist_beta05_shuffled' '1,2,3,4' (Join-Path $RunDir 'five_adaptive_seeds_1_4.csv') 'sqrt_nodes'
    Invoke-Downstream 'mlp_jocor' '1,2,3,4' (Join-Path $RunDir 'five_jocor_seeds_1_4.csv')
    Invoke-Summary
    Assert-Gate 'uniform0p4_5seed'
}

Write-Host 'MLP-24 preregistered uniform-0.4 phases completed. Phase D is intentionally separate and is allowed only after the five-seed gate passes.'
