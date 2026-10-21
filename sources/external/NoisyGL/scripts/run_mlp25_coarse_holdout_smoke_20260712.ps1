$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $RepoRoot)
$Python = $env:MLP25_PYTHON
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw 'Set MLP25_PYTHON to the existing ROCm Python executable.'
}
$EmbeddingRoot = Join-Path $ProjectRoot 'gb_graph_aware_minival_6_1\checkpoints\embeddings_v2'
$RunDir = Join-Path $RepoRoot 'runs\mlp25_coarse_holdout_smoke_20260712'
$Real = 'frozengrace_gb_mlp_softdist_beta05'
$Shuffled = 'frozengrace_gb_mlp_softdist_beta05_shuffled'

New-Item -ItemType Directory -Force $RunDir | Out-Null
Set-Location $RepoRoot

function Invoke-Run {
    param(
        [string]$Datasets,
        [string]$Methods,
        [int]$MaxBalls,
        [int]$MinBallSize,
        [double]$SplitImprovement,
        [string]$Output
    )
    & $Python scripts\run_ssl_swap_downstream.py `
        --embedding_root $EmbeddingRoot `
        --datasets $Datasets `
        --ssl_methods grace `
        --base_methods $Methods `
        --seeds 0 `
        --noise_type uniform `
        --noise_rate 0.4 `
        --noise_scope train `
        --device cuda:0 `
        --split_type percent `
        --train_percent 0.2 `
        --val_percent 0.1 `
        --test_percent 0.7 `
        --gb_max_balls_policy fixed `
        --gb_max_balls $MaxBalls `
        --gb_min_ball_size $MinBallSize `
        --gb_split_improvement $SplitImprovement `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "MLP-25 downstream command failed with exit code $LASTEXITCODE"
    }
}

$Holdouts = 'citeseerfull,coraml,attribwiki'
Invoke-Run $Holdouts $Real 64 32 0.05 (Join-Path $RunDir 'fixed_seed0.csv')
Invoke-Run 'citeseerfull' "$Real,$Shuffled" 43 32 0.05 (Join-Path $RunDir 'vote_citeseerfull_seed0.csv')
Invoke-Run 'coraml' "$Real,$Shuffled" 30 32 0.05 (Join-Path $RunDir 'vote_coraml_seed0.csv')
Invoke-Run 'attribwiki' "$Real,$Shuffled" 25 32 0.05 (Join-Path $RunDir 'vote_attribwiki_seed0.csv')
Invoke-Run $Holdouts "$Real,$Shuffled" 128 64 0.01 (Join-Path $RunDir 'candidate_seed0.csv')

& $Python scripts\summarize_mlp25_coarse_holdout_smoke_20260712.py --run_dir $RunDir
if ($LASTEXITCODE -ne 0) {
    throw "MLP-25 summarizer failed with exit code $LASTEXITCODE"
}
