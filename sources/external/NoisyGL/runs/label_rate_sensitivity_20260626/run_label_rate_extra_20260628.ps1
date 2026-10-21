$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$desktop = Join-Path $env:USERPROFILE 'Desktop'
$py = Get-ChildItem -LiteralPath $desktop -Recurse -Filter python.exe -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -like '*GRACE-master*\.venv-rocm\Scripts\python.exe' } |
  Select-Object -First 1 -ExpandProperty FullName
if (-not $py) {
  throw 'Could not locate GRACE-master .venv-rocm python.exe under the user Desktop.'
}
$methods = 'gcn,gcn_xh_ce,frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled'
$datasets = 'cora,citeseer,coauthorcs'

$ratios = @(
  @{train='0.15'; val='0.10'; test='0.75'; tag='0p15'},
  @{train='0.25'; val='0.10'; test='0.65'; tag='0p25'},
  @{train='0.30'; val='0.10'; test='0.60'; tag='0p30'}
)

foreach ($r in $ratios) {
  $out = "$root\runs\label_rate_sensitivity_20260626\label_rate_raw_train$($r.tag).csv"
  Write-Host "RUN_LABEL_RATE_EXTRA train=$($r.train) output=$out"
  & $py "$root\scripts\run_standard_downstream.py" `
    --datasets $datasets `
    --methods $methods `
    --seeds 0,1,2,3,4 `
    --noise_type uniform `
    --noise_rate 0.4 `
    --noise_scope train `
    --device cuda:0 `
    --split_type percent `
    --train_percent $r.train `
    --val_percent $r.val `
    --test_percent $r.test `
    --output $out `
    --stop_on_error
}
