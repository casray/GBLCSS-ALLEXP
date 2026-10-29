$ErrorActionPreference = 'Stop'
$runRoot = 'D:\codex_temp\graphecl_honest_pilot_20261029'
$codeRoot = 'C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609\experiments\graphecl_honest_pilot_20261029'
$pythonExe = 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe'
function Save-LaunchState($state, $code) {
    @{ stage=$state; exit_code=$code; pid=$PID; time=(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'launcher_status.json') -Encoding utf8
}
try {
    Save-LaunchState 'RESUMING' $null
    & $pythonExe -B (Join-Path $codeRoot 'crossfit.py') run
    if ($LASTEXITCODE -ne 0) { throw "crossfit.py exited with $LASTEXITCODE" }
    Save-LaunchState 'ANALYZING' 0
    & $pythonExe -B (Join-Path $codeRoot 'analyze.py')
    if ($LASTEXITCODE -ne 0) { throw "analyze.py exited with $LASTEXITCODE" }
    Save-LaunchState 'COMPLETE' 0
} catch {
    Save-LaunchState 'FAILED' 1
    Write-Error $_
    exit 1
}
