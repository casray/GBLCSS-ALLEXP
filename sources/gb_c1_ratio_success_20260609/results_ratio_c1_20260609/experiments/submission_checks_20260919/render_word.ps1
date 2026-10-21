$ErrorActionPreference = 'Stop'
$qa = Join-Path $PSScriptRoot 'word_qa'
New-Item -ItemType Directory -Path $qa -Force | Out-Null
$app = New-Object -ComObject 'KWPS.Application'
$initialDocuments = $app.Documents.Count
$oldAlerts = $app.DisplayAlerts
$oldVisible = $app.Visible
try {
    $app.DisplayAlerts = 0
    if ($initialDocuments -eq 0) { $app.Visible = $false }
    $inputs = @(
        @{ Source = 'C:\Users\Admin\Downloads\GB-LCSS_experiments_revised_20260919.docx'; Stem = 'main' },
        @{ Source = 'C:\Users\Admin\Downloads\GB-LCSS_supplementary_experiments_20260919.docx'; Stem = 'supplement' }
    )
    foreach ($item in $inputs) {
        $document = $null
        try {
            $document = $app.Documents.Open($item.Source, $false, $true, $false)
            $document.Repaginate()
            $pdf = Join-Path $qa ($item.Stem + '.pdf')
            $document.ExportAsFixedFormat($pdf, 17)
            Write-Output "Exported $pdf"
        }
        finally {
            if ($null -ne $document) {
                $document.Close(0)
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
            }
        }
    }
}
finally {
    $app.DisplayAlerts = $oldAlerts
    $app.Visible = $oldVisible
    if ($initialDocuments -eq 0) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
