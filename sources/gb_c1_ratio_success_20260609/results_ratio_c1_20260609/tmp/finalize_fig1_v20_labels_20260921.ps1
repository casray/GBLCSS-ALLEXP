$ErrorActionPreference = 'Stop'
$output = Join-Path $PSScriptRoot '..\figures\final_v20_refined_20260921\fig1_v20_refined.drawio'
[xml]$doc = Get-Content -LiteralPath $output -Raw
foreach ($pair in @(@('r142', '1'), @('r168', '2'))) {
    $cell = $doc.SelectSingleNode("//mxCell[@id='$($pair[0])']")
    $cell.SetAttribute('value', ('Projection <i>Z</i><sup>({0})</sup>' -f $pair[1]))
    $cell.SetAttribute('style', $cell.GetAttribute('style').Replace('spacing=6;', 'spacing=2;'))
    $geometry = $cell.SelectSingleNode('./mxGeometry')
    $geometry.SetAttribute('x', '480')
    $geometry.SetAttribute('width', '104')
}
$settings = New-Object System.Xml.XmlWriterSettings
$settings.Indent = $false
$settings.OmitXmlDeclaration = $true
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$writer = [System.Xml.XmlWriter]::Create($output, $settings)
$doc.Save($writer)
$writer.Close()
Write-Output 'Corrected two projection labels in existing v20 sidecar.'
