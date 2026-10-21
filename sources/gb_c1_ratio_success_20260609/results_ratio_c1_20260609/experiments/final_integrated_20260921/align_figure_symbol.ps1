$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$path = Join-Path $root 'figures\final_v20_refined_20260921\fig1_v20_refined.drawio'
[xml]$doc = [System.IO.File]::ReadAllText($path)
$symbolCount = 0
foreach ($cell in $doc.SelectNodes('//mxCell[@value]')) {
    $value = $cell.GetAttribute('value')
    if ($value.Contains('g_\phi')) {
        $symbolCount += ([regex]::Matches($value, [regex]::Escape('g_\phi'))).Count
        $cell.SetAttribute('value', $value.Replace('g_\phi', 'g_\psi'))
    }
    $value = $cell.GetAttribute('value')
    if ($value.Contains('on ORIGINAL graph')) {
        $cell.SetAttribute('value', $value.Replace('on ORIGINAL graph', 'on original graph'))
    }
}
if ($symbolCount -ne 3) { throw "Expected 3 encoder symbols, found $symbolCount" }
$settings = [System.Xml.XmlWriterSettings]::new()
$settings.Encoding = [System.Text.UTF8Encoding]::new($false)
$settings.Indent = $false
$settings.OmitXmlDeclaration = $true
$writer = [System.Xml.XmlWriter]::Create($path, $settings)
try { $doc.Save($writer) } finally { $writer.Dispose() }
Write-Output 'Aligned three encoder labels with Eq. (2): g_psi.'
