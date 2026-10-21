$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609'
$source = Join-Path $workspace 'figures\fig1_gblcss_overview_drawio_v20_method_aligned.drawio'
$outputDir = Join-Path $workspace 'figures\final_v20_refined_20260921'
$output = Join-Path $outputDir 'fig1_v20_refined.drawio'

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
[xml]$doc = Get-Content -LiteralPath $source -Raw
$diagramRoot = $doc.SelectSingleNode('//root')

function Get-Cell([string]$id) {
    $cell = $doc.SelectSingleNode("//mxCell[@id='$id']")
    if ($null -eq $cell) { throw "Missing cell: $id" }
    return $cell
}

function Set-Geometry([System.Xml.XmlElement]$cell, [double]$x, [double]$y, [double]$width, [double]$height) {
    $geometry = $cell.SelectSingleNode('./mxGeometry')
    if ($null -eq $geometry) {
        $geometry = $doc.CreateElement('mxGeometry')
        [void]$cell.AppendChild($geometry)
    }
    $geometry.SetAttribute('x', $x.ToString([Globalization.CultureInfo]::InvariantCulture))
    $geometry.SetAttribute('y', $y.ToString([Globalization.CultureInfo]::InvariantCulture))
    $geometry.SetAttribute('width', $width.ToString([Globalization.CultureInfo]::InvariantCulture))
    $geometry.SetAttribute('height', $height.ToString([Globalization.CultureInfo]::InvariantCulture))
    $geometry.SetAttribute('as', 'geometry')
    return $geometry
}

function Add-Point([System.Xml.XmlElement]$parent, [string]$elementName, [double]$x, [double]$y, [string]$asValue) {
    $point = $doc.CreateElement($elementName)
    $point.SetAttribute('x', $x.ToString([Globalization.CultureInfo]::InvariantCulture))
    $point.SetAttribute('y', $y.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($asValue) { $point.SetAttribute('as', $asValue) }
    [void]$parent.AppendChild($point)
}

function Set-EdgeGeometry([System.Xml.XmlElement]$cell, [array]$waypoints, [double]$sourceX, [double]$sourceY, [double]$targetX, [double]$targetY) {
    $geometry = $cell.SelectSingleNode('./mxGeometry')
    if ($null -eq $geometry) {
        $geometry = $doc.CreateElement('mxGeometry')
        [void]$cell.AppendChild($geometry)
    }
    while ($geometry.HasChildNodes) { [void]$geometry.RemoveChild($geometry.FirstChild) }
    $geometry.SetAttribute('relative', '1')
    $geometry.SetAttribute('as', 'geometry')
    if ($waypoints.Count -gt 0) {
        $array = $doc.CreateElement('Array')
        $array.SetAttribute('as', 'points')
        foreach ($point in $waypoints) { Add-Point $array 'mxPoint' $point[0] $point[1] '' }
        [void]$geometry.AppendChild($array)
    }
    Add-Point $geometry 'mxPoint' $sourceX $sourceY 'sourcePoint'
    Add-Point $geometry 'mxPoint' $targetX $targetY 'targetPoint'
}

function Add-Vertex([string]$id, [string]$value, [string]$style, [double]$x, [double]$y, [double]$width, [double]$height) {
    if ($null -ne $doc.SelectSingleNode("//mxCell[@id='$id']")) { throw "Duplicate cell: $id" }
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id', $id)
    $cell.SetAttribute('parent', '1')
    $cell.SetAttribute('style', $style)
    $cell.SetAttribute('value', $value)
    $cell.SetAttribute('vertex', '1')
    [void]$diagramRoot.AppendChild($cell)
    [void](Set-Geometry $cell $x $y $width $height)
    return $cell
}

function Add-Edge([string]$id, [string]$sourceId, [string]$targetId, [string]$style) {
    if ($null -ne $doc.SelectSingleNode("//mxCell[@id='$id']")) { throw "Duplicate cell: $id" }
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id', $id)
    $cell.SetAttribute('parent', '1')
    $cell.SetAttribute('source', $sourceId)
    $cell.SetAttribute('target', $targetId)
    $cell.SetAttribute('style', $style)
    $cell.SetAttribute('value', '')
    $cell.SetAttribute('edge', '1')
    [void]$diagramRoot.AppendChild($cell)
    $geometry = $doc.CreateElement('mxGeometry')
    $geometry.SetAttribute('relative', '1')
    $geometry.SetAttribute('as', 'geometry')
    [void]$cell.AppendChild($geometry)
    return $cell
}

# Stage1: separate the shared encoder, contrastive projection head, and original-graph path.
$encoder = Get-Cell 'r138'
$encoder.SetAttribute('value', '<b>Shared graph<br>encoder</b><br>\(g_\phi\)<br><font color="#64717D">(GRACE)</font>')
$encoder.SetAttribute('style', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#EAF2F8;strokeColor=#6C8EBF;strokeWidth=1.6;fontFamily=Helvetica;fontSize=13;fontColor=#25313C;align=center;verticalAlign=middle;spacing=5;opacity=100;dashed=0;')
Set-Geometry $encoder 365 103 100 90 | Out-Null

$projectionStyle = 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F7F9;strokeColor=#6C8EBF;strokeWidth=1.4;fontFamily=Helvetica;fontSize=12;fontColor=#25313C;align=center;verticalAlign=middle;spacing=5;opacity=100;dashed=0;'
Add-Vertex 'v20_refined_stage1_projection_head' '<b>Projection<br>head</b><br><font color="#64717D">contrastive only</font>' $projectionStyle 365 230 100 90 | Out-Null

$bottomViewEdge = Get-Cell 'YzSOKJwIN_MGii_z3Whu-366'
$bottomViewEdge.SetAttribute('style', 'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#25313C;strokeWidth=1.6;dashed=0;endArrow=blockThin;endFill=1;fontFamily=Helvetica;fontSize=13;fontColor=#25313C;curved=0;')
Set-EdgeGeometry $bottomViewEdge @(@(352,289.86), @(352,174)) 345 289.86 365 174

$toProjection = Add-Edge 'v20_refined_stage1_encoder_to_projection' 'r138' 'v20_refined_stage1_projection_head' 'edgeStyle=none;rounded=0;html=1;strokeColor=#25313C;strokeWidth=1.5;dashed=0;endArrow=blockThin;endFill=1;fontFamily=Helvetica;fontSize=13;fontColor=#25313C;curved=0;exitX=0.5;exitY=1;exitPerimeter=1;entryX=0.5;entryY=0;entryPerimeter=1;'

$topProjectionEdge = Get-Cell 'l140'
$topProjectionEdge.SetAttribute('style', 'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#25313C;strokeWidth=1.6;dashed=0;endArrow=blockThin;endFill=1;fontFamily=Helvetica;fontSize=13;fontColor=#25313C;curved=0;')
Set-EdgeGeometry $topProjectionEdge @(@(480,254), @(480,154)) 465 254 494 154

$bottomProjectionEdge = Get-Cell 'l141'
Set-EdgeGeometry $bottomProjectionEdge @() 465 290 493 290

$z1Label = Get-Cell 'r142'
$z1Label.SetAttribute('value', 'Projection &lt;i&gt;Z&lt;/i&gt;&lt;sup&gt;(1)&lt;/sup&gt;')
Set-Geometry $z1Label 485 89.88 95 22 | Out-Null

$z2Label = Get-Cell 'r168'
$z2Label.SetAttribute('value', 'Projection &lt;i&gt;Z&lt;/i&gt;&lt;sup&gt;(2)&lt;/sup&gt;')
Set-Geometry $z2Label 500 237 85 22 | Out-Null

$originalLabel = Get-Cell 'rCFppp0q5xDDphDL01A_-1'
$originalLabel.SetAttribute('value', 'Pretrained encoder<br>on ORIGINAL graph')
Set-Geometry $originalLabel 365 30 118 42 | Out-Null

$originalPath = Get-Cell 'rCFppp0q5xDDphDL01A_-2'
$originalPath.SetAttribute('source', 'r138')
$originalPath.SetAttribute('target', 'r197')
$originalPath.SetAttribute('style', 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#25313C;strokeWidth=1.4;exitX=0.5;exitY=0;exitPerimeter=1;entryX=0;entryY=0.5;entryPerimeter=1;')
$originalPathPoints = @(@(415,76), @(587,76), @(587,217))
Set-EdgeGeometry $originalPath $originalPathPoints 415 103 600 217

$formulaStyle = 'rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;strokeWidth=0;fontFamily=Helvetica;fontSize=11;fontColor=#25313C;align=center;verticalAlign=middle;spacing=1;opacity=100;dashed=0;'
Add-Vertex 'v20_refined_stage1_H_formula' '\(H=g_\phi(X,A)\)' $formulaStyle 486 33 90 20 | Out-Null
Add-Vertex 'v20_refined_stage1_norm_label' 'RowNorm' $formulaStyle 536 53 52 20 | Out-Null

$stage1Equation = Get-Cell 'xsD4jecQZeTNynng4fUx-3'
$stage1Equation.SetAttribute('value', 'Original-graph path: \(H=g_\phi(X,A)\), \(\bar H=\operatorname{RowNorm}(H)\)')
Set-Geometry $stage1Equation 351 373 446 34 | Out-Null

# Make the Stage2 handoff visibly originate at the frozen representation container.
$stage2Handoff = Get-Cell 'JJJ1YvlXg5No2yYtv6KG-3'
$stage2Handoff.SetAttribute('source', 'r197')
$stage2Handoff.SetAttribute('style', 'edgeStyle=none;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#25313C;strokeWidth=1.5;dashed=0;endArrow=blockThin;endFill=1;fontFamily=Helvetica;fontSize=13;fontColor=#25313C;curved=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;exitPerimeter=1;')
Set-EdgeGeometry $stage2Handoff @() 799 217 835 218.42

# Preserve the requested terminology consistently in the existing figure labels.
$stage2Title = Get-Cell 'r9'
$stage2Title.SetAttribute('value', 'Label-agnostic granular ball partition')
$finalPartitionTitle = Get-Cell 'r358'
$finalPartitionTitle.SetAttribute('value', '<b>Final granular ball partition in \(\bar H\)</b>')

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Indent = $false
$settings.OmitXmlDeclaration = $true
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$writer = [System.Xml.XmlWriter]::Create($output, $settings)
$doc.Save($writer)
$writer.Close()

Write-Output $output
