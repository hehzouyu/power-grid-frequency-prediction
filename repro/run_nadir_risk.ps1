param([Parameter(Mandatory=$true)][ValidateSet('Development','Confirmation','Audit')][string]$Stage)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$out=Join-Path $PSScriptRoot 'out/nadir-risk'
function Step {
    param($Exe,[string[]]$ArgsList,$Log)
    & $Exe @ArgsList *> $Log
    if ($LASTEXITCODE -ne 0) { throw "Failed stage. Log: $Log" }
}
Push-Location $project
try {
    if ($Stage -eq 'Audit') {
        Step python @('repro/verify_plot_nadir_risk.py') (Join-Path $out ('audit-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')+'.log'))
        return
    }
    $tag=$Stage.ToLowerInvariant()
    if ($Stage -eq 'Development') {
        if (Test-Path -LiteralPath $out) { throw 'Preserve the existing archive.' }
        New-Item -ItemType Directory $out | Out-Null
        Copy-Item (Join-Path $PSScriptRoot 'nadir_risk_protocol.json') (Join-Path $out 'protocol.json')
    } else {
        if (Test-Path -LiteralPath (Join-Path $out 'confirmation.csv')) { throw 'Preserve the existing confirmation.' }
        Step python @('repro/nadir_risk.py','verify') (Join-Path $out 'verify-before-confirmation.log')
    }
    $p=$PSScriptRoot.Replace("'","''")
    $q=$out.Replace("'","''")
    Step matlab @('-batch',"addpath('$p'); nadir_risk('$tag','$q')") (Join-Path $out "matlab-$tag.log")
    $action=if ($tag -eq 'development') { 'freeze' } else { 'confirm' }
    Step python @('repro/nadir_risk.py',$action) (Join-Path $out "$action.log")
} finally {
    Pop-Location
}
