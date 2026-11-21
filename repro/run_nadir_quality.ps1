param([Parameter(Mandatory=$true)][ValidateSet('Development','Confirmation','Audit')][string]$Stage)
$ErrorActionPreference='Stop'
$work=Split-Path -Parent $PSScriptRoot
$folder=Join-Path $PSScriptRoot 'out/nadir-quality'
$quotedRepro=$PSScriptRoot.Replace("'","''")
function Invoke-QualityStep {
    param($Executable,[string[]]$Arguments,$Log)
    & $Executable @Arguments *> $Log
    if ($LASTEXITCODE -ne 0) { throw "Execution failed. See $Log" }
}
Push-Location $work
try {
    if ($Stage -eq 'Audit') {
        $existing=Join-Path $folder 'confirm'
        $auditLog=Join-Path $existing ('audit-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')+'.log')
        Invoke-QualityStep python @('repro/verify_plot_nadir_quality.py') $auditLog
        return
    }
    $name = if ($Stage -eq 'Development') { 'dev' } else { 'confirm' }
    $run=Join-Path $folder $name
    if (Test-Path -LiteralPath $run) { throw 'Run already exists. Original archive is preserved.' }
    if ($name -eq 'confirm') {
        & python repro/nadir_quality.py verify
        if ($LASTEXITCODE -ne 0) { throw 'Frozen method verification failed' }
    }
    & python repro/prepare_nadir_quality.py $name
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize run' }
    $quotedRun=$run.Replace("'","''")
    Invoke-QualityStep matlab @('-batch',"addpath('$quotedRepro'); nadir_quality('$quotedRun')") (Join-Path $run 'matlab.log')
    $operation=if ($name -eq 'dev') { 'develop' } else { 'confirm' }
    Invoke-QualityStep python @('repro/nadir_quality.py',$operation) (Join-Path $run 'analysis.log')
} finally {
    Pop-Location
}
