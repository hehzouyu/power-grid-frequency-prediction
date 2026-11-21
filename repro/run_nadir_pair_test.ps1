param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [switch]$AnchorOnly
)

$ErrorActionPreference = 'Stop'
# Unique run directories preserve settings, partial outputs and failure logs.
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') {
    throw 'Use an alphanumeric run name with optional hyphens and underscores.'
}
$parentDir = Join-Path $PSScriptRoot 'out/nadir-pair-tests'
$runDir = Join-Path $parentDir $RunName
if (Test-Path -LiteralPath $runDir) { throw 'Run directory already exists.' }
New-Item -ItemType Directory -Path $runDir | Out-Null
$predictor = Join-Path $PSScriptRoot 'nadir_pair_predict.py'

function Invoke-Stage {
    param([string]$Name,[string]$Command,[string[]]$Arguments)
    $stageLog = Join-Path $runDir ($Name + '.log')
    $global:LASTEXITCODE = 0
    try {
        & $Command @Arguments *> $stageLog
        $stageExitCode = $LASTEXITCODE
        if ($stageExitCode -ne 0) {
            throw "Stage '$Name' exited $stageExitCode. See $stageLog"
        }
        Add-Content -LiteralPath (Join-Path $runDir 'stage-status.log') -Value "$Name OK"
    } catch {
        Add-Content -LiteralPath (Join-Path $runDir 'stage-status.log') -Value "$Name FAILED $($_.Exception.Message)"
        throw
    }
}

$matlabPath = $PSScriptRoot.Replace("'", "''")
$matlabRun = $runDir.Replace("'", "''")
Invoke-Stage -Name 'anchor-matlab' -Command 'matlab' -Arguments @('-batch',"addpath('$matlabPath'); nadir_pair_test('anchor','$matlabRun')")
Invoke-Stage -Name 'freeze' -Command 'python' -Arguments @($predictor,'freeze',$runDir)
if ($AnchorOnly) { return }

Invoke-Stage -Name 'verify-before-generation' -Command 'python' -Arguments @($predictor,'verify',$runDir)
Invoke-Stage -Name 'generate' -Command 'matlab' -Arguments @('-batch',"addpath('$matlabPath'); nadir_pair_test('generate','$matlabRun')")
Invoke-Stage -Name 'infer' -Command 'python' -Arguments @($predictor,'infer',$runDir)
Invoke-Stage -Name 'verify-before-summary' -Command 'python' -Arguments @($predictor,'verify',$runDir)
Invoke-Stage -Name 'summarize' -Command 'matlab' -Arguments @('-batch',"addpath('$matlabPath'); nadir_pair_test('summarize','$matlabRun')")
