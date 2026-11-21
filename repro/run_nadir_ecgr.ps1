param([Parameter(Mandatory=$true)][string]$RunName,[switch]$Confirm)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') { throw 'Invalid run name' }
$out = Join-Path $PSScriptRoot "out/nadir-ecgr/$RunName"
if (!$Confirm) {
    if (Test-Path -LiteralPath $out) { throw 'Run already exists' }
    New-Item -ItemType Directory -Path $out | Out-Null
    Copy-Item (Join-Path $PSScriptRoot 'nadir_ecgr_protocol.json') (Join-Path $out 'method-protocol.json')
} elseif (Test-Path -LiteralPath (Join-Path $out 'pair-index.csv')) {
    throw 'Confirmatory pairs already generated; preserve run'
}
$script = Join-Path $PSScriptRoot 'nadir_ecgr.py'
$path = $PSScriptRoot.Replace("'","''")
$dir = $out.Replace("'","''")
function Stage {
    param([string]$Name,[string]$Exe,[string[]]$Arg)
    try {
        & $Exe @Arg *> (Join-Path $out "$Name.log")
        if ($LASTEXITCODE -ne 0) { throw "$Name failed, exit $LASTEXITCODE" }
        Add-Content (Join-Path $out 'stages.log') "$Name OK"
    } catch {
        Add-Content (Join-Path $out 'stages.log') "$Name FAILED $($_.Exception.Message)"
        throw
    }
}
if (!$Confirm) {
    Stage 'matlab-development' 'matlab' @('-batch',"addpath('$path'); nadir_ecgr('develop','$dir')")
    Stage 'freeze' 'python' @($script,'freeze',$out)
} else {
    Stage 'verify-before-generation' 'python' @($script,'verify',$out)
    Stage 'matlab-generate' 'matlab' @('-batch',"addpath('$path'); nadir_pair_test('generate','$dir')")
    Stage 'matlab-features' 'matlab' @('-batch',"addpath('$path'); nadir_ecgr('confirm','$dir')")
    Stage 'inference-and-verification' 'python' @($script,'infer',$out)
}
