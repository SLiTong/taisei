param(
    [ValidateSet('1', '2', '3', '4', '5', '6')]
    [string]$Stage = '1',

    [ValidateSet('reimuA', 'marisaA', 'youmuA')]
    [string]$Shot = 'reimuA',

    [ValidateSet('auto', 'boss', 'pre-boss')]
    [string]$Bookmark = 'auto',

    [ValidateSet('Easy', 'Normal', 'Hard', 'Lunatic')]
    [string]$Difficulty = 'Easy',

    [int]$Width = 800,
    [int]$Height = 600,

    [switch]$KillExisting
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$Exe = Join-Path $RepoRoot 'dist\windows-clean\taisei.exe'
$WorkingDirectory = Join-Path $RepoRoot 'dist\windows-clean'
$StoragePath = Join-Path $RepoRoot 'dist\redraw-storage-nopause'

if(!(Test-Path -LiteralPath $Exe)) {
    throw "Taisei executable not found: $Exe"
}

if($KillExisting) {
    Get-Process taisei -ErrorAction SilentlyContinue | Stop-Process
}

if($Bookmark -eq 'auto') {
    if($Stage -in @('3', '5')) {
        $Bookmark = 'pre-boss'
    } else {
        $Bookmark = 'boss'
    }
}

$env:TAISEI_STORAGE_PATH = $StoragePath

$Args = @(
    '--unlock-all',
    '--play',
    '-i', $Stage,
    '-d', $Difficulty,
    '-s', $Shot,
    '--skip-to-bookmark', $Bookmark,
    '--width', $Width,
    '--height', $Height
)

$Process = Start-Process -FilePath $Exe -WorkingDirectory $WorkingDirectory -ArgumentList $Args -PassThru

[PSCustomObject]@{
    Pid = $Process.Id
    Stage = $Stage
    Shot = $Shot
    Bookmark = $Bookmark
    Difficulty = $Difficulty
    Exe = $Exe
    StoragePath = $StoragePath
}
