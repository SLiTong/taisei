#Requires -Version 5.1

$ErrorActionPreference = 'Stop'

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$SubprojectsRoot = Join-Path $RepoRoot 'subprojects'
$ExternalRoot = Join-Path $RepoRoot 'external'

function Test-PathUnder {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')

    return $fullPath.Equals($fullRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $fullPath.StartsWith($fullRoot + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Set-SubprojectJunction {
    param([Parameter(Mandatory = $true)][string]$Name)

    $linkPath = Join-Path $SubprojectsRoot $Name
    $targetPath = Join-Path $ExternalRoot $Name
    $expectedSymlinkText = "../external/$Name"

    if(-not (Test-PathUnder $linkPath $SubprojectsRoot)) {
        throw "Refusing to modify path outside subprojects: $linkPath"
    }

    if(-not (Test-PathUnder $targetPath $ExternalRoot)) {
        throw "Refusing to target path outside external: $targetPath"
    }

    if(-not (Test-Path -LiteralPath $targetPath)) {
        throw "Submodule target does not exist: $targetPath"
    }

    if(Test-Path -LiteralPath $linkPath) {
        $item = Get-Item -Force -LiteralPath $linkPath

        if($item.PSIsContainer -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            $currentTarget = @($item.Target)[0]

            if($currentTarget) {
                $currentTarget = [IO.Path]::GetFullPath($currentTarget)
            }

            if($currentTarget -and $currentTarget.Equals([IO.Path]::GetFullPath($targetPath), [StringComparison]::OrdinalIgnoreCase)) {
                Write-Host "Junction already exists: subprojects/$Name"
                return
            }

            Remove-Item -LiteralPath $linkPath -Force
        } elseif(-not $item.PSIsContainer) {
            $content = (Get-Content -Raw -LiteralPath $linkPath).Trim()

            if($content -ne $expectedSymlinkText) {
                throw "Refusing to replace unexpected file at $linkPath"
            }

            Remove-Item -LiteralPath $linkPath -Force
        } else {
            $children = @(Get-ChildItem -Force -LiteralPath $linkPath)

            if($children.Count -ne 0) {
                throw "Refusing to replace non-empty directory: $linkPath"
            }

            Remove-Item -LiteralPath $linkPath -Force
        }
    }

    New-Item -ItemType Junction -Path $linkPath -Target $targetPath | Out-Null
    Write-Host "Created junction: subprojects/$Name -> external/$Name"
}

Set-SubprojectJunction 'basis_universal'
Set-SubprojectJunction 'koishi'

& git -C $RepoRoot update-index --skip-worktree -- subprojects/basis_universal subprojects/koishi

Write-Host 'Windows build fixes applied.'
