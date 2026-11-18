<#
.SYNOPSIS
    Install BlackPy into .venv and run every check on Windows, then write a short summary.

.DESCRIPTION
    Steps: environment report, virtual environment and editable install, import check,
    ruff (lint and format), pytest (quick subset by default), mkdocs build --strict and,
    optionally, every script in examples\.

    Full logs go to build\windows-check\ (gitignored). The file
    build\windows-check\SUMMARY.txt lists every step as PASS or FAIL with the failing
    tests and the end of each failing log; it is short enough to paste into a chat.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check_windows.ps1
    Quick subset (about the same as CI on Windows).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check_windows.ps1 -Full -Examples
    Full test suite (slow tests included, may take 30 minutes or more) and all examples.
#>
param(
    [switch]$Full,             # full test suite instead of the quick subset
    [switch]$Examples,         # also run every script in examples\
    [switch]$SkipInstall,      # reuse the existing .venv as it is
    [int]$Workers = 4,         # pytest-xdist workers
    [string]$PythonVersion = "3.12"   # used with the py launcher to create .venv
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "build\windows-check"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Get-ChildItem $logDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
$summaryPath = Join-Path $logDir "SUMMARY.txt"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

$env:MPLBACKEND = "Agg"
# Console output only. File reads and writes keep the Windows default encoding, so
# encoding bugs in BlackPy itself still show up.
$env:PYTHONIOENCODING = "utf-8"

$results = New-Object System.Collections.Generic.List[string]
$details = New-Object System.Collections.Generic.List[string]

function Invoke-Step {
    param([string]$Name, [string]$Exe, [string]$Arguments, [string]$LogName)
    $log = Join-Path $logDir $LogName
    Write-Host "==> $Name" -ForegroundColor Cyan
    $start = Get-Date
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    try {
        $p = [System.Diagnostics.Process]::Start($psi)
        $out = $p.StandardOutput.ReadToEndAsync()
        $err = $p.StandardError.ReadToEndAsync()
        $p.WaitForExit()
        $text = $out.Result + $err.Result
        $code = $p.ExitCode
    } catch {
        $text = "Could not start '$Exe $Arguments': $_"
        $code = -1
    }
    [System.IO.File]::WriteAllText($log, $text, [System.Text.Encoding]::UTF8)
    $secs = [int]((Get-Date) - $start).TotalSeconds
    $status = if ($code -eq 0) { "PASS" } else { "FAIL" }
    $color = if ($code -eq 0) { "Green" } else { "Red" }
    Write-Host "    $status ($secs s)" -ForegroundColor $color
    $script:results.Add(("{0}  {1}  (exit {2}, {3} s, log: {4})" -f $status, $Name, $code, $secs, $LogName))
    if ($code -ne 0) {
        $lines = $text -split "`r?`n"
        $failed = $lines | Where-Object { $_ -match '^(FAILED|ERROR) ' }
        $script:details.Add("---- $Name ----")
        if ($failed) {
            $failed | Select-Object -First 60 | ForEach-Object { $script:details.Add($_) }
            $lines | Where-Object { $_ -match '(passed|failed|error).* in [0-9.]+s' } |
                Select-Object -Last 1 | ForEach-Object { $script:details.Add($_) }
        } else {
            $lines | Where-Object { $_.Trim() } | Select-Object -Last 25 |
                ForEach-Object { $script:details.Add($_) }
        }
    }
    return $code
}

# ---------------------------------------------------------------- environment
$os = (Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue)
$header = @(
    "BlackPy Windows check, $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "OS: $($os.Caption) $($os.Version) ($env:PROCESSOR_ARCHITECTURE), $env:NUMBER_OF_PROCESSORS logical CPUs",
    "Git commit: $(git rev-parse --short HEAD 2>$null)   core.autocrlf: $(git config core.autocrlf 2>$null)",
    "Mode: $(if ($Full) { 'full suite' } else { 'quick subset' })$(if ($Examples) { ' + examples' })"
)

# ---------------------------------------------------------------- install
if (-not $SkipInstall) {
    if (-not (Test-Path $venvPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-Step "Create .venv (Python $PythonVersion)" "py" "-$PythonVersion -m venv .venv" "01_venv.log" | Out-Null
        } else {
            Invoke-Step "Create .venv" "python" "-m venv .venv" "01_venv.log" | Out-Null
        }
    }
    if (-not (Test-Path $venvPython)) {
        $results.Add("FAIL  .venv was not created; install Python $PythonVersion from python.org and run again")
    } else {
        Invoke-Step "Upgrade pip" $venvPython "-m pip install --upgrade pip" "02_pip.log" | Out-Null
        Invoke-Step "Install BlackPy (.[dev,docs])" $venvPython "-m pip install -e `".[dev,docs]`"" "03_install.log" | Out-Null
    }
}

if (Test-Path $venvPython) {
    $header += "Python: $(& $venvPython --version 2>&1)"
    & $venvPython -m pip freeze 2>$null | Out-File (Join-Path $logDir "pip_freeze.txt") -Encoding utf8

    # ------------------------------------------------------------ checks
    $import = "import jax, blackpy as bp; print('blackpy', bp.__version__, 'jax', jax.__version__, jax.devices()); " +
              "print('Kerr ISCO a=0.6:', float(bp.orbits.isco(bp.metric.Kerr(), a=0.6))); " +
              "print(bp.data.load('eht/sgra').id)"
    Invoke-Step "Import check" $venvPython "-c `"$import`"" "04_import.log" | Out-Null
    Invoke-Step "ruff check" $venvPython "-m ruff check src tests examples" "05_ruff_check.log" | Out-Null
    Invoke-Step "ruff format --check" $venvPython "-m ruff format --check src tests examples" "06_ruff_format.log" | Out-Null

    $marker = if ($Full) { "" } else { "-m `"not slow`" " }
    Write-Host "    (tests can take several minutes; nothing is printed until they finish)"
    Invoke-Step "pytest ($(if ($Full) { 'full suite' } else { 'quick subset' }))" $venvPython `
        "-m pytest $marker-n $Workers --dist loadfile -q -rfE --durations=15" "07_pytest.log" | Out-Null

    Invoke-Step "mkdocs build --strict" $venvPython "-m mkdocs build --strict -d build\windows-check\site" "08_mkdocs.log" | Out-Null

    if ($Examples) {
        Get-ChildItem (Join-Path $root "examples") -Filter "*.py" | Where-Object { $_.Name -match '^\d' } |
            Sort-Object Name | ForEach-Object {
            Invoke-Step "example $($_.Name)" $venvPython "`"$($_.FullName)`"" "09_example_$($_.BaseName).log" | Out-Null
        }
    }
}

# ---------------------------------------------------------------- summary
$nFail = ($results | Where-Object { $_ -like "FAIL*" }).Count
$summary = @($header) + "" + "Steps:" + $results + "" +
    $(if ($nFail -eq 0) { "All steps passed." } else { "$nFail step(s) failed. Details:" }) + $details
$summary | Out-File $summaryPath -Encoding utf8

Write-Host ""
$summary | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "Summary: $summaryPath" -ForegroundColor Yellow
Write-Host "Full logs: $logDir" -ForegroundColor Yellow
